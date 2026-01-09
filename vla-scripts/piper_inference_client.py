#!/usr/bin/env python3
"""Piper client that can stream from live camera or an HDF5 teleoperation file."""

import argparse
import sys
import time
import logging
from threading import Event
from pathlib import Path
from typing import Optional, Sequence, Tuple

import base64
import cv2
import requests
import h5py
import numpy as np

from piper_sdk import C_PiperInterface_V2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_actions(all_actions):
    """
    Convert HDF5 actions/qpos into Piper units.

    Args:
        all_actions: Action tensor/list/ndarray (7,)

    Returns:
        dict: joint_angles (list[int]), gripper_distance (int), raw_action (list[float])
    """
    try:
        import torch  # type: ignore
        is_tensor = isinstance(all_actions, torch.Tensor)
    except ImportError:
        is_tensor = False

    action = all_actions.detach().cpu().numpy() if is_tensor else np.asarray(all_actions)
    factor = 1000 * 180.0 / np.pi  # radians -> Piper units

    joint_angles = np.round(action[0:6] * factor).astype(int).tolist()
    gripper_distance = int(np.abs(action[6]) * 1000 * 1000)  # normalized -> Piper units

    return {
        "joint_angles": joint_angles,
        "gripper_distance": gripper_distance,
        "raw_action": action.tolist(),
    }


class PiperClient:
    def __init__(
        self,
        task_instruction: str,
        server_url: str,
        can_interface: str = "can0",
        camera_port: int = 0,
        save_frames_dir: Optional[str] = None,
        hdf5_path: Optional[Path] = None,
        use_hdf5_qpos: bool = False,
    ):
        """
        Initialize Piper client for remote inference.

        Args:
            task_instruction: Task instruction
            server_url: URL of inference server (e.g., http://localhost:8000)
            can_interface: CAN interface name
            camera_port: Camera port/device (ignored when hdf5_path is provided)
            save_frames_dir: Directory to save frames (default: None for display mode)
            hdf5_path: Optional HDF5 file to stream frames instead of a live camera
            use_hdf5_qpos: If True, use qpos from HDF5, reset arm to initial qpos, and stream qpos to server
        """
        self.task_instruction = task_instruction
        self.server_url = server_url.rstrip("/")
        self.save_frames_dir = save_frames_dir
        self.hdf5_path = Path(hdf5_path) if hdf5_path else None
        self.use_hdf5_qpos = use_hdf5_qpos

        # HDF5 members
        self.h5_file: Optional[h5py.File] = None
        self.h5_cam_ds = None
        self.h5_qpos_ds = None
        self.h5_length: int = 0
        self.h5_index: int = 0
        self.last_hdf5_qpos: Optional[Sequence[float]] = None

        # Create save directory if specified
        if self.save_frames_dir:
            save_path = Path(self.save_frames_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Frame saving directory: {save_path.absolute()}")

        # Initialize Piper arm via CAN interface
        logger.info(f"Initializing Piper arm on CAN interface: {can_interface}")
        self.piper = C_PiperInterface_V2(can_interface)
        self.piper.ConnectPort()
        logger.info("Piper CAN port connected")

        # Enable Piper
        logger.info("Enabling Piper arm...")
        retry_count = 0
        while not self.piper.EnablePiper() and retry_count < 100:
            time.sleep(0.01)
            retry_count += 1
        if retry_count >= 100:
            raise RuntimeError("Failed to enable Piper arm after 100 retries")
        logger.info("Piper arm enabled successfully")

        # Load HDF5 or camera
        if self.hdf5_path:
            self._load_hdf5(self.hdf5_path)
        else:
            self._init_camera(camera_port)

        # Control flags
        self.running = Event()
        self.running.set()

        # Check server health
        self._check_server_health()

        # Reset arm to initial qpos if requested
        if self.hdf5_path and self.use_hdf5_qpos:
            self._reset_arm_to_initial_qpos()

    def _init_camera(self, camera_port: int) -> None:
        logger.info(f"Initializing camera on port: {camera_port}")
        self.cap = cv2.VideoCapture(camera_port)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera on port {camera_port}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _load_hdf5(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file '{path}' not found")
        logger.info(f"Loading HDF5 frames from: {path}")
        self.h5_file = h5py.File(path, "r")
        try:
            self.h5_cam_ds = self.h5_file["observations/images/cam_high"]
            self.h5_qpos_ds = self.h5_file["observations/qpos"]
        except KeyError as e:
            raise KeyError(
                "HDF5 file missing required dataset. Expected 'observations/images/cam_high' "
                "and 'observations/qpos'."
            ) from e

        if self.h5_cam_ds.shape[0] != self.h5_qpos_ds.shape[0]:
            raise ValueError(
                f"HDF5 dataset length mismatch: cam_high={self.h5_cam_ds.shape[0]}, "
                f"qpos={self.h5_qpos_ds.shape[0]}"
            )
        self.h5_length = self.h5_cam_ds.shape[0]
        logger.info(f"HDF5 contains {self.h5_length} frames")

        # Placeholder for camera when using HDF5
        self.cap = None

    def _check_server_health(self) -> None:
        """Check if server is alive."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("policy_loaded"):
                    logger.info("Server is healthy and policy is loaded")
                else:
                    logger.warning("Server is healthy but policy is not loaded")
            else:
                logger.error(f"Server health check failed: status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to server at {self.server_url}: {e}")
            raise RuntimeError(f"Cannot connect to inference server at {self.server_url}")

    def get_camera_frame(self):
        """Capture frame from camera."""
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None
        return frame

    def _get_hdf5_frame(self) -> Optional[np.ndarray]:
        if self.h5_index >= self.h5_length:
            return None
        frame_rgb = np.asarray(self.h5_cam_ds[self.h5_index])
        # Ensure uint8 contiguous for OpenCV; normalize floats to 0-255
        if frame_rgb.dtype != np.uint8:
            if np.issubdtype(frame_rgb.dtype, np.floating):
                frame_rgb = np.clip(frame_rgb, 0.0, 1.0) * 255.0
            frame_rgb = np.clip(frame_rgb, 0, 255).round().astype(np.uint8, copy=False)
        # Convert RGB -> BGR for OpenCV encode/display
        frame_bgr = frame_rgb[..., ::-1].copy()
        self.h5_index += 1
        return frame_bgr

    def frame_to_base64(self, frame):
        """Convert frame to base64 encoded string."""
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            image_base64 = base64.b64encode(buffer).decode("utf-8")
            return image_base64
        except Exception as e:
            logger.error(f"Failed to encode frame: {e}")
            return None

    def get_proprioception(self):
        """
        Get current joint state from robot
        """
        try:
            joint_feedback = self.piper.GetArmJointMsgs()
            gripper_feedback = self.piper.GetArmGripperMsgs()
            joint_state = [
                joint_feedback.joint_state.joint_1,
                joint_feedback.joint_state.joint_2,
                joint_feedback.joint_state.joint_3,
                joint_feedback.joint_state.joint_4,
                joint_feedback.joint_state.joint_5,
                joint_feedback.joint_state.joint_6,
                gripper_feedback.gripper_state.grippers_angle
            ]
            return joint_state
        except Exception as e:
            logger.warning(f"Failed to get joint feedback: {e}, using zero proprioception")
            return [0, 0, 0, 0, 0, 0, 0]


    def request_inference(self, frame, proprioception):
        """
        Send inference request to server.

        Args:
            frame (numpy.ndarray): BGR frame
            proprioception (list): Joint state [j0-j5, gripper]

        Returns:
            dict: Response data or None if failed
        """
        try:
            image_base64 = self.frame_to_base64(frame)
            if image_base64 is None:
                return None

            request_data = {
                "image": image_base64,
                "task_instruction": self.task_instruction,
                "proprioception": proprioception,
            }

            start_time = time.time()
            response = requests.post(f"{self.server_url}/infer", json=request_data, timeout=30)
            request_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"Inference success (time: {request_time:.3f}s, FPS: {1/request_time:.1f})")
                    return data
                else:
                    logger.error(f"Server inference failed: {data.get('error')}")
                    return None
            else:
                logger.error(f"Server error: status {response.status_code}, {response.text}")
                return None

        except requests.Timeout:
            logger.error("Inference request timeout (30s)")
            return None
        except requests.ConnectionError:
            logger.error(f"Failed to connect to server at {self.server_url}")
            return None
        except Exception as e:
            logger.error(f"Inference request failed: {e}")
            return None

    def _split_joint_and_gripper(self, seq: Sequence[float]) -> Tuple[Sequence[float], float]:
        if len(seq) < 6:
            raise ValueError(f"Expected at least 6 joint values, got {len(seq)}")
        joints = seq[:6]
        gripper = seq[6] if len(seq) > 6 else 0
        return joints, gripper

    def control_robot(self, joint_angles, gripper_distance):
        """
        Send control commands to Piper robot.

        Args:
            joint_angles (list): Joint angles [j0-j5] in Piper units
            gripper_distance (int): Gripper distance in Piper units
        """
        try:
            j0, j1, j2, j3, j4, j5 = joint_angles

            logger.info(f"Controlling joints: j0={j0}, j1={j1}, j2={j2}, j3={j3}, j4={j4}, j5={j5}")

            # Set motion control mode and speed
            self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)

            # Send joint control command
            self.piper.JointCtrl(j0, j1, j2, j3, j4, j5)

            # Control gripper
            logger.info(f"Controlling gripper: distance={gripper_distance}")

            self.piper.GripperCtrl(gripper_distance, 1000, 0x01, 0)

            arm_status = self.piper.GetArmStatus()
            logger.debug(f"Arm status: {arm_status}")

        except Exception as e:
            logger.error(f"Failed to control robot: {e}")

    def _reset_arm_to_initial_qpos(self) -> None:
        initial_qpos = self.h5_qpos_ds[0].tolist()
        cmd = parse_actions(initial_qpos)
        logger.info(
            f"Resetting arm to initial qpos from HDF5: joints={cmd['joint_angles']}, "
            f"gripper={cmd['gripper_distance']}"
        )
        self.control_robot(cmd["joint_angles"], cmd["gripper_distance"])
        time.sleep(0.5)

    def _next_frame_and_proprio(self):
        """
        Fetch next frame and proprioception based on source mode.

        Returns:
            (frame, proprioception) or (None, None) when HDF5 is exhausted.
        """
        if self.hdf5_path:
            frame = self._get_hdf5_frame()
            if frame is None:
                return None, None
            self.last_hdf5_qpos = self.h5_qpos_ds[self.h5_index - 1].tolist()
            proprioception = self.last_hdf5_qpos if self.use_hdf5_qpos else self.get_proprioception()
            return frame, proprioception

        self.last_hdf5_qpos = None
        frame = self.get_camera_frame()
        if frame is None:
            return None, None
        proprioception = self.get_proprioception()
        return frame, proprioception

    def run(self):
        """Main inference loop."""
        logger.info(f"Starting client for task: {self.task_instruction}")
        logger.info(f"Server URL: {self.server_url}")
        if self.save_frames_dir:
            logger.info(f"Saving frames to: {self.save_frames_dir}")
        elif self.hdf5_path:
            logger.info("HDF5 mode: not using live camera")
        else:
            logger.info("Press 'q' to quit")

        frame_count = 0
        inference_count = 0
        failed_count = 0

        try:
            while self.running.is_set():
                frame, proprioception = self._next_frame_and_proprio()
                if frame is None:
                    if self.hdf5_path:
                        logger.info("All HDF5 frames processed")
                        self.running.clear()
                        break
                    continue

                response = self.request_inference(frame, proprioception)

                # Control: if using HDF5 qpos, send converted qpos to arm; otherwise use server response.
                if self.use_hdf5_qpos and self.last_hdf5_qpos is not None:
                    cmd = parse_actions(self.last_hdf5_qpos)
                    self.control_robot(cmd["joint_angles"], cmd["gripper_distance"])
                elif response is not None:
                    joint_angles = response["joint_angles"]
                    gripper_distance = response["gripper_distance"]
                    self.control_robot(joint_angles, gripper_distance)

                if response is not None:
                    inference_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed inference count: {failed_count}")

                frame_count += 1

                # Visualization
                cv2.putText(
                    frame,
                    f"Frame: {frame_count} | Inferences: {inference_count} | Task: {self.task_instruction}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if response is not None:
                    joint_info = f"Joints: {response['joint_angles'][:3]}"
                    gripper_info = f"Gripper: {response['gripper_distance']}"
                    cv2.putText(frame, joint_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, gripper_info, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save or display frame
                if self.save_frames_dir:
                    self._save_frame(frame, frame_count)
                else:
                    cv2.imshow("Piper Client", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit signal received")
                        self.running.clear()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def _save_frame(self, frame, frame_count):
        """Save frame to specified directory."""
        try:
            filename = f"frame_{frame_count:06d}.jpg"
            filepath = Path(self.save_frames_dir) / filename
            cv2.imwrite(str(filepath), frame)
            if frame_count % 100 == 0:
                logger.info(f"Saved frames up to {frame_count}")
        except Exception as e:
            logger.error(f"Failed to save frame {frame_count}: {e}")

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")

        # Disable gripper
        try:
            self.piper.GripperCtrl(0, 0, 0x00, 0)
            logger.info("Gripper disabled")
        except Exception as e:
            logger.error(f"Failed to disable gripper: {e}")

        # Release camera
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
            logger.info("Camera released")

        # Close HDF5
        if self.h5_file:
            self.h5_file.close()
            logger.info("HDF5 file closed")

        cv2.destroyAllWindows()
        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Piper Robot Client for Remote Inference (camera or HDF5)")

    parser.add_argument("task_instruction", type=str, help="Task instruction for the robot (e.g., 'Pick up the cup')")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="Inference server URL")
    parser.add_argument("--can-interface", type=str, default="can0", help="CAN interface name for Piper arm")
    parser.add_argument(
        "--camera-port", type=int, default=0, help="Camera port/device index (ignored when --hdf5-file is used)"
    )
    parser.add_argument(
        "--save-frames-dir",
        type=str,
        default=None,
        help="Directory to save frames instead of displaying them (default: None for display mode)",
    )
    parser.add_argument(
        "--hdf5-file",
        type=Path,
        default=None,
        help="Use frames from this HDF5 instead of a live camera (expects observations/images/cam_high)",
    )
    parser.add_argument(
        "--use-hdf5-qpos",
        action="store_true",
        help="Use qpos from the HDF5 file for inference input and reset arm to the initial qpos",
    )

    args = parser.parse_args()

    if not args.task_instruction or len(args.task_instruction.strip()) == 0:
        parser.error("Task instruction cannot be empty")

    if args.use_hdf5_qpos and args.hdf5_file is None:
        parser.error("--use-hdf5-qpos requires --hdf5-file to be set")

    try:
        client = PiperClient(
            task_instruction=args.task_instruction,
            server_url=args.server_url,
            can_interface=args.can_interface,
            camera_port=args.camera_port,
            save_frames_dir=args.save_frames_dir,
            hdf5_path=args.hdf5_file,
            use_hdf5_qpos=args.use_hdf5_qpos,
        )

        client.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
