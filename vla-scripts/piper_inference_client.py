import argparse
import sys
import time
import logging
from threading import Thread, Event
from pathlib import Path

import cv2
import requests
import base64

from piper_sdk import C_PiperInterface_V2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PiperClient:
    def __init__(self, task_instruction, server_url, can_interface="can0", camera_port=0, save_frames_dir=None):
        """
        Initialize Piper client for remote inference
        
        Args:
            task_instruction (str): Task instruction
            server_url (str): URL of inference server (e.g., http://localhost:8000)
            can_interface (str): CAN interface name (default: "can0")
            camera_port (int): Camera port/device (default: 0)
            save_frames_dir (str): Directory to save frames (default: None for display mode)
        """
        self.task_instruction = task_instruction
        self.server_url = server_url.rstrip('/')  # Remove trailing slash
        self.save_frames_dir = save_frames_dir
        
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
        
        # Camera setup
        logger.info(f"Initializing camera on port: {camera_port}")
        self.cap = cv2.VideoCapture(camera_port)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera on port {camera_port}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Control flags
        self.running = Event()
        self.running.set()
        
        # Check server health
        self._check_server_health()
    
    def _check_server_health(self):
        """Check if server is alive"""
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
        """
        Capture frame from camera
        
        Returns:
            numpy.ndarray: BGR frame or None if failed
        """
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None
        return frame
    
    def frame_to_base64(self, frame):
        """
        Convert frame to base64 encoded string
        
        Args:
            frame (numpy.ndarray): BGR frame
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            logger.error(f"Failed to encode frame: {e}")
            return None
    
    def get_proprioception(self):
        """
        Get current joint state from robot
        
        Returns:
            list: [j0, j1, j2, j3, j4, j5, gripper_distance] or zeros if failed
        """
        try:
            joint_feedback = self.piper.GetArmJointCtrl()
            gripper_feedback = self.piper.GetArmGripperCtrl()
            joint_state = [
                joint_feedback.joint_ctrl.joint_1,
                joint_feedback.joint_ctrl.joint_2,
                joint_feedback.joint_ctrl.joint_3,
                joint_feedback.joint_ctrl.joint_4,
                joint_feedback.joint_ctrl.joint_5,
                joint_feedback.joint_ctrl.joint_6,
                gripper_feedback.gripper_ctrl.grippers_angle
            ]
            return joint_state
        except Exception as e:
            logger.warning(f"Failed to get joint feedback: {e}, using zero proprioception")
            return [0, 0, 0, 0, 0, 0, 0]
    
    def request_inference(self, frame, proprioception):
        """
        Send inference request to server
        
        Args:
            frame (numpy.ndarray): BGR frame
            proprioception (list): Joint state [j0-j5, gripper]
            
        Returns:
            dict: Response data or None if failed
        """
        try:
            # Encode frame
            image_base64 = self.frame_to_base64(frame)
            if image_base64 is None:
                return None
            
            # Prepare request data
            request_data = {
                "image": image_base64,
                "task_instruction": self.task_instruction,
                "proprioception": proprioception
            }
            
            # Send request
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/infer",
                json=request_data,
                timeout=30
            )
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
    
    def control_robot(self, joint_angles, gripper_distance):
        """
        Send control commands to Piper robot
        
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
            
            self.piper.GripperCtrl(
                gripper_distance,
                1000,  # Standard gripper effort
                0x01,  # Enable gripper
                0      # No zero setting
            )
            
            # Get and log arm status
            arm_status = self.piper.GetArmStatus()
            logger.debug(f"Arm status: {arm_status}")
            
        except Exception as e:
            logger.error(f"Failed to control robot: {e}")
    
    def run(self):
        """Main inference loop"""
        logger.info(f"Starting client for task: {self.task_instruction}")
        logger.info(f"Server URL: {self.server_url}")
        if self.save_frames_dir:
            logger.info(f"Saving frames to: {self.save_frames_dir}")
        else:
            logger.info("Press 'q' to quit")
        
        frame_count = 0
        inference_count = 0
        failed_count = 0
        
        try:
            while self.running.is_set():
                # Capture frame
                frame = self.get_camera_frame()
                if frame is None:
                    continue
                
                # Get proprioception
                proprioception = self.get_proprioception()
                
                # Request inference
                response = self.request_inference(frame, proprioception)
                
                if response is not None:
                    # Extract actions
                    joint_angles = response["joint_angles"]
                    gripper_distance = response["gripper_distance"]
                    
                    # Control robot
                    self.control_robot(joint_angles, gripper_distance)
                    
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
                    2
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
                    # Check for quit signal
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit signal received")
                        self.running.clear()
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()
    
    def _save_frame(self, frame, frame_count):
        """
        Save frame to specified directory
        
        Args:
            frame (numpy.ndarray): Frame to save
            frame_count (int): Frame count for filename
        """
        try:
            filename = f"frame_{frame_count:06d}.jpg"
            filepath = Path(self.save_frames_dir) / filename
            cv2.imwrite(str(filepath), frame)
            if frame_count % 100 == 0:  # Log every 100 frames
                logger.info(f"Saved frames up to {frame_count}")
        except Exception as e:
            logger.error(f"Failed to save frame {frame_count}: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Disable gripper
        try:
            self.piper.GripperCtrl(0, 0, 0x00, 0)
            logger.info("Gripper disabled")
        except Exception as e:
            logger.error(f"Failed to disable gripper: {e}")
        
        # Release camera
        if self.cap:
            self.cap.release()
            logger.info("Camera released")
        
        # Close display
        cv2.destroyAllWindows()
        
        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Piper Robot Client for Remote Inference"
    )
    
    parser.add_argument(
        "task_instruction",
        type=str,
        help="Task instruction for the robot (e.g., 'Pick up the cup')"
    )
    
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="Inference server URL (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--can-interface",
        type=str,
        default="can0",
        help="CAN interface name for Piper arm (default: can0)"
    )
    
    parser.add_argument(
        "--camera-port",
        type=int,
        default=0,
        help="Camera port/device index (default: 0 for /dev/video0)"
    )
    
    parser.add_argument(
        "--save-frames-dir",
        type=str,
        default=None,
        help="Directory to save frames instead of displaying them (default: None for display mode)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.task_instruction or len(args.task_instruction.strip()) == 0:
        parser.error("Task instruction cannot be empty")
    
    try:
        # Initialize and run client
        client = PiperClient(
            task_instruction=args.task_instruction,
            server_url=args.server_url,
            can_interface=args.can_interface,
            camera_port=args.camera_port,
            save_frames_dir=args.save_frames_dir
        )
        
        client.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
