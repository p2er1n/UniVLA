#!/usr/bin/env python3
"""Minimal Piper client: capture camera frames, call inference server, control arm."""

import argparse
import base64
import logging
import math
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
import PIL.Image as Image

from piper_sdk import C_PiperInterface_V2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POS_SCALE_M = 1e-6  # 0.001 mm -> meters
ANG_SCALE_RAD = math.pi / 180000.0  # 0.001 degrees -> radians
POS_TO_PIPER = 1.0 / POS_SCALE_M
ANG_TO_PIPER = 1.0 / ANG_SCALE_RAD


def _get_field(obj: Any, name: str) -> float:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    raise AttributeError(f"End pose missing field: {name}")


def _extract_end_pose(msg: Any) -> Any:
    if isinstance(msg, tuple) and len(msg) == 3:
        _, _, end_pose = msg
        return end_pose
    if hasattr(msg, "end_pose"):
        return msg.end_pose
    return msg


def _convert_end_pose(end_pose: Any) -> Dict[str, float]:
    x = _get_field(end_pose, "X_axis") * POS_SCALE_M
    y = _get_field(end_pose, "Y_axis") * POS_SCALE_M
    z = _get_field(end_pose, "Z_axis") * POS_SCALE_M
    rx = _get_field(end_pose, "RX_axis") * ANG_SCALE_RAD
    ry = _get_field(end_pose, "RY_axis") * ANG_SCALE_RAD
    rz = _get_field(end_pose, "RZ_axis") * ANG_SCALE_RAD
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "rx": float(rx),
        "ry": float(ry),
        "rz": float(rz),
    }


def _request_inference(server_url: str, payload: Dict[str, Any], timeout_s: float) -> Optional[Dict[str, Any]]:
    try:
        response = requests.post(server_url, json=payload, timeout=timeout_s)
        if response.status_code != 200:
            logger.error("Server error: status %s, %s", response.status_code, response.text)
            return None
        data = response.json()
        if not data.get("success", False):
            logger.error("Server inference failed: %s", data.get("error"))
            return None
        return data
    except requests.Timeout:
        logger.error("Inference request timeout")
        return None
    except requests.ConnectionError:
        logger.error("Failed to connect to server at %s", server_url)
        return None
    except Exception as exc:
        logger.error("Inference request failed: %s", exc)
        return None



def _action_to_end_pose_units(
    action: Sequence[float], current_pose: Dict[str, float]
) -> Tuple[Tuple[int, int, int, int, int, int], int]:
    if len(action) < 7:
        raise ValueError(f"Expected 7 action values, got {len(action)}")
    target = [
        current_pose["x"] + float(action[0]),
        current_pose["y"] + float(action[1]),
        current_pose["z"] + float(action[2]),
        current_pose["rx"] + float(action[3]),
        current_pose["ry"] + float(action[4]),
        current_pose["rz"] + float(action[5]),
    ]
    x = int(round(target[0] * POS_TO_PIPER))
    y = int(round(target[1] * POS_TO_PIPER))
    z = int(round(target[2] * POS_TO_PIPER))
    rx = int(round(target[3] * ANG_TO_PIPER))
    ry = int(round(target[4] * ANG_TO_PIPER))
    rz = int(round(target[5] * ANG_TO_PIPER))
    
    # gripper 返回的是-1和1，-1代表开，1代表合，先将-1和1进行映射
    mapping = {
        -1.0: 0.07,
        1.0: 0
    }
    if action[6] not in mapping:
        raise ValueError(f"Invalid gripper action value: {action[6]}")
    gripper_distance = int(mapping[action[6]] * 1000 * 1000)
    return (x, y, z, rx, ry, rz), gripper_distance


def _control_robot_end_pose(
    piper: C_PiperInterface_V2, pose_units: Tuple[int, int, int, int, int, int], gripper_distance: int
) -> None:
    x, y, z, rx, ry, rz = pose_units
    piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
    piper.EndPoseCtrl(x, y, z, rx, ry, rz)
    piper.GripperCtrl(gripper_distance, 1000, 0x01, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Piper end-pose client v2")
    parser.add_argument("task_instruction", type=str, help="Task instruction for the robot")
    parser.add_argument("--server-url", type=str, default="http://localhost:6006", help="Inference server URL")
    parser.add_argument("--can-interface", type=str, default="can0", help="CAN interface name for Piper arm")
    parser.add_argument("--camera-port", type=int, default=0, help="Camera port/device index")
    parser.add_argument("--loop-sleep-s", type=float, default=0.02, help="Sleep seconds between requests")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="Inference request timeout")
    args = parser.parse_args()

    if not args.task_instruction.strip():
        parser.error("Task instruction cannot be empty")

    server_url = args.server_url.rstrip("/")

    logger.info("Connecting Piper on CAN interface: %s", args.can_interface)
    piper = C_PiperInterface_V2(args.can_interface)
    piper.ConnectPort()

    logger.info("Enabling Piper arm...")
    retries = 0
    while not piper.EnablePiper() and retries < 100:
        time.sleep(0.01)
        retries += 1
    if retries >= 100:
        logger.error("Failed to enable Piper arm after 100 retries")
        sys.exit(1)

    logger.info("Initializing camera on port: %s", args.camera_port)
    cap = cv2.VideoCapture(args.camera_port)
    if not cap.isOpened():
        logger.error("Failed to open camera on port %s", args.camera_port)
        sys.exit(1)

    request_id = 0
    infer_id = int(time.time() * 1000)
    
    # 给摄像头热身
    for i in range(50):
        ok, frame = cap.read()
    
    try:
        while True:
            request_id += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Failed to read frame from camera")
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # print(frame)
            # print(type(frame))
            # print(frame.shape)
            # img_pil = Image.fromarray(frame)
            # img_pil.save("frame.png")
            # break
            
            # 直接将frame这个ndarray放到json请求参数中

            end_pose_msg = piper.GetArmEndPoseMsgs()
            end_pose = _extract_end_pose(end_pose_msg)
            current_pose = _convert_end_pose(end_pose)

            payload = {
                "image": frame.tolist(),
                "task_instruction": args.task_instruction,
                "infer_id": infer_id,
            }
            response = _request_inference(server_url, payload, args.timeout_s)
            if response is None:
                if args.loop_sleep_s > 0:
                    time.sleep(args.loop_sleep_s)
                continue
            logger.info("Response: %s", response)
            action = response.get("action")
            if action is None:
                logger.error("Server response missing action field")
                if args.loop_sleep_s > 0:
                    time.sleep(args.loop_sleep_s)
                continue

            try:
                pose_units, gripper_distance = _action_to_end_pose_units(action, current_pose)
                # pose_units = tuple([pose_units[0], pose_units[1], pose_units[2], -178816, 64052, -72476])
                _control_robot_end_pose(piper, pose_units, gripper_distance)
                logger.info("Action: %s", pose_units)
            except Exception as exc:
                logger.error("Failed to process action: %s", exc)

            if args.loop_sleep_s > 0:
                time.sleep(args.loop_sleep_s)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
