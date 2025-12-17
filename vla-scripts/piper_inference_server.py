import argparse
import sys
import logging
import json
import base64
import io

import torch
import torchvision
import numpy as np
from flask import Flask, request, jsonify

from real_world_deployment import UniVLAInference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Global inference policy
policy = None
factor = 57295.7795  # 1000*180/π


def initialize_policy(model_path, decoder_path):
    """Initialize UniVLA policy"""
    global policy
    logger.info(f"Initializing UniVLA with model: {model_path}")
    policy = UniVLAInference(
        saved_model_path=model_path,
        pred_action_horizon=12,
        decoder_path=decoder_path
    )
    logger.info("UniVLA policy initialized successfully")


def image_to_tensor(image_data):
    """
    Convert base64 encoded image to tensor
    
    Args:
        image_data (str): Base64 encoded image string
        
    Returns:
        torch.Tensor: Preprocessed image tensor (1, 3, 224, 224)
    """
    # Decode base64 to bytes
    image_bytes = base64.b64decode(image_data)
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    import cv2
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
    
    # Resize to 224x224
    image_resized = torchvision.transforms.Resize((224, 224))(image_tensor)
    
    # Add batch dimension
    image_batch = image_resized.unsqueeze(0)
    
    return image_batch


def parse_actions(all_actions):
    """
    Parse action tensor to joint angles and gripper distance
    
    Args:
        all_actions: Action tensor (7,)
        
    Returns:
        dict: Parsed actions containing joint angles and gripper distance
    """
    # Convert to numpy array if needed
    action = all_actions.detach().cpu().numpy() if isinstance(all_actions, torch.Tensor) else all_actions
    
    # Parse joint angles (6 DOF) - convert radians to Piper units
    # factor = 1000*180/π ≈ 57295.7795
    joint_angles = action[0:6] * factor
    joint_angles = np.round(joint_angles).astype(int).tolist()
    
    # Parse gripper distance - convert normalized (0-1) to Piper units (×1000000)
    gripper_distance = action[6]
    gripper_distance_piper = int(np.abs(gripper_distance) * 1000 * 1000)
    
    return {
        "joint_angles": joint_angles,
        "gripper_distance": gripper_distance_piper,
        "raw_action": action.tolist()
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "policy_loaded": policy is not None
    }), 200


@app.route('/infer', methods=['POST'])
def infer():
    """
    Inference endpoint
    
    Expected POST data:
    {
        "image": "<base64_encoded_image>",
        "task_instruction": "<task_instruction>",
        "proprioception": [j0, j1, j2, j3, j4, j5, gripper_distance]
    }
    
    Returns:
    {
        "success": true/false,
        "joint_angles": [j0, j1, j2, j3, j4, j5],
        "gripper_distance": <int>,
        "raw_action": <list>,
        "error": "<error_message>"
    }
    """
    try:
        # Check if policy is loaded
        if policy is None:
            return jsonify({
                "success": False,
                "error": "Policy not initialized"
            }), 500
        
        # Parse request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Validate required fields
        required_fields = ["image", "task_instruction", "proprioception"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
        
        # Extract data
        image_data = data["image"]
        task_instruction = data["task_instruction"]
        proprioception = data["proprioception"]
        
        # Convert image to tensor
        try:
            image_tensor = image_to_tensor(image_data)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to process image: {str(e)}"
            }), 400
        
        # Convert proprioception to tensor
        try:
            proprio = torch.tensor([proprioception], dtype=torch.float32)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to process proprioception: {str(e)}"
            }), 400
        
        # Run inference
        try:
            logger.info(f"Inference request: task='{task_instruction}'")
            all_actions = policy.step(image_tensor, task_instruction, proprio)
            
            # Parse actions
            parsed_actions = parse_actions(all_actions)
            
            logger.info(f"Inference success: joint_angles={parsed_actions['joint_angles']}, gripper_distance={parsed_actions['gripper_distance']}")
            
            return jsonify({
                "success": True,
                "joint_angles": parsed_actions["joint_angles"],
                "gripper_distance": parsed_actions["gripper_distance"],
                "raw_action": parsed_actions["raw_action"]
            }), 200
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}", exc_info=True)
            return jsonify({
                "success": False,
                "error": f"Inference failed: {str(e)}"
            }), 500
    
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Request processing failed: {str(e)}"
        }), 500


def main():
    parser = argparse.ArgumentParser(
        description="UniVLA Inference Server"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="qwbu/univla-7b",
        help="Path or ID of the UniVLA model (default: qwbu/univla-7b)"
    )
    
    parser.add_argument(
        "--decoder",
        type=str,
        default="/root/autodl-fs/UniVLA/vla-scripts/runs/univla-7b+real_world+b8+lr-0.00035+lora-r32+dropout-0.0=w-LowLevelDecoder-ws-12/5/action_decoder-5.pt",
        help="Path to action decoder checkpoint"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize policy
        initialize_policy(args.model, args.decoder)
        
        # Start Flask server
        logger.info(f"Starting server on {args.host}:{args.port}")
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
