import os
import sys
import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import tqdm
from libero.libero import benchmark
from collections import deque

from flask import Flask, request, jsonify

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # project/
sys.path.insert(0, str(ROOT))                # 放最前面，优先级最高
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_latent_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "./vla-scripts/libero_log/finetune-libero"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    
    action_decoder_path:str = "./vla-scripts/libero_log/finetune-libero/action_decoder.pt"
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    save_video: bool = False                         # Whether to save rollout videos

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"               # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    window_size: int = 12

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/eval_logs"   # Local directory for eval logs
    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)


from prismatic.models.policy.transformer_utils import MAPBlock

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class ActionDecoderHead(torch.nn.Module):
    def __init__(self, window_size = 5):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = 512, n_heads = 8)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = 512, n_heads = 8)

        self.proj = nn.Sequential(
                                nn.Linear(512, 7 * window_size),
                                nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed):
        latent_action_tokens = latent_action_tokens[:, -4:]
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(self.latent_action_pool(latent_action_tokens, init_embed = visual_embed))
        
        return action


class ActionDecoder(nn.Module):
    def __init__(self,window_size=5):
        super().__init__()
        self.net = ActionDecoderHead(window_size=window_size)

        self.temporal_size = window_size
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]


    def reset(self):
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

    
    def forward(self, latent_actions, visual_embed, mask, action_low, action_high):
        # Forward action decoder
        pred_action = self.net(latent_actions.to(torch.float), visual_embed.to(torch.float)).reshape(-1, self.temporal_size, 7)
        pred_action = np.array(pred_action.tolist())
        
        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = pred_action  
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

        # Ensemble temporally to predict actions
        action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)
        
        action_prediction = np.where(
            mask,
            0.5 * (action_prediction + 1) * (action_high - action_low) + action_low,
            action_prediction,
        )

        return action_prediction

def init(vla_path, decoder_path):
    global action_decoder, model, processor, resize_size, latent_action_detokenize
    global last_infer_id, hist_action, prev_hist_action
    global cfg
    
    cfg = GenerateConfig(
        pretrained_checkpoint=vla_path,
        action_decoder_path=decoder_path,
        task_suite_name='realworld_piper'
    )
    
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load action decoder
    action_decoder = ActionDecoder(cfg.window_size)
    action_decoder.net.load_state_dict(torch.load(cfg.action_decoder_path))
    action_decoder.eval().cuda()

    # Load model
    model = get_model(cfg)

    # wrapped_model Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # wrapped_model Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)    
    
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    latent_action_detokenize = [f'<ACT_{i}>' for i in range(32)]
    
    last_infer_id = -1
    hist_action = ''
    prev_hist_action = ['']
    
def infer(obs, task_description, infer_id):
    global last_infer_id, hist_action, prev_hist_action, action_decoder, model, resize_size, cfg
    
    if infer_id != last_infer_id:
        prev_hist_action = ['']
        action_decoder.reset()
    hist_action = ''
    
    # Get preprocessed image
    # 先逆向旋转一下
    obs["agentview_image"] = obs["agentview_image"][::-1, ::-1]
    img = get_libero_image(obs, resize_size)
    
    # Prepare observations dict
    # Note: UniVLA does not take proprio state as input
    observation = {
        "full_image": img,
        # "state": np.concatenate(
        #     (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        # ),
    }
    
    # Query model to get action
    latent_action, visual_embed, generated_ids = get_latent_action(
        cfg,
        model,
        observation,
        task_description,
        processor=processor,
        hist_action=prev_hist_action[-1],
    )
    
    # Record history latent actions
    hist_action = ''
    for latent_action_ids in generated_ids[0]:
        hist_action += latent_action_detokenize[latent_action_ids.item() - 32001]
    prev_hist_action.append(hist_action)
    
    action_norm_stats = model.get_action_stats(cfg.unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])

    action = action_decoder(latent_action, visual_embed, mask, action_low, action_high)

    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # wrapped_model The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if cfg.model_family == "openvla":
        action = invert_gripper_action(action)
    
    last_infer_id = infer_id
    
    return action

# def _decode_image(image_data: str) -> np.ndarray:
#     image_bytes = base64.b64decode(image_data)
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if image_bgr is None:
#         raise ValueError("Failed to decode image from bytes")
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     return image_rgb

@app.route('/', methods=['POST'])
def infer_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        required_fields = ["image", "task_instruction", "infer_id"]
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400

        try:
            # image_rgb = _decode_image(data["image"])
            # 转进来是jpg压缩之后的b64码
            # 先解码b64
            image_bytes = base64.b64decode(data["image"])
            image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            return jsonify({"success": False, "error": f"Failed to process image: {exc}"}), 400

        task_instruction = data["task_instruction"]
        if not isinstance(task_instruction, str) or not task_instruction.strip():
            return jsonify({"success": False, "error": "Invalid task_instruction"}), 400

        infer_id = data["infer_id"]
        if not isinstance(infer_id, int):
            return jsonify({"success": False, "error": "Invalid infer_id"}), 400
        
        obs = {
            "agentview_image": image_rgb,
        }

        action = infer(obs, task_instruction, infer_id)
        return jsonify({"success": True, "action": action.tolist()}), 200

    except Exception as exc:
        logger.error("Inference request failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": f"Inference failed: {exc}"}), 500

if __name__ == "__main__":
    vla_path = '/autodl-fs/data/UniVLA/vla-scripts/libero-realworld-runs/univla-7b+realworld_piper+b16+lr-0.00035+lora-r32+dropout-0.0--image_aug=w-LowLevelDecoder-ws-12/'
    decoder_path = '/autodl-fs/data/UniVLA/vla-scripts/libero-realworld-runs/univla-7b+realworld_piper+b16+lr-0.00035+lora-r32+dropout-0.0--image_aug=w-LowLevelDecoder-ws-12/action_decoder-30000.pt'
    init(vla_path, decoder_path)
    app.run(host='0.0.0.0', port=6006)
