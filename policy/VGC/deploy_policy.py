# import packages and module here
import sys
import os
import torch
import numpy as np
import cv2
from collections import deque
from hydra import initialize, compose
from omegaconf import OmegaConf
import hydra

# Add paths
current_file_path = os.path.abspath(__file__)
vgc_dir = os.path.dirname(current_file_path)
sys.path.append(vgc_dir)

# Add DP3 path for utilities
policy_dir = os.path.dirname(vgc_dir)
dp3_pkg_path = os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy')
sys.path.append(dp3_pkg_path)
sys.path.append(os.path.join(dp3_pkg_path, 'diffusion_policy_3d'))

from vgc.dataset.vgc_dataset import quat_to_rot6d

def encode_obs(observation):  # Post-Process Observation
    # Convert observation to model input format
    # observation structure from RoboTwin:
    # {
    #   'joint_action': {'vector': ...},
    #   'pointcloud': ...,
    #   'observation': { 'cam_name': {'rgb': ...}, ... },
    #   'endpose': {'left_endpose': ..., 'right_endpose': ...}
    # }
    
    obs = dict()
    
    # 1. Agent Pos (32D: 14 joint + 9 left + 9 right)
    joint_state = observation['joint_action']['vector'] # (14,)
    
    left_endpose = observation['endpose']['left_endpose'] # (7,)
    right_endpose = observation['endpose']['right_endpose'] # (7,)
    
    # Convert to 9D
    def convert_pose(pose):
        pos = pose[:3]
        quat = pose[3:]
        rot6d = quat_to_rot6d(np.array(quat)[None, :])[0]
        return np.concatenate([pos, rot6d])
        
    left_9d = convert_pose(left_endpose)
    right_9d = convert_pose(right_endpose)
    
    obs['agent_pos'] = np.concatenate([joint_state, left_9d, right_9d])
    
    # 2. Point Cloud
    obs['point_cloud'] = observation['pointcloud'] # (N, 6)
    
    # 3. Images
    # Need to collect images from observation dict and stack them
    # Order matters! Should match training order (usually sorted by name)
    images = []
    cam_names = sorted([k for k in observation['observation'].keys() if "rgb" in observation['observation'][k]])
    
    for cam_name in cam_names:
        # Decode if necessary (RoboTwin usually returns raw bytes or numpy array)
        # Assuming numpy array (H, W, C) or bytes
        img_data = observation['observation'][cam_name]['rgb']

        if isinstance(img_data, bytes):
             img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
             # cv2.imdecode returns BGR, but DINOv2 expects RGB
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
             img = img_data
             
        # Resize to 224x224 and Normalize
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        images.append(img)
        
    # Stack: (K, H, W, C) -> (K, C, H, W)
    images = np.stack(images, axis=0)
    images = np.moveaxis(images, -1, 1)
    
    obs['images'] = images
    
    return obs


def get_model(usr_args):
    # Load config
    config_path = "config"
    config_name = "vgc_policy"
    
    # We need to merge with task config
    # usr_args contains task_name, task_config etc.
    
    # Initialize Hydra
    # Note: We are inside policy/VGC/
    with initialize(config_path=config_path, version_base='1.2'):
        # Compose config, overriding task
        # FORCE use_pc_features=False for inference to ensure DINO is loaded
        overrides = [
            f"task={usr_args['task_name']}",
            "policy.use_pc_features=False"
        ]
        cfg = compose(config_name=config_name, overrides=overrides)
        
    # Instantiate Model
    model = hydra.utils.instantiate(cfg.policy)
    
    # Load Checkpoint
    # Search for the latest checkpoint in data/outputs if not specified
    # usr_args might contain 'checkpoint_path' if we pass it manually, or we infer it
    
    ckpt_path = None
    
    # 1. Try explicit path if passed (we can add this to eval.sh args if needed)
    # 2. Try standard DP3 path structure if ckpt_setting is used
    # 3. Search for latest in data/outputs
    
    base_output_dir = os.path.join(vgc_dir, "data/outputs")
    
    # Find latest directory matching task name
    if os.path.exists(base_output_dir):
        # Try to find run directories directly in base_output_dir (Flat structure)
        runs = sorted([r for r in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, r)) and usr_args['task_name'] in r], reverse=True)
        
        ckpt_dir = None
        if runs:
            # Found in flat structure
            latest_run = runs[0]
            ckpt_dir = os.path.join(base_output_dir, latest_run, "checkpoints")
        else:
            # Try nested structure (outputs/DATE/RUN_DIR)
            dates = sorted([d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))], reverse=True)
            for date in dates:
                date_dir = os.path.join(base_output_dir, date)
                # List all run directories
                runs = sorted([r for r in os.listdir(date_dir) if usr_args['task_name'] in r], reverse=True)
                if runs:
                    # Found latest run
                    latest_run = runs[0]
                    ckpt_dir = os.path.join(date_dir, latest_run, "checkpoints")
                    break
        
        if ckpt_dir and os.path.exists(ckpt_dir):
            # Find best or latest ckpt
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') or f.endswith('.pth')]
            if ckpts:
                # Prefer 'latest.ckpt' or 'best.ckpt' or just the last one
                if 'latest.ckpt' in ckpts:
                    ckpt_path = os.path.join(ckpt_dir, 'latest.ckpt')
                elif 'best.ckpt' in ckpts:
                    ckpt_path = os.path.join(ckpt_dir, 'best.ckpt')
                else:
                    ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])
            
    if ckpt_path is None:
        print(f"Warning: No checkpoint found for task {usr_args['task_name']}. Using random weights.")
    else:
        print(f"Loading checkpoint from: {ckpt_path}")
        
    return VGCWrapper(model, cfg, usr_args, ckpt_path)


class VGCWrapper:
    def __init__(self, model, cfg, usr_args, ckpt_path=None):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.model.to(self.device)
        self.model.eval()
        
        self.n_obs_steps = cfg.policy.n_obs_steps
        self.n_action_steps = cfg.policy.n_action_steps
        
        # Observation Cache (Deque)
        self.obs_cache = deque(maxlen=self.n_obs_steps)
        
        # Load Weights
        if ckpt_path:
            self.load_checkpoint(ckpt_path)
        
    def load_checkpoint(self, ckpt_path):
        payload = torch.load(ckpt_path, map_location=self.device)
        
        # Handle different saving formats
        if 'state_dicts' in payload:
            state_dict = payload['state_dicts']['model']
        elif 'state_dict' in payload:
            state_dict = payload['state_dict']
        else:
            state_dict = payload # Raw state dict
            
        # Filter out DINO weights if they are missing in the checkpoint but present in the model
        # This happens when we train with precomputed features (DINO not in model) 
        # but infer with images (DINO in model)
        model_state_dict = self.model.state_dict()
        
        # Keys present in model but missing in checkpoint
        missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
        
        # If missing keys are related to visual_encoder (DINO), it's expected
        # because we initialized DINO from pretrained weights in __init__
        # So we should just ignore them during load_state_dict
        
        # We can use strict=False, but that might hide other errors.
        # Better approach: Check if missing keys are indeed DINO keys
        
        dino_keys = [k for k in missing_keys if "visual_encoder" in k]
        other_missing_keys = [k for k in missing_keys if "visual_encoder" not in k]
        
        if len(other_missing_keys) > 0:
            print(f"Warning: Missing non-DINO keys in checkpoint: {other_missing_keys}")
            
        if len(dino_keys) > 0:
            print(f"Note: {len(dino_keys)} DINOv2 keys are missing in checkpoint (expected since training used precomputed features). Keeping pretrained weights.")
            
        # Load with strict=False to allow missing DINO keys
        self.model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully.")

    def update_obs(self, obs):
        # obs is a dict of numpy arrays
        # We need to add it to cache
        self.obs_cache.append(obs)
        
    def reset_obs(self):
        self.obs_cache.clear()
        
    def get_action(self):
        # 1. Stack observations from cache
        # We need n_obs_steps
        if len(self.obs_cache) == 0:
            return np.zeros((self.n_action_steps, 14)) # Dummy
            
        # Repeat last obs if not enough
        while len(self.obs_cache) < self.n_obs_steps:
            self.obs_cache.append(self.obs_cache[-1])
            
        # Stack
        # obs_cache is list of dicts. We want dict of stacked arrays (B, T, ...)
        # B=1
        
        batch = {'obs': {}}
        keys = self.obs_cache[0].keys()
        
        for k in keys:
            # Stack along time dimension
            val = np.stack([x[k] for x in self.obs_cache], axis=0) # (T, ...)
            # Add Batch dim
            val = val[None, ...] # (1, T, ...)
            # Convert to tensor
            batch['obs'][k] = torch.from_numpy(val).float().to(self.device)
            
        # 2. Inference
        with torch.no_grad():
            action = self.model.get_action(batch) # (1, Horizon, Action_Dim)
            
        # 3. Post-process
        action = action[0].cpu().numpy() # (Horizon, Action_Dim)
        
        # Return only n_action_steps
        return action[:self.n_action_steps]


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    
    if len(model.obs_cache) == 0:
        model.update_obs(obs)

    actions = model.get_action()

    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):  
    model.reset_obs()

