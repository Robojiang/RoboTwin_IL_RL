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
aim_dir = os.path.dirname(current_file_path) # policy/AIM
policy_dir = os.path.dirname(aim_dir) # policy
sys.path.append(aim_dir)

# Add DP3 path for utilities
dp3_pkg_path = os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy')
sys.path.append(dp3_pkg_path)
sys.path.append(os.path.join(dp3_pkg_path, 'diffusion_policy_3d'))

# Add VGC for dataset utils
vgc_dir = os.path.join(policy_dir, 'VGC')
sys.path.append(vgc_dir)

from vgc.dataset.vgc_dataset import quat_to_rot6d

def encode_obs(observation):  # Post-Process Observation
    # Convert observation to model input format
    obs = dict()
    
    # 1. Agent Pos (32D: 14 joint + 9 left + 9 right)
    # Check if joint_action is valid
    if 'joint_action' in observation and 'vector' in observation['joint_action']:
        joint_state = observation['joint_action']['vector'] # (14,)
    else:
        # Fallback or dummy?
        joint_state = np.zeros(14)
    
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
    
    obs['agent_pos'] = np.concatenate([joint_state, left_9d, right_9d]).astype(np.float32)
    
    # 2. Point Cloud
    obs['point_cloud'] = observation['pointcloud'].astype(np.float32) # (N, 6)
    
    return obs


def get_model(usr_args):
    # Load config
    config_path = "config"
    config_name = "aim_policy"
    
    # Initialize Hydra
    with initialize(config_path=config_path, version_base='1.2'):
        overrides = [
            f"task={usr_args['task_name']}"
        ]
        cfg = compose(config_name=config_name, overrides=overrides)
        
    # Instantiate Model
    model = hydra.utils.instantiate(cfg.policy)
    
    # Checkpoint Search Logic
    ckpt_path = None
    
    # 1. Search in AIM/checkpoints/<task_name>
    checkpoints_dir = os.path.join(aim_dir, "checkpoints")
    task_ckpt_dir = os.path.join(checkpoints_dir, usr_args['task_name'])
    
    if os.path.exists(task_ckpt_dir):
        ckpts = [f for f in os.listdir(task_ckpt_dir) if f.endswith('.ckpt') or f.endswith('.pth')]
        if ckpts:
            if 'latest.ckpt' in ckpts:
                ckpt_path = os.path.join(task_ckpt_dir, 'latest.ckpt')
            elif 'best.ckpt' in ckpts:
                ckpt_path = os.path.join(task_ckpt_dir, 'best.ckpt')
            else:
                ckpt_path = os.path.join(task_ckpt_dir, sorted(ckpts)[-1])

    # 2. Search in data/outputs (Hydra Output)
    if ckpt_path is None:
        base_output_dir = os.path.join(aim_dir, "data/outputs")
        if os.path.exists(base_output_dir):
            if usr_args['task_name'] in os.listdir(base_output_dir):
                # Structure: data/outputs/task_name/checkpoints
                 ckpt_dir = os.path.join(base_output_dir, usr_args['task_name'], "checkpoints")
                 if os.path.exists(ckpt_dir):
                     ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
                     if ckpts:
                         ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])
            else:
                # Structure: data/outputs/DATE_TIME_NAME_TASK/
                 runs = sorted([r for r in os.listdir(base_output_dir) if usr_args['task_name'] in r], reverse=True)
                 for run in runs:
                     ckpt_dir = os.path.join(base_output_dir, run, "checkpoints")
                     if os.path.exists(ckpt_dir):
                         ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
                         if ckpts:
                             ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])
                             break

    if ckpt_path is None:
        print(f"Warning: No checkpoint found for task {usr_args['task_name']}. Using random weights.")
    else:
        print(f"Loading checkpoint from: {ckpt_path}")
        
    return AIMWrapper(model, cfg, usr_args, ckpt_path)


class AIMWrapper:
    def __init__(self, model, cfg, usr_args, ckpt_path=None):
        self.model = model
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.n_obs_steps = cfg.policy.n_obs_steps
        self.n_action_steps = cfg.policy.n_action_steps
        
        # Observation Cache
        self.obs_cache = deque(maxlen=self.n_obs_steps)
        
        # Load Weights
        if ckpt_path:
            self.load_checkpoint(ckpt_path)
        
    def load_checkpoint(self, ckpt_path):
        payload = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(payload['state_dicts']['model'])
        if 'ema_model' in payload['state_dicts'] and payload['state_dicts']['ema_model'] is not None:
             print("Loading EMA model...")
             self.model.load_state_dict(payload['state_dicts']['ema_model'])
             
    def reset(self):
        self.obs_cache.clear()

    def update_obs(self, observation):
        # 1. Process Observation
        obs = encode_obs(observation)
        # 2. Update Cache
        self.obs_cache.append(obs)
        
    def predict(self, observation):
        # 1. Update Cache
        self.update_obs(observation)
        
        # 2. Create Batch
        # Pad if not enough history
        # (This handles the case where cache might be short at start)
        # Note: update_obs already appended the latest one.
        # But if we need more padding:
        temp_cache = list(self.obs_cache)
        while len(temp_cache) < self.n_obs_steps:
             temp_cache.append(temp_cache[-1]) # Pad with latest? Or duplicate?
             # Actually, if we just started, we might want to prepend duplicates of first frame
             # But here we append. If len=1, now len=2 (duplicates). 
             # Logic in original separate predict was: while len(cache) < N: cache.append(obs)
             # But here cache is persistent. 
             # Let's fix padding logic: if cache is not full, we use valid entries repeated?
             
        # Actually safer way to handle padding for inference without modifying persistent cache state 
        # (unless we really want to fill it)
        
        # Stack history
        batch_obs = {}
        keys = list(temp_cache[0].keys())
        for k in keys:
            try:
                # We need exactly n_obs_steps
                # If we have [1, 2], n=2 -> OK
                # If we have [1], n=2 -> need [1, 1]
                vals = [x[k] for x in temp_cache]
                while len(vals) < self.n_obs_steps:
                     # Prepend or append? Usually replicate first frame if history missing at start
                     vals.insert(0, vals[0]) 
                
                # Take last n_obs_steps just in case
                vals = vals[-self.n_obs_steps:]
                
                val = np.stack(vals)
                batch_obs[k] = torch.from_numpy(val).to(self.device).unsqueeze(0) # (1, T, ...)
            except Exception as e:
                print(f"Error stacking key {k}: {e}")

        batch = {'obs': batch_obs}
        
        # 3. Inference
        with torch.no_grad():
            action = self.model.get_action(batch) # (1, T, D)
            
        # 4. Return Action Sequence
        raw_action = action[0].cpu().numpy() # (T_action, D_action)
        return raw_action

def reset_model(model):
    model.reset()

def eval(env, model, obs):
    actions = model.predict(obs)
    
    steps_to_run = model.n_action_steps
    if steps_to_run > len(actions):
        steps_to_run = len(actions)
        
    for i in range(steps_to_run):
        action = actions[i]
        env.take_action(action)
        
        # If there are more steps to run in this chunk, 
        # update model with intermediate observations
        if i < steps_to_run - 1:
             new_obs = env.get_obs()
             model.update_obs(new_obs)

