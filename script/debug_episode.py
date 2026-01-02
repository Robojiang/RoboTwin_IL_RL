
import sys
import os
import argparse
import yaml
import numpy as np
import importlib
from pathlib import Path
from collections import deque

# Add paths
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "policy"))
sys.path.append(os.path.join(parent_directory, "description/utils"))

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "task_config/_camera_config.yml")
    assert os.path.isfile(camera_config_path), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]

def main():
    parser = argparse.ArgumentParser(description="Debug Episode Script")
    parser.add_argument("--task_name", type=str, default="beat_block_hammer", help="Name of the task")
    parser.add_argument("--task_config", type=str, default="demo_clean", help="Task configuration file")
    parser.add_argument("--policy_name", type=str, default="DP3", help="Policy name")
    parser.add_argument("--ckpt_setting", type=str, default="demo_3d_vision_easy", help="Checkpoint setting")
    parser.add_argument("--expert_data_num", type=int, default=100, help="Number of expert data")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--render_freq", type=int, default=20, help="Render frequency (set to >0 to visualize)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    
    args_cli = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu_id)
    
    # Load Policy Config
    policy_config_path = os.path.join(parent_directory, f"policy/{args_cli.policy_name}/deploy_policy.yml")
    if os.path.exists(policy_config_path):
        with open(policy_config_path, "r", encoding="utf-8") as f:
            usr_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    else:
        usr_args = {}
        
    # Override with CLI args
    usr_args.update(vars(args_cli))
    
    # Load Task Config
    task_config_path = os.path.join(parent_directory, f"task_config/{args_cli.task_config}.yml")
    with open(task_config_path, "r", encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    task_args['task_name'] = args_cli.task_name
    task_args["task_config"] = args_cli.task_config
    task_args["ckpt_setting"] = args_cli.ckpt_setting
    task_args["render_freq"] = args_cli.render_freq # Force render freq
    task_args["eval_mode"] = True # Ensure step_lim is set
    
    # Embodiment Setup
    embodiment_type = task_args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    
    # Update usr_args with arm dims
    usr_args["left_arm_dim"] = len(task_args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(task_args["right_embodiment_config"]["arm_joints_name"][1])
    
    # Initialize Environment
    print(f"Initializing Task: {args_cli.task_name}")
    env = class_decorator(args_cli.task_name)
    
    # Initialize Policy
    print(f"Initializing Policy: {args_cli.policy_name}")
    policy_module = importlib.import_module(f"{args_cli.policy_name}.deploy_policy")
    get_model = getattr(policy_module, "get_model")
    encode_obs = getattr(policy_module, "encode_obs")
    
    # Ensure expert_data_num is set correctly
    if usr_args.get("expert_data_num") is None:
        usr_args["expert_data_num"] = args_cli.expert_data_num
        
    model = get_model(usr_args)
    
    # Setup Demo / Reset Env
    print(f"Setting up demo with seed {args_cli.seed}")
    env.setup_demo(now_ep_num=0, seed=args_cli.seed, is_test=True, **task_args)
    
    # Set Dummy Instruction
    env.set_instruction("do the task")
    
    # Reset Model
    model.env_runner.reset_obs()
    
    # Loop
    print("Starting Episode Loop...")
    step_count = 0
    
    # Initial Observation
    obs = env.get_obs()
    encoded_obs = encode_obs(obs)
    model.update_obs(encoded_obs)
    
    while env.take_action_cnt < env.step_lim:
        # Get Action Chunk
        actions = model.get_action()
        
        for i, action in enumerate(actions):
            step_count += 1
            print(f"\nStep: {step_count}")
            print(f"Action: {action}")
            
            # Print Observation Summary
            print("Observation Keys:", obs.keys())
            if 'qpos' in obs:
                print("QPos:", obs['qpos'])
            
            env.take_action(action)
            
            # Check success
            if env.check_success():
                print("\033[92mSuccess!\033[0m")
                break
                
            obs = env.get_obs()
            encoded_obs = encode_obs(obs)
            model.update_obs(encoded_obs)
            
            # Render is handled by env.take_action if render_freq > 0
            
        if env.check_success():
            break
            
    print("Episode Finished.")
    env.close_env()

if __name__ == "__main__":
    main()
