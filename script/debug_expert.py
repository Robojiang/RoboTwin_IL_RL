
import sys
import os
import argparse
import yaml
import numpy as np
import importlib
from pathlib import Path

# Add paths
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
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

def main():
    parser = argparse.ArgumentParser(description="Debug Expert Script")
    parser.add_argument("--task_name", type=str, default="beat_block_hammer", help="Name of the task")
    parser.add_argument("--task_config", type=str, default="demo_clean", help="Task configuration file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--render_freq", type=int, default=20, help="Render frequency")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    
    args_cli = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu_id)
    
    # Load Task Config
    task_config_path = os.path.join(parent_directory, f"task_config/{args_cli.task_config}.yml")
    with open(task_config_path, "r", encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    task_args['task_name'] = args_cli.task_name
    task_args["task_config"] = args_cli.task_config
    task_args["render_freq"] = args_cli.render_freq
    task_args["eval_mode"] = True
    
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
    
    # Initialize Environment
    print(f"Initializing Task: {args_cli.task_name}")
    env = class_decorator(args_cli.task_name)
    
    # Setup Demo
    print(f"Setting up demo with seed {args_cli.seed}")
    env.setup_demo(now_ep_num=0, seed=args_cli.seed, is_test=True, **task_args)
    
    # Run Expert
    print("Running Expert Policy...")
    try:
        env.play_once()
        print("Expert finished.")
    except Exception as e:
        print(f"Expert failed: {e}")
        
    env.close_env()

if __name__ == "__main__":
    main()
