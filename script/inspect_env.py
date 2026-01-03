import sys
import os
import yaml
import numpy as np
import importlib
import cv2

# Add paths
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "description/utils"))

from envs import CONFIGS_PATH

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
    task_name = "beat_block_hammer"
    task_config_name = "demo_clean"
    gpu_id = 0
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Load Task Config
    task_config_path = os.path.join(parent_directory, f"task_config/{task_config_name}.yml")
    with open(task_config_path, "r", encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    task_args['task_name'] = task_name
    task_args["task_config"] = task_config_name
    task_args["render_freq"] = 1 # Enable rendering to ensure cameras are active
    task_args["eval_mode"] = True
    
    # Embodiment Setup (Simplified for brevity, assuming standard config)
    embodiment_type = task_args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        return _embodiment_types[embodiment_type]["file_path"]

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["dual_arm_embodied"] = True
    
    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    
    print(f"Initializing Task: {task_name}")
    env = class_decorator(task_name)
    
    print("Setting up demo...")
    env.setup_demo(now_ep_num=0, seed=0, is_test=True, **task_args)
    
    # Force an update
    env._update_render()
    env.cameras.update_picture()
    
    print("\n--- Environment Inspection ---")
    print(f"Data Type Config: {env.data_type}")
    
    # Check RGB
    rgb_dict = env.cameras.get_rgb()
    print(f"\nRGB Keys: {list(rgb_dict.keys())}")
    for k, v in rgb_dict.items():
        if isinstance(v, dict):
             print(f"  {k}: {v.keys()}") # Should be {'rgb': ...} or similar?
             if 'rgb' in v:
                 print(f"  {k}['rgb'] shape: {v['rgb'].shape}")
             elif 'rgba' in v:
                 print(f"  {k}['rgba'] shape: {v['rgba'].shape}")
        else:
             print(f"  {k} shape: {v.shape}")

    # Check Point Cloud
    print("\nChecking Point Cloud...")
    try:
        pcd = env.cameras.get_pcd(if_combine=True)
        if pcd is None:
            print("get_pcd returned None")
        else:
            print(f"get_pcd shape: {pcd.shape}")
            if len(pcd) > 0:
                print(f"Sample point: {pcd[0]}")
    except Exception as e:
        print(f"get_pcd failed: {e}")

    # Check Observation Dictionary
    obs = env.get_obs()
    print("\nObservation Dictionary Keys:", obs.keys())
    if 'pointcloud' in obs:
        print(f"obs['pointcloud'] type: {type(obs['pointcloud'])}")
        if isinstance(obs['pointcloud'], (list, np.ndarray)):
             print(f"obs['pointcloud'] len/shape: {len(obs['pointcloud'])}")

    env.close_env()

if __name__ == "__main__":
    main()