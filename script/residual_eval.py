import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
import importlib
import argparse

from stable_baselines3 import PPO

# --- 从 residual_train.py 复制过来的辅助函数 ---
def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

# --- 核心评估函数 ---
def eval_residual_policy(
    task_name,
    TASK_ENV,
    args,
    il_model,
    rl_model,  # 新增 RL 模型参数
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None
):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']} (IL+RL)\033[0m")

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    now_id = 0
    succ_seed = 0
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    # IL 模型重置函数
    reset_il_model = eval_function_decorator(args["policy_name"], "reset_model")

    # 观察值编码函数 (用于 RL 模型)
    def encode_obs_for_rl(observation):
        obs = dict()
        obs['agent_pos'] = np.array(observation['joint_action']['vector'], dtype=np.float32)
        obs['point_cloud'] = np.array(observation['pointcloud'], dtype=np.float32)
        return obs

    while succ_seed < test_num:
        try:
            # 从 args 中移除冲突的 seed 参数
            eval_args = args.copy()
            eval_args.pop("seed", None)  # 确保 seed 不会重复传递
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **eval_args)
        except UnStableError as e:
            print(f"Unstable seed {now_seed}, skipping. Error: {e}")
            TASK_ENV.close_env()
            now_seed += 1
            continue
        
        succ_seed += 1
        
        # --- 评估循环开始 ---
        succ = False
        reset_il_model(il_model)  # 重置 IL 模型状态
        
        # 获取 IL 模型的第一个动作块
        observation = TASK_ENV.get_obs()
        il_model.update_obs(encode_obs_for_rl(observation))
        il_actions = il_model.get_action()
        il_action_index = 0

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            # 1. 获取 IL 动作
            if il_action_index >= len(il_actions):
                obs_for_il = TASK_ENV.get_obs()
                il_model.update_obs(encode_obs_for_rl(obs_for_il))
                il_actions = il_model.get_action()
                il_action_index = 0
            il_action = il_actions[il_action_index]
            il_action_index += 1

            # 2. 获取 RL 残差动作
            obs_for_rl = encode_obs_for_rl(TASK_ENV.get_obs())
            rl_action, _ = rl_model.predict(obs_for_rl, deterministic=True)

            # 3. 组合动作用于执行
            final_action = il_action + rl_action
            TASK_ENV.take_action(final_action)

            if TASK_ENV.eval_success:
                succ = True
                break
        
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc

# --- 主函数 ---
def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    policy_name = usr_args["policy_name"]
    rl_model_path = usr_args["rl_model_path"]  # 获取 RL 模型路径

    # --- 加载配置 ---
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

   # 更新配置
    args['task_name'] = task_name
    args["task_config"] = task_config
    args["policy_name"] = policy_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    # 关键部分：确保设置 dual_arm_embodied
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True  # 这行是关键
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False  # 这行是关键
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    # 创建环境
    TASK_ENV = class_decorator(args["task_name"])

    # 加载 IL 模型
    get_il_model = eval_function_decorator(policy_name, "get_model")
    il_model = get_il_model(usr_args)
    print(f"IL Model '{policy_name}' loaded.")

    # 加载 RL 模型
    rl_model = PPO.load(rl_model_path)
    print(f"RL Model loaded from: {rl_model_path}")

    # 运行评估
    st_seed = 100000 * (1 + usr_args["seed"])
    _, suc_num = eval_residual_policy(
        task_name,
        TASK_ENV,
        args,
        il_model,
        rl_model,
        st_seed,
        test_num=100
    )
    
    print(f"\nFinal Success Rate: {suc_num}/100")

def parse_args_and_config(path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the IL policy config file.")
    parser.add_argument("--rl_model_path", type=str, help="Path to the trained residual RL model (.zip file).")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["rl_model_path"] = path

    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try: value = eval(value)
            except: pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config

if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    path = './rl_models/beat_block_hammer/DP3/demo_tao_for_dp3/2025-10-13_13-00-44/residual_policy.zip'
    usr_args = parse_args_and_config(path)
    main(usr_args)