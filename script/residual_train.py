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
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import random
# 初始化 wandb
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


class RLPolicyEnvWrapper(gym.Env):
    def __init__(self, task_env, il_model, encode_obs_func, args):
        """
        将自定义环境包装为符合 SB3 的 gym.Env 接口。
        :param task_env: 自定义环境实例（如 beat_block_hammer）
        :param il_model: 模仿学习模型，用于预测动作块
        :param encode_obs_func: 将原始观察值编码为模型输入格式的函数
        :param args: 环境配置参数字典
        """
        super(RLPolicyEnvWrapper, self).__init__()
        self.task_env = task_env
        self.il_model = il_model
        self.encode_obs_func = encode_obs_func
        self.args = args.copy()  # 复制参数字典
        
        # 设置为评估模式
        self.args["eval_mode"] = True
        
        # 初始化环境跟踪变量
        self.episode_id = 0
        self.steps_taken = 0
        self.max_steps = args.get("step_lim", 300)
        self.episode_count = 0  # 用于跟踪完成的episode数
        self.reset_func = None
        self.clear_cache_freq = args.get("clear_cache_freq", 10)
        
        # IL模型相关
        self.il_actions = None
        self.il_action_index = 0
        
        # 定义观察空间
        self.observation_space = spaces.Dict({
            "point_cloud": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1024, 6), dtype=np.float32
            ),
            "agent_pos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
            ),
        })
        
        # 定义动作空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(14,), dtype=np.float32
        )
    
    def set_reset_func(self, reset_func):
        """设置模型重置函数"""
        self.reset_func = reset_func
        
    def reset(self, *, seed=None, options=None):
        """
        重置环境，使用 setup_demo 替代标准的 reset。
        :param seed: 随机数种子
        :param options: 重置选项
        :return: 初始观察值, 信息字典
        """
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 生成环境随机种子
        env_seed = random.randint(0, 10000) if seed is None else seed
        
        # 尝试初始化环境，最多尝试10次
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # 禁用渲染（训练时不需要渲染）
                render_freq = self.args.get("render_freq", 0)
                self.args["render_freq"] = 0
                
                # 初始化环境
                self.task_env.setup_demo(
                    now_ep_num=self.episode_id,
                    seed=env_seed + attempt, 
                    is_test=True, 
                    **self.args
                )
                
                # 成功初始化，跳出循环
                break
            except UnStableError as e:
                print(f"Attempt {attempt+1}/{max_attempts}: Unstable object detected with seed {env_seed + attempt}. Trying a different seed.")
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to initialize environment after {max_attempts} attempts") from e
        
        # 重置模型（如果需要）
        if self.reset_func is not None:
            self.reset_func(self.il_model)
        
        # 获取观察值
        obs = self.task_env.get_obs()
        encoded_obs = {
            "agent_pos": np.array(obs["joint_action"]["vector"], dtype=np.float32),
            "point_cloud": np.array(obs["pointcloud"], dtype=np.float32)
        }
        
        # 初始化 IL 动作
        self.il_model.update_obs(self.encode_obs_func(obs))
        self.il_actions = self.il_model.get_action()
        self.il_action_index = 0
        
        # 重置状态变量
        self.steps_taken = 0
        self.episode_id += 1
        
        # 恢复渲染设置
        self.args["render_freq"] = render_freq
        
        # 返回观察值和信息字典 (符合 Gymnasium API)
        info = {}
        return encoded_obs, info
        
    def step(self, rl_action):
        """
        执行一步动作，结合 IL 动作和 RL 残差动作。
        :param rl_action: RL 模型预测的残差动作
        """
        # 获取当前 IL 动作
        if self.il_action_index >= len(self.il_actions):
            # 获取新的 IL 动作块
            obs = self.task_env.get_obs()
            self.il_model.update_obs(self.encode_obs_func(obs))
            self.il_actions = self.il_model.get_action()
            self.il_action_index = 0
            
        il_action = self.il_actions[self.il_action_index]
        
        # 计算最终动作（IL 动作 + RL 残差）
        final_action = il_action + rl_action
        
        # 执行动作
        self.task_env.take_action(final_action)
        
        # 获取新的观察值
        obs = self.task_env.get_obs()
        encoded_obs = {
            "agent_pos": np.array(obs["joint_action"]["vector"], dtype=np.float32),
            "point_cloud": np.array(obs["pointcloud"], dtype=np.float32)
        }
        
        # 计算奖励
        reward = self.compute_reward()
        
        # 检查是否完成
        success = self.task_env.check_success()
        # 分解 done 为 terminated 和 truncated
        terminated = success  # 任务成功完成时为 True
        truncated = (self.steps_taken >= self.max_steps)  # 达到最大步数时为 True
        
        # 更新状态
        self.steps_taken += 1
        self.il_action_index += 1
        
        # 如果episode结束（terminated或truncated），关闭环境并可能清理缓存
        if terminated or truncated:
            self.episode_count += 1
            
            # 关闭环境，并根据频率决定是否清理缓存
            should_clear_cache = (self.episode_count % self.clear_cache_freq == 0)
            self.task_env.close_env(clear_cache=should_clear_cache)
        
        # 返回结果
        info = {"success": success}
        return encoded_obs, reward, terminated, truncated, info
    
    def compute_reward(self):
        """
        计算奖励函数。
        """
        # 调用环境的 compute_reward 方法（如果存在）
        if hasattr(self.task_env, 'compute_reward'):
            return self.task_env.compute_reward()
        
        # 否则使用基于成功状态的简单奖励
        success = self.task_env.check_success()
        
        if success:
            # 成功完成任务的奖励
            return 10.0
        elif self.steps_taken >= self.max_steps:
            # 达到最大步数的惩罚
            return -1.0
        else:
            # 每步的小惩罚，鼓励尽快完成
            return 0
            
    def close(self):
        """关闭环境并清理资源"""
        if hasattr(self.task_env, 'close_env'):
            self.task_env.close_env(clear_cache=True)  # 环境关闭时总是清理缓存

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

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def train_rl(usr_args):
    # 设置训练目录和日志
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    policy_name = usr_args["policy_name"]
    
    # 创建保存目录
    save_dir = Path(f"rl_models/{task_name}/{policy_name}/{task_config}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型和环境
    TASK_ENV = class_decorator(task_name)
    get_model = eval_function_decorator(policy_name, "get_model")
    reset_func = eval_function_decorator(policy_name, "reset_model")
    
    # 加载配置
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # 更新配置
    args['task_name'] = task_name
    args["task_config"] = task_config
    args["policy_name"] = policy_name
    
    # 处理机器人配置和相机配置
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    
    # 更新usr_args中的机器人维度
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    
    # 加载模仿学习模型
    il_model = get_model(usr_args)
    
    # 定义观察值编码函数
    def encode_obs(observation):
        obs = dict()
        obs['agent_pos'] = observation['joint_action']['vector']
        obs['point_cloud'] = observation['pointcloud']
        return obs
    
    # 使用RL包装器包装环境
    rl_env = RLPolicyEnvWrapper(TASK_ENV, il_model, encode_obs, args)
    rl_env.set_reset_func(reset_func)  # 设置模型重置函数
    
    # 设置RL模型参数
    total_timesteps = usr_args.get("total_timesteps", 100000)
    learning_rate = usr_args.get("learning_rate", 0.0003)
    seed = usr_args.get("seed", None)
    
    

    # 创建wandb回调函数
    class WandbCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(WandbCallback, self).__init__(verbose)
            self.eval_rewards = []
            self.eval_interval = 2000  # 每隔多少步评估一次
            self.last_eval_step = 0
        
        def _on_step(self):
            # 记录训练指标
            if len(self.model.ep_info_buffer) > 0:
                ep_reward_mean = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                ep_len_mean = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                
                wandb.log({
                    "train/reward": ep_reward_mean,
                    "train/episode_length": ep_len_mean,
                    "train/global_step": self.num_timesteps
                })
            
            # 周期性评估模型 - 这是获取真实成功率的好方法
            if self.num_timesteps - self.last_eval_step >= self.eval_interval:
                self.last_eval_step = self.num_timesteps
                # 这里可以实现一个简单的评估循环，但我们省略它
                # 实际上应该设置一个单独的评估环境并收集统计数据
            
            return True
     # 创建wandb回调
    wandb_callback = WandbCallback()
    
        # 在train_rl函数中，创建检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  # 每5000步保存一次
        save_path=f"{save_dir}/checkpoints/",
        name_prefix="ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 将多个回调组合在一起
    callbacks = CallbackList([wandb_callback, checkpoint_callback])
    
    # 初始化wandb项目
    run = wandb.init(
        project=f"robotwin-rl-{task_name}",
        config={
            "task": task_name,
            "config": task_config,
            "policy": policy_name,
            "learning_rate": learning_rate,
            "total_timesteps": total_timesteps,
            "date": current_time,
            "seed": seed
        },
        name=f"{task_name}_{task_config}_{current_time}"
    )
    
    # 创建RL模型
    print("Creating RL model...")
    ppo_model = PPO(
        "MultiInputPolicy", 
        rl_env, 
        verbose=1,
        learning_rate=learning_rate,
        seed=seed
    )
    
   
    
    # 训练RL模型
    print(f"Starting RL training for {total_timesteps} steps...")
    ppo_model.learn(
        total_timesteps=total_timesteps, 
        callback=callbacks
    )
    
    # 保存训练好的模型
    model_save_path = os.path.join(save_dir, "residual_policy")
    ppo_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # 完成wandb记录
    wandb.finish()
    
    return model_save_path

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    model_path = train_rl(usr_args)
    print(f"Training completed. Model saved at: {model_path}")

    # main(usr_args)
