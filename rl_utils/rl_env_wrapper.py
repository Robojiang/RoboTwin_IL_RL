import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RLPolicyEnvWrapper(gym.Env):
    def __init__(self, task_env, base_model):
        super(RLPolicyEnvWrapper, self).__init__()
        self.task_env = task_env
        self.base_model = base_model

        # 定义观察空间和动作空间
        obs = self.task_env.get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs["observation"]["il_action"].shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.task_env.robot.action_dim,), dtype=np.float32
        )

    def reset(self):
        obs = self.task_env.reset()
        return obs["observation"]

    def step(self, rl_action):
        # 获取模仿学习的动作
        il_action = self.base_model.get_action(self.task_env.get_obs())

        # 计算最终动作
        final_action = il_action + rl_action

        # 执行最终动作
        self.task_env.take_action(final_action)

        # 获取新的观察值
        obs = self.task_env.get_obs()

        # 计算奖励
        reward = self.task_env.compute_reward()

        # 检查是否完成
        done = self.task_env.check_success()

        return obs["observation"], reward, done, {}