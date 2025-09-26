import gym
from gym import spaces
import numpy as np
import copy
import math


def inverse_softmax(x, T=1.0):
    x = -x / T
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # 按列做 softmax
    return e_x / np.sum(e_x, axis=0, keepdims=True)  # 按列归一化


# 离散动作空间
class Env(gym.Env):
    """
    自定义环境：时间序列预测
    状态空间：包含两个部分
        S1: 历史x天的时间序列（连续空间）
        S2: 预测结果和真实结果（连续空间）
    动作空间：智能体的预测结果（下一个时间步的预测值）
    """

    def __init__(self, time_series, target_time_series, window_size, train_size, val_size, hidden_dim, history_dim,
                 num_states, num_nodes, feature_num,
                 max_timesteps):
        super(Env, self).__init__()

        self.time_series = time_series
        self.target_time_series = target_time_series
        self.max_steps = len(time_series)  # 最大步数等于时间序列的长度
        self.max_timesteps = max_timesteps
        self.window_size = window_size
        self.train_size = int(self.max_steps * train_size)
        self.val_size = self.train_size
        # self.val_size2 = int(self.max_steps * val_size / 2)
        # self.test_size = self.max_steps - self.train_size - self.val_size2
        self.test_size = self.max_steps - 1000
        self.hidden_dim = hidden_dim
        self.history_dim = history_dim
        self.num_states = num_states
        self.num_nodes = num_nodes
        self.actionhz = {}
        self.probhz = {}
        self.errorhz = {}
        self.choicehz = {}
        self.last_action = np.ones(num_nodes) * -9999
        self.con_actions = np.ones(num_nodes)
        self.con_actions_theta = 8
        self.con_theta = 0.02
        self.con_theta_episode_count = 0
        self.con_theta_max = 0.1
        self.chajubs = 1
        self.score1bs = 1
        self.score2bs = 1
        self.future_weight = 0.5
        self.feature_num = feature_num
        self.com_p = np.ones((self.train_size - self.max_timesteps - np.max(self.window_size) - 1, self.num_nodes)) * (
            -99999.0)
        self.sort_p = np.mean(inverse_softmax(self.com_p), axis=1)

        self.action_space = spaces.Discrete(self.num_states)

        # 定义状态空间
        # S1: 当前及历史观测
        # S2: 当前隐状态
        # S3: 上一时刻置信
        # S4: 上一时刻error
        self.observation_space = spaces.Dict({
            'S1': spaces.Box(low=-1, high=1, shape=(self.num_nodes, window_size[0], self.feature_num),
                             dtype=np.float32),
            'S3': spaces.Box(low=-1, high=1, shape=(self.num_nodes, self.history_dim, self.num_states),
                             dtype=np.float32),
            'S4': spaces.Box(low=-1, high=1, shape=(self.num_nodes, self.history_dim, self.num_states),
                             dtype=np.float32),
            'S5': spaces.Box(low=-1, high=1, shape=(self.num_nodes, 1, self.num_states),
                             dtype=np.float32),
        })

        self.current_step = window_size  # 初始化当前步（确保有足够的历史数据生成状态）

    def reset(self, type):
        self.last_action = np.ones(self.num_nodes) * -9999
        self.type = type
        if type == 0:
            # print(self.sort_p.shape)
            self.current_step = np.random.choice(
                range(np.max(self.window_size) + 1, self.train_size - self.max_timesteps), size=1, replace=True,
                p=self.sort_p)[0]  # 随机从某一步开始
            self.current_start = self.current_step
            self.max_steps = self.current_step + self.max_timesteps
            self.con_actions = np.ones(self.num_nodes)
        elif type == 1:
            self.current_step = np.max(self.window_size) + 1
            self.max_steps = self.train_size
        else:
            self.current_step = 1000
            self.max_steps = len(self.time_series)

    def step(self, sort_p):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        if done:
            if self.type == 0:
                self.sort_p = np.average(sort_p, axis=0)
