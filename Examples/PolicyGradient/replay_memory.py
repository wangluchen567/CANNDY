import random
import collections
import numpy as np


class ReplayMemory(object):
    """经验回放池"""

    def __init__(self, max_size):
        # 定义经验池（队列形式）
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        # 增加一条经验
        # 元组形式存储 (obs, action, reward, next_obs, done)
        self.buffer.append(exp)

    def sample(self, batch_size):
        # 从经验池中随机抽取一个batch的数据
        mini_batch = random.sample(self.buffer, batch_size)
        # 对每个元素进行分解，单独放到每个元素对应的数组中
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch), \
               np.array(action_batch), \
               np.array(reward_batch), \
               np.array(next_obs_batch), \
               np.array(done_batch)

    def __len__(self):
        return len(self.buffer)
