import torch
import numpy as np


class Agent(object):
    def __init__(self, algorithm, obs_dim, act_dim, e_greed=0.1, e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)

        self.alg = algorithm
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        pred_Q = self.alg.predict(obs)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, action, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        # 训练一次网络
        loss = self.alg.learn(obs, action, reward, next_obs, terminal)
        return loss