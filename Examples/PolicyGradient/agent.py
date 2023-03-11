import numpy as np


class Agent(object):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.alg = algorithm
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def sample(self, obs):
        act_prob = self.alg.predict(obs)
        act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        act_prob = self.alg.predict(obs)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        # 在其他应用中可以考虑使用随机选择的策略（如sample函数）
        return act

    def learn(self, obs, action, reward):
        # 训练一次网络
        loss = self.alg.learn(obs, action, reward)
        return loss
