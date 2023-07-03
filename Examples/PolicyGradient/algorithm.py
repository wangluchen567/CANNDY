import numpy as np
from Core.Optimizer import Adam
from PGLoss import CrossEntropyWithSoftmax


class PolicyGradient(object):
    def __init__(self, model, lr):
        """
        Policy Gradient algorithm
        :param model: (Model) policy的前向网络
        :param lr: (float) 学习率
        """
        self.model = model
        assert isinstance(lr, float)
        self.lr = lr
        # 使用Adam优化器
        self.optimizer = Adam(self.model, learning_rate=self.lr)

    def predict(self, obs):
        # 使用policy model预测输出的动作概率
        obs = obs.reshape(1, -1)
        act_prob = self.model.forward(obs)
        act_prob = act_prob.flatten()
        return act_prob

    def learn(self, obs, action, reward):
        # 用policy gradient 算法更新policy model
        act_prob = self.model.forward(obs)  # 获取输出动作概率
        self.optimizer.zero_grad()
        Loss = CrossEntropyWithSoftmax(self.model, action, act_prob, reward)
        loss = Loss.forward()
        loss = np.mean(loss)
        Loss.backward()
        self.optimizer.step()
        return loss
