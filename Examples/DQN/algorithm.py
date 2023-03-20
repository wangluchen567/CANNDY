import copy
import numpy as np
from Core.Optimizer import Adam
from DQNLoss import MSELoss

class DQN(object):
    def __init__(self, model, act_dim, gamma, lr):
        """
        DQN algorithm
        :param model: (Model) 定义Q函数的网络结构
        :param act_dim: (int) action空间的维度，即有几个action
        :param gamma: (float) reward的衰减因子
        :param lr: (float) learning_rate，学习率
        """
        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

        self.model = model
        self.target_model = copy.deepcopy(model)

        # 使用Adam优化器
        self.optimizer = Adam(self.model, learning_rate=self.lr)

    def predict(self, obs):
        # 使用self.model的value网络(forward)来获取 [Q(s,a1),Q(s,a2),...]
        obs = obs.reshape(-1, 1)
        Q_value = self.model.forward(obs)
        Q_value = Q_value.flatten()
        return Q_value

    def learn(self, obs, action, reward, next_obs, terminal):
        # 使用DQN算法更新self.model的value网络

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model.forward(next_obs.T)
        best_v = np.max(next_pred_value, axis=0)

        # terminal中True为结束(最后一个记录了)，此时target = reward
        target = reward + (1.0 - terminal) * self.gamma * best_v
        # target = target.detach()  # 变为常量，阻止梯度传递

        # 比如 pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4], [...]]
        pred_value = self.model.forward(obs.T)  # 获取Q预测值
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = self.to_one_hot(action, self.act_dim)

        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        #  ==> pred_action_value = [[3.9]]
        pred_action_value = np.sum((action_onehot * pred_value.T), axis=1)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        Loss = MSELoss(self.model, target, pred_action_value, action_onehot)
        self.optimizer.zero_grad()
        mse_loss = Loss.forward()
        Loss.backward()
        self.optimizer.step()
        return mse_loss

    def sync_target(self):
        # 把 self.model 的模型参数值同步到 self.target_model
        self.target_model.load_params_list(self.model.get_params_list())

    # def to_one_hot(self, var, dim):
    #     var_dim = var.shape[0]
    #     one_hot = torch.zeros((var_dim, dim)).to(self.device)
    #     one_hot.scatter_(1, var.long().unsqueeze(-1), 1)
    #     return one_hot

    @staticmethod
    def to_one_hot(x, num_class):
        """转为OneHot编码"""
        batch_size = x.shape[0]
        one_hot = np.zeros((batch_size, num_class))
        one_hot[np.arange(batch_size), x.flatten()] = 1
        return one_hot

