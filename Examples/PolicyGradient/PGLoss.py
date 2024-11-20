import numpy as np


class CrossEntropyWithSoftmax():
    def __init__(self, model, truth, output, reward):
        self.model = model
        self.truth = truth
        self.output = output
        self.reward = reward
        self.num_class = self.output.shape[1]
        self.truth_one_hot = self.to_one_hot(self.truth, self.num_class)

    def forward(self):
        """前向传播"""
        # loss = -R * (log(P) * one_hot(A)) = -R * CrossEntropy(P, A)
        loss = -self.reward * np.sum(self.truth_one_hot * np.log(self.output + 1e-9), axis=1)
        return loss

    def backward(self):
        """反向传播"""
        reward = self.reward.reshape(-1, 1).repeat(self.output.shape[1], 1)
        delta = reward * (self.output - self.truth_one_hot)
        for i in range(-1, -len(self.model.Layers) - 1, -1):
            delta = self.model.Layers[i].backward(delta)

    @staticmethod
    def to_one_hot(x, num_class):
        """转为OneHot编码"""
        batch_size = x.shape[0]
        one_hot = np.zeros((batch_size, num_class))
        one_hot[np.arange(batch_size), x.flatten()] = 1
        return one_hot
