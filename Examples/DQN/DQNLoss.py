import numpy as np


class MSELoss():
    def __init__(self, model, truth, output, action):
        self.model = model
        self.truth = truth
        self.output = output
        self.action = action

    def forward(self):
        """前向传播"""
        loss = np.sum(np.square(self.truth - self.output)) / self.truth.size
        return loss

    def backward(self):
        """反向传播"""
        delta = self.output - self.truth
        delta = delta.reshape(-1, 1).repeat(self.action.shape[1], 1) * self.action
        for i in range(-1, -self.model.num_layers - 1, -1):
            delta = self.model.Layers[i].backward(delta)