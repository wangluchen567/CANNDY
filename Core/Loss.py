import numpy as np


class Loss():
    def __init__(self, model, truth, output):
        self.model = model
        self.truth = truth
        self.output = output.T

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    @staticmethod
    def to_one_hot(x, num_class):
        """转为OneHot编码"""
        batch_size = x.shape[0]
        one_hot = np.zeros((batch_size, num_class))
        one_hot[np.arange(batch_size), x.flatten()] = 1
        return one_hot


class MSELoss(Loss):
    def __init__(self, model, truth, output):
        super(MSELoss, self).__init__(model, truth, output)

    def forward(self):
        """前向传播"""
        loss = 0.5 * np.sum(np.square(self.truth - self.output))
        return loss

    def backward(self):
        """反向传播"""
        delta = self.output - self.truth
        for i in range(-1, -self.model.num_layers - 1, -1):
            delta = self.model.Layers[i].backward(delta)


class CrossEntropyWithSoftmax(Loss):
    def __init__(self, model, truth, output):
        super(CrossEntropyWithSoftmax, self).__init__(model, truth, output)
        self.num_class = self.output.shape[1]
        self.truth_one_hot = self.to_one_hot(self.truth, self.num_class)

    def forward(self):
        """前向传播"""
        loss = -np.sum(self.truth_one_hot * np.log(self.output + 1e-9))
        return loss

    def backward(self):
        """反向传播"""
        delta = self.output - self.truth_one_hot
        for i in range(-1, -self.model.num_layers - 1, -1):
            delta = self.model.Layers[i].backward(delta)


class CrossEntropyWithSoftmaxMask(Loss):
    def __init__(self, model, truth, output, mask):
        super(CrossEntropyWithSoftmaxMask, self).__init__(model, truth, output)
        self.reverse_mask = np.array(1 - mask, dtype=bool)
        self.num_class = self.output.shape[1]
        self.truth_one_hot = self.to_one_hot(self.truth, self.num_class)
        self.output[self.reverse_mask, :] = 0
        self.truth_one_hot[self.reverse_mask, :] = 0

    def forward(self):
        """前向传播"""
        loss = -np.sum(self.truth_one_hot * np.log(self.output + 1e-9))
        return loss

    def backward(self):
        """反向传播"""
        delta = self.output - self.truth_one_hot
        for i in range(-1, -self.model.num_layers - 1, -1):
            delta = self.model.Layers[i].backward(delta)
