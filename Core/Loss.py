"""
Copyright (c) 2023 LuChen Wang
CANNDY is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import numpy as np


class Loss:
    """损失函数父类"""

    def __init__(self, model, truth, output):
        self.model = model
        self.truth = truth
        self.output = output
        if len(self.truth) != len(self.output):
            raise ValueError("The num of truth does not match the num of output")

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    @staticmethod
    def to_one_hot(x, num_class):
        """类别转为OneHot编码"""
        x = np.array(x, dtype=int)
        batch_size = x.shape[0]
        one_hot = np.zeros((batch_size, num_class))
        one_hot[np.arange(batch_size), x.flatten()] = 1
        return one_hot


class MSELoss(Loss):
    def __init__(self, model, truth, output):
        super(MSELoss, self).__init__(model, truth, output)
        if truth.shape != output.shape:
            raise ValueError("The shape of truth does not match the shape of output")

    def forward(self):
        """前向传播"""
        loss = np.sum(np.square(self.truth - self.output)) / self.truth.size
        return loss

    def backward(self):
        """反向传播"""
        delta = self.output - self.truth
        self.model.backward(delta)


class CrossEntropyWithSoftmax(Loss):
    def __init__(self, model, truth, output):
        super(CrossEntropyWithSoftmax, self).__init__(model, truth, output)
        self.num_samples, self.num_class = self.output.shape
        self.truth_one_hot = self.to_one_hot(self.truth, self.num_class)

    def forward(self):
        """前向传播"""
        loss = -np.sum(self.truth_one_hot * np.log(self.output + 1e-9)) / self.num_samples
        return loss

    def backward(self):
        """反向传播"""
        delta = self.output - self.truth_one_hot
        self.model.backward(delta)


class CrossEntropyWithSoftmaxMask(Loss):
    def __init__(self, model, truth, output, mask):
        super(CrossEntropyWithSoftmaxMask, self).__init__(model, truth, output)
        self.mask = mask
        _, self.num_class = self.output.shape
        self.num_samples = np.sum(self.mask)
        self.truth_one_hot = self.to_one_hot(self.truth, self.num_class)
        self.output[~self.mask, :] = 0
        self.truth_one_hot[~self.mask, :] = 0

    def forward(self):
        """前向传播"""
        loss = -np.sum(self.truth_one_hot * np.log(self.output + 1e-9)) / self.num_samples
        return loss

    def backward(self):
        """反向传播"""
        delta = self.output - self.truth_one_hot
        self.model.backward(delta)
