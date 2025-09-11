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


class Activation:
    """激活函数父类"""

    def __call__(self, *args, **kwargs):
        """方便直接使用对象名调用forward函数"""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x):
        return x * (x > 0)

    def backward(self, x):
        return x > 0


class Sigmoid(Activation):
    def forward(self, x):
        # 防止指数溢出
        # y = 1 / (1 + exp(-x)), x >= 0
        # y = exp(x) / (1 + exp(x)), x < 0
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))


class Tanh(Activation):
    def forward(self, x):
        # 防止指数溢出
        # y = (1-exp(-2*x))/(1+exp(-2*x)), x >= 0
        # y = (exp(2*x)-1)/(1+exp(2*x)), x < 0
        return np.where(x >= 0,
                        (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x)),
                        (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1))

    def backward(self, x):
        return 1 - self.forward(x) * self.forward(x)


class Softmax(Activation):
    def forward(self, x, dim=1):
        # 因为在求exp时，可能因为指数过大，出现溢出的情况
        # 而在softmax中，重要的是两个数字之间的差值，只要差值相同，softmax的结果就相同
        x -= np.max(x, axis=dim, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)
        return y

    def backward(self, x):
        # Softmax的梯度反向传播集成在CrossEntropyWithSoftmax中了
        return np.ones(x.shape)
