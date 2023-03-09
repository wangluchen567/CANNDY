import numpy as np


class Linear():
    def __init__(self, input_size, output_size, activation=None, bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.bias = bias
        # 初始化权重
        self.weight = np.random.uniform(-1, 1, size=(output_size, input_size))
        if self.bias:
            self.weight = np.random.uniform(-1, 1, size=(output_size, input_size + 1))
        # 初始化激活函数
        if self.activation is not None:
            self.activation = activation()
        # 初始化梯度
        self.grad = np.zeros(self.weight.shape)
        # 计算参数量
        self.num_param = (input_size + 1) * output_size

    def zero_grad(self):
        """梯度置0"""
        self.grad = np.zeros(self.weight.shape)

    def forward(self, input):
        """前向传播"""
        self.input_1 = input.copy()
        if self.bias:
            self.input_1 = np.vstack((self.input_1, np.ones(shape=(1, self.input_1.shape[1]))))
        # H = W * X or H = [W, b] * [X, 1].T
        self.output = np.matmul(self.weight, self.input_1)
        output_act = self.output.copy()
        # 激活函数激活
        if self.activation is not None:
            output_act = self.activation.forward(output_act)
        return output_act

    def backward(self, grad):
        """反向传播求梯度"""
        delta = grad
        if self.activation is not None:
            delta = grad * self.activation.backward(self.output).T
        # # 计算batch大小
        # batch_size = self.output.shape[1]
        # # 计算梯度(累计梯度)
        # self.grad += np.matmul(self.input_1, delta).T / batch_size
        # 计算梯度(累计梯度) 取平均下降较慢
        self.grad += np.matmul(self.input_1, delta).T
        # 将delta * w传递到上一层网络
        if self.bias:
            # 偏置求导被消掉了无法反向传播
            delta_w = np.matmul(delta, self.weight[:, :-1])
        else:
            delta_w = np.matmul(delta, self.weight)
        return delta_w
