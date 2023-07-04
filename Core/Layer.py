import numpy as np


class Layer():
    def __init__(self, input_size, output_size, activation, bias):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.bias = bias

    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    @staticmethod
    def random_uniform(input_size, output_size, have_bias, lower=-1, upper=1):
        if have_bias:
            return np.random.uniform(lower, upper, size=(output_size, input_size + 1))
        else:
            return np.random.uniform(lower, upper, size=(output_size, input_size))

    @staticmethod
    def kaiming_uniform(input_size, output_size, have_bias, a=0):
        weight_bound = np.sqrt((6 / ((1 + a * a) * input_size)))
        bias_bound = 1 / np.sqrt(input_size)
        if have_bias:
            weight = np.zeros((output_size, input_size + 1))
            weight[:, :-1] = np.random.uniform(-weight_bound, weight_bound, size=(output_size, input_size))
            weight[:, -1] = np.random.uniform(-bias_bound, bias_bound, size=output_size)
            return weight
        else:
            weight = np.random.uniform(-weight_bound, weight_bound, size=(output_size, input_size))
            return weight


class Linear(Layer):
    def __init__(self, input_size, output_size, activation=None, bias=True):
        super(Linear, self).__init__(input_size, output_size, activation, bias)
        # 初始化权重
        # 随机初始化权重
        # self.weight = self.random_uniform(input_size, output_size, bias)
        # 何凯明的方法初始化权重
        self.weight = self.kaiming_uniform(input_size, output_size, bias)
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
        self.output = self.weight @ self.input_1
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
        # self.grad += (self.input_1 @ delta).T * batch_size
        # 计算梯度(累计梯度) 取平均下降较慢
        self.grad += (self.input_1 @ delta).T
        # 将delta * w传递到上一层网络
        if self.bias:
            # 偏置求导被消掉了无法反向传播
            delta_w = delta @ self.weight[:, :-1]
        else:
            delta_w = delta @ self.weight
        return delta_w

    def set_parameters(self, weight):
        assert self.weight.shape == weight.shape
        self.weight = weight

    def get_parameters(self):
        return self.weight


class GraphConv(Linear):
    def __init__(self, input_size, output_size, adj_norm, activation=None):
        super(GraphConv, self).__init__(input_size, output_size, activation, False)
        self.adj_norm = adj_norm

    def forward(self, input):
        input = input @ self.adj_norm.T
        output = super().forward(input)
        return output
