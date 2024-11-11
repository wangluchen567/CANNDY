import numpy as np


class Layer:
    """层级父类"""

    def __init__(self, input_size, output_size, activation, bias):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.bias = bias

    def forward(self, **kwargs):
        raise NotImplementedError

    def backward(self, **kwargs):
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
    """线性层"""

    def __init__(self, input_size, output_size, activation=None, bias=True):
        super(Linear, self).__init__(input_size, output_size, activation, bias)
        # 保存输入与输出
        self.input_1, self.output = None, None
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
        self.num_param = self.weight.size

    def zero_grad(self):
        """梯度置0"""
        self.grad = np.zeros(self.weight.shape)

    def forward(self, input_):
        """前向传播"""
        # 形状转置 (n,m) => (m,n)
        self.input_1 = input_.T.copy()
        if self.bias:
            self.input_1 = np.vstack((self.input_1, np.ones(shape=(1, self.input_1.shape[1]))))
        # H.T = W @ X.T or [W, b] @ [X, 1].T
        # 形状: (k,n) = (k,m) @ (m,n)
        self.output = self.weight @ self.input_1
        # 形状转置: (k,n) => (n,k)
        output_act = self.output.T.copy()
        # 激活函数激活
        if self.activation is not None:
            output_act = self.activation.forward(output_act)
        return output_act

    def backward(self, grad):
        """反向传播求梯度"""
        delta = grad
        if self.activation is not None:
            delta = grad * self.activation.backward(self.output).T
        # 计算batch大小
        batch_size = self.output.shape[1]
        # 计算梯度(累计梯度) 取平均
        self.grad += (self.input_1 @ delta).T / batch_size
        # 将delta @ w传递到上一层网络
        if self.bias:
            # 偏置求导被消掉了无需参与反向传播
            delta_next = delta @ self.weight[:, :-1]
        else:
            delta_next = delta @ self.weight
        return delta_next

    def set_parameters(self, weight):
        assert self.weight.shape == weight.shape
        self.weight = weight

    def get_parameters(self):
        return self.weight


class GraphConv(Linear):
    """图卷积层"""

    def __init__(self, input_size, output_size, adj_norm, activation=None):
        super(GraphConv, self).__init__(input_size, output_size, activation, False)
        self.adj_norm = adj_norm

    def forward(self, input_):
        output = super().forward(self.adj_norm @ input_)
        return output
