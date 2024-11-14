import numpy as np


class Layer:
    """层级父类"""

    def __init__(self, input_size=None, output_size=None, activation=None, bias=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.bias = bias

    def __call__(self, *args, **kwargs):
        """方便直接使用对象名调用forward函数"""
        return self.forward(*args, **kwargs)

    def zero_grad(self):
        """梯度置为0矩阵"""
        raise NotImplementedError

    def get_parameters(self):
        """获取该层的权重参数"""
        raise NotImplementedError

    def set_parameters(self, *args, **kwargs):
        """设置该层的权重参数"""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """该层前向传播"""
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        """该层反向传播"""
        raise NotImplementedError

    def xavier_uniform_(self, matrix: np.ndarray, gain=1.0):
        """Xavier均匀分布随机初始化(适用于Sigmoid和Tanh函数)"""
        fan_in, fan_out = self.cal_fan_in_and_fan_out(matrix)
        bound = gain * np.sqrt(6.0 / float(fan_in + fan_out))
        return np.random.uniform(-bound, bound, matrix.shape)

    def xavier_normal_(self, matrix: np.ndarray, gain=1.0):
        """Xavier正态分布随机初始化(适用于Sigmoid和Tanh函数)"""
        fan_in, fan_out = self.cal_fan_in_and_fan_out(matrix)
        std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
        return np.random.normal(0., std, matrix.shape)

    def get_gain(self):
        """获取gain值"""
        if self.activation is None:
            return 1.0
        elif self.activation.__name__ == 'ReLU':
            return np.sqrt(2)
        elif self.activation.__name__ == 'Tanh':
            return 5 / 3
        else:
            return 1.0

    def kaiming_uniform_(self, matrix: np.ndarray, a=0, mode='fan_in', gain=1.0):
        """何凯明均匀分布随机初始化
        linear/sigmoid/conv/identity: gain = :math:`1`
        relu: gain = :math:`\\sqrt{2}`
        tanh: gain = :math:`\\frac{5}{3}`
        leaky_relu: gain = :math:`\\sqrt{\\frac{2}{1 + a^2}}`
        """
        fan_in, fan_out = self.cal_fan_in_and_fan_out(matrix)
        fan = fan_in if mode == 'fan_in' else fan_out
        bound = gain * np.sqrt(3.0) / np.sqrt((1 + a * a) * fan)
        return np.random.uniform(-bound, bound, matrix.shape)

    def kaiming_normal_(self, matrix: np.ndarray, a=0, mode='fan_in', gain=1.0):
        """何凯明正态分布随机初始化
        linear/sigmoid/conv/identity: gain = :math:`1`
        relu: gain = :math:`\\sqrt{2}`
        tanh: gain = :math:`\\frac{5}{3}`
        leaky_relu: gain = :math:`\\sqrt{\\frac{2}{1 + a^2}}`
        """
        fan_in, fan_out = self.cal_fan_in_and_fan_out(matrix)
        fan = fan_in if mode == 'fan_in' else fan_out
        std = gain / np.sqrt((1 + a * a) * fan)
        return np.random.normal(0., std, matrix.shape)

    @staticmethod
    def cal_fan_in_and_fan_out(matrix: np.ndarray):
        """计算扇入扇出值"""
        dimensions = matrix.ndim  # 矩阵维度
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for matrix with fewer than 2 dimensions")
        input_size = matrix.shape[1]  # 输入大小（或输入通道数）
        output_size = matrix.shape[0]  # 输出大小（或输出通道数）
        field_size = 1  # 感受野大小
        if dimensions > 2:
            field_size = np.size(matrix[0][0])
        # 计算扇入扇出值
        fan_in = input_size * field_size
        fan_out = output_size * field_size
        return fan_in, fan_out


class Linear(Layer):
    """线性层"""

    def __init__(self, input_size, output_size, activation=None, bias=True):
        super(Linear, self).__init__(input_size, output_size, activation, bias)
        # 保存输入与输出
        self.input_1, self.output = None, None
        # 初始化权重
        self.weight = np.zeros((self.output_size, self.input_size + self.bias))
        # 何凯明的方法初始化权重
        self.weight = self.kaiming_uniform_(self.weight, gain=self.get_gain())
        # 初始化激活函数
        if self.activation is not None:
            self.activation = activation()
        # 初始化梯度
        self.grad = np.zeros_like(self.weight)
        # 计算参数量
        self.num_param = self.weight.size

    def zero_grad(self):
        """梯度置0"""
        self.grad = np.zeros_like(self.weight)

    def get_parameters(self):
        return self.weight

    def set_parameters(self, weight):
        assert self.weight.shape == weight.shape
        self.weight = weight

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


class GraphConv(Linear):
    """图卷积层"""

    def __init__(self, input_size, output_size, adj_norm, activation=None):
        super(GraphConv, self).__init__(input_size, output_size, activation, False)
        self.adj_norm = adj_norm

    def forward(self, input_):
        output = super().forward(self.adj_norm @ input_)
        return output


class RNNCell(Layer):
    """RNN模块"""

    def __init__(self, input_size, output_size, activation=None, bias=True):
        super(RNNCell, self).__init__(input_size, output_size, activation, bias)
        # 保存整个过程中的输入与输出
        self.input_1_list, self.hidden_1_list, self.output_list = None, None, None
        self.init_empty()
        # 初始化权重
        self.weight_input = np.zeros((self.output_size, self.input_size + self.bias))
        self.weight_hidden = np.zeros((self.output_size, self.output_size + self.bias))
        # 使用Glorot Xavier的方法初始化权重
        self.weight_input = self.xavier_uniform_(self.weight_input)
        self.weight_hidden = self.xavier_uniform_(self.weight_hidden)
        # 初始化激活函数
        if self.activation is not None:
            self.activation = activation()
        # 对权重进行拼接以方便反向传播更新权重
        self.weight = np.hstack((self.weight_input, self.weight_hidden))
        # 记录拼接位置
        self.splice = self.weight_input.shape[1]
        # 初始化梯度
        self.grad = np.zeros_like(self.weight)
        # 计算参数量
        self.num_param = self.weight.size

    def init_empty(self):
        """初始化置空"""
        self.input_1_list, self.hidden_1_list, self.output_list = [], [], []

    def zero_grad(self):
        """梯度置0"""
        self.grad = np.zeros_like(self.weight)

    def get_parameters(self):
        return self.weight

    def set_parameters(self, weight):
        """设置参数值（权重值）"""
        assert self.weight.shape == weight.shape
        self.weight = weight
        self.weight_input = self.weight[:, :self.splice]
        self.weight_hidden = self.weight[:, self.splice:]

    def forward(self, input_, hidden):
        """前向传播"""
        # 形状转置 (n,m) => (m,n)
        input_1 = input_.T.copy()
        hidden_1 = hidden.T.copy()
        if self.bias:
            input_1 = np.vstack((input_1, np.ones(shape=(1, input_1.shape[1]))))
            hidden_1 = np.vstack((hidden_1, np.ones(shape=(1, hidden_1.shape[1]))))
        # H.T = W @ X.T or [W, b] @ [X, 1].T
        # 形状: (k,n) = (k,m) @ (m,n)
        output = self.weight_input @ input_1 + self.weight_hidden @ hidden_1
        # 保存所有的输入与输出
        self.input_1_list.append(input_1)
        self.hidden_1_list.append(hidden_1)
        self.output_list.append(output)
        # 形状转置: (k,n) => (n,k)
        output_act = output.T.copy()
        # 激活函数激活
        if self.activation is not None:
            output_act = self.activation.forward(output_act)
        return output_act

    def backward(self, grad):
        """反向传播求梯度"""
        delta = grad
        # 求梯度时需要从最后一个弹出一个元素求梯度
        input_1 = self.input_1_list.pop(-1)
        hidden_1 = self.hidden_1_list.pop(-1)
        output = self.output_list.pop(-1)
        if self.activation is not None:
            delta = grad * self.activation.backward(output).T
        # 计算batch大小
        batch_size = output.shape[1]
        # 计算各个权重的梯度并取平均
        grad_input = (input_1 @ delta).T / batch_size
        grad_hidden = (hidden_1 @ delta).T / batch_size
        # 计算梯度(累计梯度)
        self.grad += np.hstack((grad_input, grad_hidden))
        # 将delta @ w传递到上一层网络
        if self.bias:
            # 偏置求导被消掉了无需参与反向传播
            delta_next = delta @ self.weight_hidden[:, :-1]
        else:
            delta_next = delta @ self.weight_hidden
        return delta_next


class RNN(Layer):
    """RNN网络层"""

    def __init__(self, input_size, hidden_size, num_layers, activation=None, bias=True, batch_first=False):
        super(RNN, self).__init__(input_size, hidden_size, activation, bias)
        self.hidden_size = hidden_size
        self.num_layers = num_layers if num_layers > 1 else 1
        self.batch_first = batch_first
        self.num_steps = None
        # 保存输入、输出与隐状态以进行梯度优化
        self.inputs, self.outputs, self.state = None, None, None
        # 根据给定层数创建网络层
        self.Layers = [RNNCell(self.input_size, self.hidden_size, self.activation, self.bias)]
        if self.num_layers > 1:
            self.Layers.extend(
                [RNNCell(self.hidden_size, self.hidden_size, self.activation, self.bias)
                 for _ in range(self.num_layers - 1)])

    def zero_grad(self):
        for layer in self.Layers:
            layer.zero_grad()

    def get_parameters(self):
        weights = []
        for layer in self.Layers:
            weights.append(layer.get_parameters())

    def set_parameters(self, weights):
        assert len(weights) == len(self.Layers)
        for i in range(len(weights)):
            self.Layers[i].set_parameters(weights[i])

    def init_state(self, batch_size):
        """初始化第一个隐状态（默认为全0矩阵）"""
        return np.zeros((self.num_layers, batch_size, self.hidden_size))

    def forward(self, inputs, state=None):
        """前向传播"""
        # 若是指定batch在第一维则需要修改形状
        if self.batch_first:
            self.inputs = inputs.transpose(1, 0, 2).copy()
        else:
            self.inputs = inputs.copy()
        # input size: (num_steps, batch_size, feature_size)
        # output size : (num_steps, batch_size, hidden_size)
        # 获取输入的矩阵形状
        self.num_steps, batch_size, feature_size = self.inputs.shape
        # 初始化输出矩阵
        self.outputs = np.zeros((self.num_steps, batch_size, self.hidden_size))
        # 初始化第一个隐状态
        self.state = state
        if self.state is None:
            self.state = self.init_state(batch_size)
        # 为了不累计Cell中的输入输出，需要先初始化置空
        for i in range(self.num_layers):
            self.Layers[i].init_empty()
        # 保存模型输出
        for i in range(self.num_steps):
            self.state[0] = self.Layers[0].forward(self.inputs[i], self.state[0])
            for j in range(1, self.num_layers):
                self.state[j] = self.Layers[j].forward(self.state[j - 1], self.state[j])
            self.outputs[i] = self.state[-1]
        # 输出模型
        # output size : (num_steps, batch_size, hidden_size)
        # state size : (num_layers, batch_size, hidden_size)
        return self.outputs, self.state

    def backward(self, grad):
        """梯度反向传播"""
        # 初始化每层返回的delta梯度
        grads = [np.zeros_like(grad) for _ in range(self.num_layers)]
        grads[-1] = grad
        for i in range(self.num_steps):
            # 先反向传播最后一层
            grads[-1] = self.Layers[-1].backward(grads[-1])
            # 再逐层传播并对上一个时间片的梯度累加
            for j in range(self.num_layers - 2, -1, -1):
                grads[j] += self.Layers[j].backward(grads[j + 1])

