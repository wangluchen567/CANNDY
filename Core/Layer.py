import numpy as np


class Layer:
    """层级父类"""

    def __init__(self, input_size=None, output_size=None, activation=None, bias=False):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.bias = bias
        # 当前是否是训练模式
        self.training = True
        self.num_params = 0

    def __call__(self, *args, **kwargs):
        """方便直接使用对象名调用forward函数"""
        return self.forward(*args, **kwargs)

    def zero_grad(self):
        """梯度置为0矩阵"""
        pass

    def get_parameters(self):
        """获取该层的权重参数"""
        pass

    def set_parameters(self, *args, **kwargs):
        """设置该层的权重参数"""
        pass

    def get_num_params(self):
        """获取该层的参数数量"""
        pass

    def forward(self, *args, **kwargs):
        """该层前向传播"""
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        """该层反向传播"""
        raise NotImplementedError

    def xavier_uniform_(self, matrix: np.ndarray, gain=1.0, bias=True):
        """Xavier均匀分布随机初始化(适用于Sigmoid和Tanh函数)"""
        fan_in, fan_out = self.cal_fan_in_and_fan_out(matrix, bias)
        bound = gain * np.sqrt(6.0 / float(fan_in + fan_out))
        return np.random.uniform(-bound, bound, matrix.shape)

    def xavier_normal_(self, matrix: np.ndarray, gain=1.0, bias=True):
        """Xavier正态分布随机初始化(适用于Sigmoid和Tanh函数)"""
        fan_in, fan_out = self.cal_fan_in_and_fan_out(matrix, bias)
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

    def kaiming_uniform_(self, matrix: np.ndarray, a=0, mode='fan_in', gain=1.0, bias=True):
        """何凯明均匀分布随机初始化
        linear/sigmoid/conv/identity: gain = :math:`1`
        relu: gain = :math:`\\sqrt{2}`
        tanh: gain = :math:`\\frac{5}{3}`
        leaky_relu: gain = :math:`\\sqrt{\\frac{2}{1 + a^2}}`
        """
        fan_in, fan_out = self.cal_fan_in_and_fan_out(matrix, bias)
        fan = fan_in if mode == 'fan_in' else fan_out
        bound = gain * np.sqrt(3.0) / np.sqrt((1 + a * a) * fan)
        return np.random.uniform(-bound, bound, matrix.shape)

    def kaiming_normal_(self, matrix: np.ndarray, a=0, mode='fan_in', gain=1.0, bias=True):
        """何凯明正态分布随机初始化
        linear/sigmoid/conv/identity: gain = :math:`1`
        relu: gain = :math:`\\sqrt{2}`
        tanh: gain = :math:`\\frac{5}{3}`
        leaky_relu: gain = :math:`\\sqrt{\\frac{2}{1 + a^2}}`
        """
        fan_in, fan_out = self.cal_fan_in_and_fan_out(matrix, bias)
        fan = fan_in if mode == 'fan_in' else fan_out
        std = gain / np.sqrt((1 + a * a) * fan)
        return np.random.normal(0., std, matrix.shape)

    @staticmethod
    def cal_fan_in_and_fan_out(matrix: np.ndarray, bias=True):
        """计算扇入扇出值"""
        dimensions = matrix.ndim  # 矩阵维度
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for matrix with fewer than 2 dimensions")
        input_size = matrix.shape[1] - bias  # 输入大小（或输入通道数）
        output_size = matrix.shape[0]  # 输出大小（或输出通道数）
        field_size = 1  # 感受野大小
        if dimensions > 2:
            field_size = np.size(matrix[0][0])
        # 计算扇入扇出值
        fan_in = input_size * field_size
        fan_out = output_size * field_size
        return fan_in, fan_out

    @staticmethod
    def init_param2d(param):
        """初始化给定二维参数(用于二维卷积与池化)"""
        if isinstance(param, int):
            return param, param
        elif isinstance(param, tuple) or isinstance(param, list):
            if len(param) != 2:
                raise ValueError("param must be 2-dim tuples or lists")
            return param
        else:
            raise ValueError("param must be int or 2-dim tuples or lists")

    @staticmethod
    def padding_mat1d(matrix: np.ndarray, padding):
        """对一维矩阵进行padding填充(填充0)"""
        assert matrix.ndim == 3
        if isinstance(padding, int):  # 整数填充
            # 在最后一个维度两侧均匀填充padding个0
            return np.pad(matrix, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        elif (isinstance(padding, tuple) or isinstance(padding, list)) and len(padding) == 2:  # 两个方向的元组填充
            # 在最后一个维度两侧各填充padding[0]个0和padding[1]个0
            return np.pad(matrix, ((0, 0), (0, 0), (padding[0], padding[1])), mode='constant')
        else:
            raise ValueError("The padding parameter must be an integer or "
                             "a tuple or list in the form of (left, right)")

    @staticmethod
    def padding_cut1d(matrix: np.ndarray, padding):
        """剪裁一维矩阵的padding部分"""
        assert matrix.ndim == 3
        if isinstance(padding, int):
            return matrix[:, :, padding:matrix.shape[2] - padding]
        elif (isinstance(padding, tuple) or isinstance(padding, list)) and len(padding) == 2:
            return matrix[:, :, padding[0]:matrix.shape[2] - padding[1]]
        else:
            raise ValueError("The padding parameter must be an integer or "
                             "a tuple or list in the form of (left, right)")

    @staticmethod
    def padding_mat2d(matrix: np.ndarray, padding):
        """对二维矩阵进行padding填充(填充0)"""
        assert matrix.ndim == 4
        if isinstance(padding, int):  # 整数填充
            # 在最后两个维度四周均匀填充padding个0
            return np.pad(matrix, ((0, 0), (0, 0), (padding, padding),
                                   (padding, padding)), mode='constant')
        elif (isinstance(padding, tuple) or isinstance(padding, list)) and len(padding) == 2:  # 两个方向的元组填充
            # 在高度两侧各填充padding[0]个0，在宽度两侧各填充padding[1]个0
            return np.pad(matrix, ((0, 0), (0, 0), (padding[0], padding[0]),
                                   (padding[1], padding[1])), mode='constant')
        elif (isinstance(padding, tuple) or isinstance(padding, list)) and len(padding) == 4:  # 四个方向的元组填充
            # 分别在上下左右填充padding[0], padding[1], padding[2], padding[3]个0
            return np.pad(matrix, ((0, 0), (0, 0), (padding[0], padding[1]),
                                   (padding[2], padding[3])), mode='constant')
        else:
            raise ValueError("The padding parameter must be an integer or "
                             "a tuple or list in the form of (height, width) or (top, bottom, left, right)")

    @staticmethod
    def padding_cut2d(matrix: np.ndarray, padding):
        """剪裁二维矩阵的padding部分"""
        assert matrix.ndim == 4
        _, _, s2, s3 = matrix.shape
        if isinstance(padding, int):
            return matrix[:, :, padding:s2 - padding, padding:s3 - padding]
        elif (isinstance(padding, tuple) or isinstance(padding, list)) and len(padding) == 2:
            return matrix[:, :, padding[0]:s2 - padding[0], padding[1]:s3 - padding[1]]
        elif (isinstance(padding, tuple) or isinstance(padding, list)) and len(padding) == 4:
            return matrix[:, :, padding[0]:s2 - padding[1], padding[2]:s3 - padding[3]]
        else:
            raise ValueError("The padding parameter must be an integer or "
                             "a tuple or list in the form of (height, width) or (top, bottom, left, right)")


class Linear(Layer):
    """线性层"""

    def __init__(self, input_size, output_size, activation=None, bias=True):
        super(Linear, self).__init__(input_size, output_size, activation, bias)
        # 保存输入与输出以及batch大小
        self.input_1, self.output, self.batch_size = None, None, 1
        # 初始化权重
        self.weight = np.zeros((self.output_size, self.input_size + self.bias))
        # 何凯明的方法初始化权重
        self.weight = self.kaiming_uniform_(self.weight, gain=self.get_gain(), bias=self.bias)
        # 实例化激活函数
        if self.activation is not None:
            self.activation = self.activation()
        # 初始化梯度
        self.grad = np.zeros_like(self.weight)
        # 计算参数量
        self.num_params = self.weight.size

    def zero_grad(self):
        """梯度置0"""
        self.grad = np.zeros_like(self.weight)

    def get_parameters(self):
        """获取该层权重参数"""
        return self.weight.tolist()

    def set_parameters(self, weight_):
        """设置该层权重参数"""
        # 将权重变为array类型
        weight = weight_ if isinstance(weight_, np.ndarray) else np.array(weight_)
        assert self.weight.shape == weight.shape
        self.weight = weight

    def forward(self, input_):
        """前向传播"""
        # 记录batch大小
        self.batch_size = input_.shape[0]
        # 形状转置 (n,d) => (d,n)
        self.input_1 = input_.T.copy()
        if self.bias:
            self.input_1 = np.vstack((self.input_1, np.ones(shape=(1, self.input_1.shape[1]))))
        # Y = [X 1] @ [W b]^T = X @ W + b
        # 形状: (n,c) = ((c,d) @ (d,n)).T
        self.output = (self.weight @ self.input_1).T
        # 激活函数激活
        output_ = self.output.copy()
        if self.activation is not None:
            output_ = self.activation.forward(output_)
        return output_

    def backward(self, delta):
        """反向传播求梯度"""
        if self.activation is not None:
            delta = delta * self.activation.backward(self.output)
        # 计算梯度(累积梯度) 取平均
        # 形状: (c,d) = ((d,n) @ (n,c)).T
        self.grad += (self.input_1 @ delta).T / self.batch_size
        # 将delta @ w传递到上一层网络
        if self.bias:
            # 偏置与上一层无关，无需参与反向传播
            delta_next = delta @ self.weight[:, :-1]
        else:
            delta_next = delta @ self.weight
        return delta_next


class Dropout(Layer):
    """随机失活层"""

    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.p = p  # 随机失活概率(丢弃神经元比例)
        self.mask = None  # 记录被dropout的部分

    def forward(self, input_):
        """前向传播"""
        output = input_.copy()
        # 若失活概率非0且是训练模式则进行mask
        if self.p > 0 and self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=output.shape)
            # 进行dropout操作
            output = output * self.mask / (1 - self.p)
        return output

    def backward(self, delta):
        """反向传播"""
        # 若失活概率非0且是训练模式则需要对梯度进行mask
        if self.p > 0 and self.training:
            return delta * self.mask / (1 - self.p)
        else:
            return delta


class GCNConv(Linear):
    """图卷积层"""

    def __init__(self, input_size, output_size, adj_norm, activation, bias=True):
        super(GCNConv, self).__init__(input_size, output_size, activation, bias)
        self.adj_norm = adj_norm
        self.weight = self.xavier_uniform_(self.weight, bias=self.bias)

    def forward(self, input_):
        output = super().forward(self.adj_norm @ input_)
        return output


class RNNCell(Layer):
    """循环神经网络模块"""

    def __init__(self, input_size, output_size, activation=None, bias=True):
        super(RNNCell, self).__init__(input_size, output_size, activation, bias)
        # 保存整个过程中的输入与输出
        self.input_1_list, self.hidden_1_list, self.output_list = [], [], []
        # 记录batch大小
        self.batch_size = 1
        # 初始化权重
        self.weight_input = np.zeros((self.output_size, self.input_size + self.bias))
        self.weight_hidden = np.zeros((self.output_size, self.output_size + self.bias))
        # 使用Glorot Xavier的方法初始化权重
        self.weight_input = self.xavier_uniform_(self.weight_input, bias=self.bias)
        self.weight_hidden = self.xavier_uniform_(self.weight_hidden, bias=self.bias)
        # 实例化激活函数
        if self.activation is not None:
            self.activation = self.activation()
        # 对权重进行拼接以方便反向传播更新权重
        self.weight = np.hstack((self.weight_input, self.weight_hidden))
        # 记录拼接位置
        self.splice = self.weight_input.shape[1]
        # 初始化梯度
        self.grad = np.zeros_like(self.weight)
        # 计算参数量
        self.num_params = self.weight.size

    def init_empty(self):
        """初始化置空"""
        self.input_1_list, self.hidden_1_list, self.output_list = [], [], []

    def zero_grad(self):
        """梯度置0"""
        self.grad = np.zeros_like(self.weight)

    def get_parameters(self):
        return self.weight.tolist()

    def set_parameters(self, weight_):
        """设置参数值（权重值）"""
        # 将权重变为array类型
        weight = weight_ if isinstance(weight_, np.ndarray) else np.array(weight_)
        assert self.weight.shape == weight.shape
        self.weight = weight
        self.weight_input = self.weight[:, :self.splice]
        self.weight_hidden = self.weight[:, self.splice:]

    def forward(self, input_, hidden):
        """前向传播"""
        # 记录batch大小
        self.batch_size = input_.shape[0]
        # 形状转置 (n,d) => (d,n)
        input_1 = input_.T.copy()
        hidden_1 = hidden.T.copy()
        if self.bias:
            input_1 = np.vstack((input_1, np.ones(shape=(1, input_1.shape[1]))))
            hidden_1 = np.vstack((hidden_1, np.ones(shape=(1, hidden_1.shape[1]))))
        # 形状: (n,c) = ((c,d) @ (d,n) +  (c,c) @ (c,n)).T
        output = (self.weight_input @ input_1 + self.weight_hidden @ hidden_1).T
        # 保存所有的输入与输出
        self.input_1_list.append(input_1)
        self.hidden_1_list.append(hidden_1)
        self.output_list.append(output)
        # 激活函数激活
        output_ = output.copy()
        if self.activation is not None:
            output_ = self.activation.forward(output_)
        return output_

    def backward(self, delta):
        """反向传播求梯度"""
        # 求梯度时需要从最后一个弹出一个元素求梯度
        input_1 = self.input_1_list.pop(-1)
        hidden_1 = self.hidden_1_list.pop(-1)
        output = self.output_list.pop(-1)
        if self.activation is not None:
            delta = delta * self.activation.backward(output)
        # 计算各个权重的梯度并取平均
        # 形状: (c,d) = ((d,n) @ (n,c)).T
        grad_input = (input_1 @ delta).T / self.batch_size
        # 形状: (c,c) = ((c,n) @ (n,c)).T
        grad_hidden = (hidden_1 @ delta).T / self.batch_size
        # 计算梯度(累积梯度)
        self.grad += np.hstack((grad_input, grad_hidden))
        # 将delta @ w传递到上一层网络
        if self.bias:
            # 偏置与上一层无关，无需参与反向传播
            delta_next = delta @ self.weight_hidden[:, :-1]
        else:
            delta_next = delta @ self.weight_hidden
        return delta_next


class RNN(Layer):
    """循环神经网络层"""

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
        self.get_num_params()  # 获取该层参数数量

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

    def get_num_params(self):
        for layer in self.Layers:
            self.num_params += layer.num_params

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
        # 为了不累积Cell中的输入输出，需要先初始化置空
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

    def backward(self, delta):
        """梯度反向传播"""
        # 初始化每层返回的delta梯度
        deltas = [np.zeros_like(delta) for _ in range(self.num_layers)]
        deltas[-1] = delta
        for i in range(self.num_steps):
            # 先反向传播最后一层
            deltas[-1] = self.Layers[-1].backward(deltas[-1])
            # 再逐层传播并对上一个时间片的梯度累加
            for j in range(self.num_layers - 2, -1, -1):
                deltas[j] += self.Layers[j].backward(deltas[j + 1])


class Flatten(Layer):
    """展平层:用于连接全连接层的模块"""

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_):
        """前向传播"""
        # 记录输入的形状
        self.input_shape = input_.shape
        # 注意要按照batch_size进行flatten
        batch_size = input_.shape[0]
        output = input_.reshape(batch_size, -1)
        return output

    def backward(self, delta):
        """反向传播"""
        # 将梯度变为输入的形状反向传播到上一层
        return delta.reshape(self.input_shape)


class Conv1d(Layer):
    """一维卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None, bias=True):
        super().__init__(activation=activation, bias=bias)
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        self.ci = in_channels
        self.co = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 记录输入输出以及batch大小以方便反向传播
        self.input, self.output, self.batch_size = None, None, 1
        # 初始化权重
        self.weight = np.zeros((self.co, self.ci + self.bias, self.kernel_size))
        # 何凯明的方法初始化权重
        self.weight = self.kaiming_uniform_(self.weight, mode='fan_out', gain=self.get_gain(), bias=bias)
        # 实例化激活函数
        if self.activation is not None:
            self.activation = self.activation()
        # 初始化梯度
        self.grad = np.zeros_like(self.weight)
        # 计算参数数量
        self.num_params = self.weight.size

    def zero_grad(self):
        self.grad = np.zeros_like(self.weight)

    def get_parameters(self):
        return self.weight.tolist()

    def set_parameters(self, weight_):
        # 将权重变为array类型
        weight = weight_ if isinstance(weight_, np.ndarray) else np.array(weight_)
        assert self.weight.shape == weight.shape
        self.weight = weight

    def forward(self, input_):
        """前向传播"""
        # input_shape (NCZ格式):
        # (num_data/batch_size, in_channels, size)
        if input_.ndim != 3:
            raise ValueError("Conv1d can only handle 3-dim data")
        # 根据给定的padding得到加入padding后的输入
        self.input = self.padding_mat1d(input_, self.padding)
        # 得到输入的形状
        nd, ci, iz = self.input.shape
        self.batch_size = nd  # batch_size === num_data
        # 如果使用偏置则需要加入对应偏置
        if self.bias:
            bias_ = np.ones((nd, 1, iz))
            self.input = np.concatenate((self.input, bias_), axis=1)
        # 下面正式对卷积进行计算
        # 获取核函数形状值及步长大小
        kz, sz, co = self.kernel_size, self.stride, self.co
        # 计算输出特征图形状
        oz = int(np.floor((iz - kz + sz) / sz))
        # 初始化输出的特征图 (NCZ格式)
        self.output = np.zeros((nd, co, oz))
        # 进行卷积操作，计算输出
        for z in range(oz):
            i = z * sz
            # 形状: (nd, cip, kz)
            input_part = self.input[:, :, i:i + kz]
            # (nd, co) <= (nd, cip, kz) dot (co, cip, kz)
            self.output[:, :, z] = np.tensordot(input_part, self.weight, axes=([1, 2], [1, 2]))
        return self.output

    def backward(self, delta):
        """反向传播"""
        if self.activation is not None:
            delta = delta * self.activation.backward(self.output)
        # 获取梯度的形状(与输出的形状相同)
        nd, co, oz = delta.shape
        # 核函数大小和步长大小
        kz, sz = self.kernel_size, self.stride
        # 初始化传播到上一层梯度的形状
        delta_next = np.zeros_like(self.input, dtype=delta.dtype)
        weight_ = self.weight
        # 如果存在偏置需要去掉额外通道
        # 偏置不参与反向传播
        if self.bias:
            delta_next = delta_next[:, :-1, :]
            weight_ = weight_[:, :-1, :]
        # 进行卷积操作求梯度
        for z in range(oz):
            i = z * sz
            # 形状: (nd, cip, kz)
            input_part = self.input[:, :, i:i + kz]
            # (co, cip, kz) <= (nd, co) dot (nd, cip, kz)
            self.grad += np.tensordot(delta[:, :, z], input_part, axes=([0], [0]))
            # (nd, ci, kz) <= (nd, co) dot (co, ci, kz)
            delta_next[:, :, i:i + kz] += np.tensordot(delta[:, :, z], weight_, axes=([1], [0]))
        # 还要考虑padding的梯度，由于是常数0填充，所以这里直接裁剪掉padding
        delta_next = self.padding_cut1d(delta_next, self.padding)
        return delta_next


class Conv2d(Layer):
    """二维卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None, bias=True):
        super().__init__(activation=activation, bias=bias)
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        self.ci = in_channels
        self.co = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 记录输入输出以及batch大小以方便反向传播
        self.input, self.output, self.batch_size = None, None, 1
        # 根据给定参数初始化相关参数
        self.kh, self.kw = self.init_param2d(self.kernel_size)
        self.sh, self.sw = self.init_param2d(self.stride)
        # 初始化权重
        self.weight = np.zeros((self.co, self.ci + self.bias, self.kh, self.kw))
        # 何凯明的方法初始化权重
        self.weight = self.kaiming_uniform_(self.weight, mode='fan_out', gain=self.get_gain(), bias=bias)
        # 实例化激活函数
        if self.activation is not None:
            self.activation = self.activation()
        # 初始化梯度
        self.grad = np.zeros_like(self.weight)
        # 计算参数数量
        self.num_params = self.weight.size

    def zero_grad(self):
        self.grad = np.zeros_like(self.weight)

    def get_parameters(self):
        return self.weight.tolist()

    def set_parameters(self, weight_):
        # 将权重变为array类型
        weight = weight_ if isinstance(weight_, np.ndarray) else np.array(weight_)
        assert self.weight.shape == weight.shape
        self.weight = weight

    def forward(self, input_):
        """前向传播"""
        # input_shape (NCHW格式):
        # (num_data/batch_size, in_channels, height, width)
        if input_.ndim != 4:
            raise ValueError("Conv2d can only handle 4-dim data")
        # 根据给定的padding得到加入padding后的输入
        self.input = self.padding_mat2d(input_, self.padding)
        # 得到输入的形状
        nd, ci, ih, iw = self.input.shape
        self.batch_size = nd  # batch_size === num_data
        # 如果使用偏置则需要加入对应偏置
        if self.bias:
            bias_ = np.ones((nd, 1, ih, iw))
            self.input = np.concatenate((self.input, bias_), axis=1)
        # 下面正式对卷积进行计算
        # 获取核函数形状值及步长大小
        kh, kw, sh, sw, co = self.kh, self.kw, self.sh, self.sw, self.co
        # 计算输出特征图形状
        oh = int(np.floor((ih - kh + sh) / sh))
        ow = int(np.floor((iw - kw + sw) / sw))
        # 初始化输出的特征图 (NCHW格式)
        self.output = np.zeros((nd, co, oh, ow))
        # 进行卷积操作，计算输出
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (nd, cip, kh, kw)
                input_part = self.input[:, :, i:i + kh, j:j + kw]
                # (nd, co) <= (nd, cip, kh, kw) dot (co, cip, kh, kw)
                self.output[:, :, h, w] = np.tensordot(input_part, self.weight, axes=([1, 2, 3], [1, 2, 3]))
        return self.output

    def backward(self, delta):
        """反向传播"""
        if self.activation is not None:
            delta = delta * self.activation.backward(self.output)
        # 获取梯度的形状(与输出的形状相同)
        nd, co, oh, ow = delta.shape
        # 核函数大小和步长大小
        kh, kw, sh, sw = self.kh, self.kw, self.sh, self.sw
        # 初始化传播到上一层梯度的形状
        delta_next = np.zeros_like(self.input, dtype=delta.dtype)
        weight_ = self.weight
        # 如果存在偏置需要去掉额外通道
        # 偏置不参与反向传播
        if self.bias:
            delta_next = delta_next[:, :-1, :, :]
            weight_ = weight_[:, :-1, :, :]
        # 进行卷积操作求梯度
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (nd, cip, kh, kw)
                input_part = self.input[:, :, i:i + kh, j:j + kw]
                # (co, cip, kh, kw) <= (nd, co) dot (nd, cip, kh, kw)
                self.grad += np.tensordot(delta[:, :, h, w], input_part, axes=([0], [0]))
                # (nd, ci, kh, kw) <= (nd, co) dot (co, ci, kh, kw)
                delta_next[:, :, i:i + kh, j:j + kw] += np.tensordot(delta[:, :, h, w], weight_, axes=([1], [0]))
        # 还要考虑padding的梯度，由于是常数0填充，所以这里直接裁剪掉padding
        delta_next = self.padding_cut2d(delta_next, self.padding)
        return delta_next


class MaxPool1d(Layer):
    """一维最大池化层"""

    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 记录输入输出以方便反向传播
        self.input, self.output = None, None
        self.max_index = None  # 记录最大池化时的位置

    def forward(self, input_):
        """前向传播"""
        # input_shape (NCZ格式):
        # (num_data/batch_size, in_channels, size)
        if input_.ndim != 3:
            raise ValueError("MaxPool1d can only handle 3-dim data")
        # 根据给定的padding得到加入padding后的输入
        self.input = self.padding_mat1d(input_, self.padding)
        # 获取形状为ZNC格式的输入数据
        input_t = self.input.transpose(2, 0, 1)
        # 下面正式计算最大池化
        # 获取各种形状与步长
        nd, ci, iz = self.input.shape
        kz, sz, co = self.kernel_size, self.stride, ci
        # 计算输出特征图形状
        oz = int(np.floor((iz - kz + sz) / sz))
        # 初始化输出的特征图 (NCZ格式)
        self.output = np.zeros((nd, co, oz))
        # 初始化梯度mask下标
        self.max_index = np.zeros((oz, nd * ci, 1), dtype=int)
        # 进行卷积操作，同时记录mask下标
        for z in range(oz):
            i = z * sz
            # 形状: (kz, nd, ci)
            input_part = input_t[i:i + kz, :, :]
            # 将后两维合并以方便操作 (kz, nd*ci)
            input_part_ = input_part.reshape(kz, -1)
            # 进行最大池化
            part_max = input_part_.max(axis=0)
            # 得到该区域最大池化的结果
            self.output[:, :, z] = part_max.reshape(nd, ci)
            # 得到该区域所有最大值的位置
            part_max_pos = np.where(input_part_ == part_max)
            # 得到该区域第一个最大值的下标的位置
            _, unique_index = np.unique(part_max_pos[-1], return_index=True)
            # 得到该区域所有最大值的下标
            part_max_index = np.vstack(part_max_pos).T[unique_index, :-1]
            # 保存这些最大值下标
            self.max_index[z] = part_max_index
        return self.output

    def backward(self, delta):
        """反向传播"""
        # 获取梯度的形状(与输出的形状相同)
        nd, co, oz = delta.shape
        # 初始化反向传播到上一层的梯度
        delta_next = np.zeros_like(self.input, dtype=delta.dtype)
        # 核函数大小和步长大小
        kz, sz = self.kernel_size, self.stride
        for z in range(oz):
            i = z * sz
            # 形状: (nd, ci, kz)
            delta_next_part = delta_next[:, :, i:i + kz]
            # 调整轴以方便操作
            delta_next_part = delta_next_part.reshape(-1, kz)
            # 获取该部分的最大值下标
            part_max_index = self.max_index[z]
            # 将前面的梯度利用mask传递到上一层
            delta_next_part[np.arange(len(part_max_index)), part_max_index[:, 0]] += delta[:, :, z].flatten()
        # 还要考虑padding的梯度，由于是常数0填充，所以这里直接裁剪掉padding
        delta_next = self.padding_cut1d(delta_next, self.padding)
        return delta_next


class MaxPool2d(Layer):
    """二维最大池化层"""

    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 记录输入输出以方便反向传播
        self.input, self.output = None, None
        # 根据给定参数初始化相关参数
        self.kh, self.kw = self.init_param2d(self.kernel_size)
        self.sh, self.sw = self.init_param2d(self.stride)
        self.max_index = None  # 记录最大池化时的位置

    def forward(self, input_):
        """前向传播"""
        # input_shape (NCHW格式):
        # (num_data/batch_size, in_channels, height, width)
        if input_.ndim != 4:
            raise ValueError("MaxPool2d can only handle 4-dim data")
        # 根据给定的padding得到加入padding后的输入
        self.input = self.padding_mat2d(input_, self.padding)
        # 获取形状为HWNC格式的输入数据
        input_t = self.input.transpose(2, 3, 0, 1)
        # 下面正式计算最大池化
        # 获取各种形状与步长
        nd, ci, ih, iw = self.input.shape
        kh, kw, sh, sw, co = self.kh, self.kw, self.sh, self.sw, ci
        # 计算输出特征图形状
        oh = int(np.floor((ih - kh + sh) / sh))
        ow = int(np.floor((iw - kw + sw) / sw))
        # 初始化输出的特征图
        self.output = np.zeros((nd, co, oh, ow))
        # 初始化梯度mask下标
        self.max_index = np.zeros((oh, ow, nd * ci, 2), dtype=int)
        # 进行卷积操作，同时记录mask下标
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (kh, kw, nd, ci)
                input_part = input_t[i:i + kh, j:j + kw, :, :]
                # 将后两维合并以方便操作 (kh, kw, nd*ci)
                input_part_ = input_part.reshape(kh, kw, -1)
                # 进行最大池化
                part_max = input_part_.max(axis=(0, 1))
                # 得到该区域最大池化的结果
                self.output[:, :, h, w] = part_max.reshape(nd, ci)
                # 得到该区域所有最大值的位置
                part_max_pos = np.where(input_part_ == part_max)
                # 得到该区域第一个最大值的下标的位置
                _, unique_index = np.unique(part_max_pos[-1], return_index=True)
                # 得到该区域所有最大值的下标
                part_max_index = np.vstack(part_max_pos).T[unique_index, :-1]
                # 保存这些最大值下标
                self.max_index[h][w] = part_max_index
        return self.output

    def backward(self, delta):
        """反向传播"""
        # 获取梯度的形状(与输出的形状相同)
        nd, co, oh, ow = delta.shape
        # 初始化反向传播到上一层的梯度
        delta_next = np.zeros_like(self.input, dtype=delta.dtype)
        # 核函数大小和步长大小
        kh, kw, sh, sw = self.kh, self.kw, self.sh, self.sw
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (nd, ci, kh, kw)
                delta_next_part = delta_next[:, :, i:i + kh, j:j + kw]
                # 将后两维合并并调整轴以方便操作
                delta_next_part = delta_next_part.reshape(-1, kh, kw)
                # 获取该部分的最大值下标
                part_max_index = self.max_index[h][w]
                # 将前面的梯度利用mask传递到上一层
                delta_next_part[np.arange(len(part_max_index)), part_max_index[:, 0], part_max_index[:, 1]] \
                    += delta[:, :, h, w].flatten()
        # 还要考虑padding的梯度，由于是常数0填充，所以这里直接裁剪掉padding
        delta_next = self.padding_cut2d(delta_next, self.padding)
        return delta_next


class MeanPool1d(Layer):
    """一维平均池化层"""

    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 记录输入输出以方便反向传播
        self.input, self.output = None, None

    def forward(self, input_):
        """前向传播"""
        # input_shape (NCZ格式):
        # (num_data/batch_size, in_channels, size)
        if input_.ndim != 3:
            raise ValueError("MeanPool1d can only handle 3-dim data")
        # 根据给定的padding得到加入padding后的输入
        self.input = self.padding_mat1d(input_, self.padding)
        # 下面正式计算平均池化
        # 获取各种形状与步长
        nd, ci, iz = self.input.shape
        kz, sz, co = self.kernel_size, self.stride, ci
        # 计算输出特征图形状
        oz = int(np.floor((iz - kz + sz) / sz))
        # 初始化输出的特征图
        self.output = np.zeros((nd, co, oz))
        # 进行卷积操作
        for z in range(oz):
            i = z * sz
            # 形状: (nd, ci, kz)
            input_part = self.input[:, :, i:i + kz]
            # 取这部分的平均值作为平均池化的输出结果
            self.output[:, :, z] = input_part.mean(axis=-1)
        return self.output

    def backward(self, delta):
        """反向传播"""
        # 获取梯度的形状(与输出的形状相同)
        nd, co, oz = delta.shape
        # 初始化反向传播到上一层的梯度
        delta_next = np.zeros_like(self.input)
        # 核函数大小和步长大小
        kz, sz = self.kernel_size, self.stride
        for z in range(oz):
            i = z * sz
            # 形状: (nd, ci, kz)
            delta_next_part = delta_next[:, :, i:i + kz]
            # 调整形状以便操作
            delta_next_part = delta_next_part.reshape(nd, -1, kz)
            # 计算对应元素的平均梯度值
            delta_mean = delta[:, :, z] / kz
            # 将前面的梯度传递到上一层
            delta_next_part += np.repeat(delta_mean[:, :, np.newaxis], kz, axis=-1)
        # 还要考虑padding的梯度，由于是常数0填充，所以这里直接裁剪掉padding
        delta_next = self.padding_cut1d(delta_next, self.padding)
        return delta_next


class MeanPool2d(Layer):
    """二维平均池化层"""

    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 记录输入输出以方便反向传播
        self.input, self.output = None, None
        # 根据给定参数初始化相关参数
        self.kh, self.kw = self.init_param2d(self.kernel_size)
        self.sh, self.sw = self.init_param2d(self.stride)

    def forward(self, input_):
        """前向传播"""
        # input_shape (NCHW格式):
        # (num_data/batch_size, in_channels, height, width)
        if input_.ndim != 4:
            raise ValueError("MeanPool2d can only handle 4-dim data")
        # 根据给定的padding得到加入padding后的输入
        self.input = self.padding_mat2d(input_, self.padding)
        # 下面正式计算平均池化
        # 获取各种形状与步长
        nd, ci, ih, iw = self.input.shape
        kh, kw, sh, sw, co = self.kh, self.kw, self.sh, self.sw, ci
        # 计算输出特征图形状
        oh = int(np.floor((ih - kh + sh) / sh))
        ow = int(np.floor((iw - kw + sw) / sw))
        # 初始化输出的特征图
        self.output = np.zeros((nd, co, oh, ow))
        # 进行卷积操作
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (nd, ci, kh, kw)
                input_part = self.input[:, :, i:i + kh, j:j + kw]
                # 取这部分的平均值作为平均池化的输出结果
                self.output[:, :, h, w] = input_part.mean(axis=(2, 3))
        return self.output

    def backward(self, delta):
        """反向传播"""
        # 获取梯度的形状(与输出的形状相同)
        nd, co, oh, ow = delta.shape
        # 初始化反向传播到上一层的梯度
        delta_next = np.zeros_like(self.input)
        # 核函数大小和步长大小
        kh, kw, sh, sw = self.kh, self.kw, self.sh, self.sw
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (nd, ci, kh, kw)
                delta_next_part = delta_next[:, :, i:i + kh, j:j + kw]
                # 调整形状以便操作
                delta_next_part = delta_next_part.reshape(nd, -1, (kh * kw))
                # 计算对应元素的平均梯度值
                delta_mean = delta[:, :, h, w] / (kh * kw)
                # 将前面的梯度传递到上一层
                delta_next_part += np.repeat(delta_mean[:, :, np.newaxis], (kh * kw), axis=-1)
        # 还要考虑padding的梯度，由于是常数0填充，所以这里直接裁剪掉padding
        delta_next = self.padding_cut2d(delta_next, self.padding)
        return delta_next


class BatchNorm(Layer):
    """批归一化层"""

    def __init__(self, num_features, eps=1.e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        # 当前批次的数据数量、均值与方差
        self.mean, self.var = None, None
        # 需要记录全局的均值与方差(使用EMA算法)
        self.running_mean = np.zeros(self.num_features)
        self.running_var = np.ones(self.num_features)
        # 记录输入值和归一化变换后的输入值
        self.input, self.input_hat = None, None
        # 初始化权重
        self.weight = np.zeros((2, self.num_features))
        self.weight[0] += 1  # gamma值初始化为1
        self.weight[1] += 0  # beta值初始化为0
        # 初始化梯度
        self.grad = np.zeros_like(self.weight)

    def zero_grad(self):
        """梯度置为0矩阵"""
        self.grad = np.zeros_like(self.weight)

    def get_parameters(self):
        """获取该层的参数"""
        params = [self.weight.tolist(), self.running_mean.tolist(), self.running_var.tolist()]
        return params

    def set_parameters(self, params):
        """设置该层的参数"""
        if isinstance(params, np.ndarray):
            # 若输入是array类型则说明是weight
            assert self.weight.shape == params.shape
            self.weight = params
        elif isinstance(params, list):
            # 若输入是list类型则说明是为多个参数赋值
            weight = params[0] if isinstance(params[0], np.ndarray) else np.array(params[0])
            mean = params[1] if isinstance(params[1], np.ndarray) else np.array(params[1])
            var = params[2] if isinstance(params[2], np.ndarray) else np.array(params[2])
            assert self.weight.shape == weight.shape
            assert self.running_mean.shape == mean.shape
            assert self.running_var.shape == var.shape
            self.weight = weight
            self.running_mean = mean
            self.running_var = var
        else:
            raise ValueError("This parameter type is not supported")

    def forward(self, input_):
        """前向传播"""
        self.input = input_.copy()
        # 在训练阶段需要使用当前的均值和方差
        if self.training:
            self.mean = np.mean(self.input, axis=0)  # 计算均值
            self.var = np.var(self.input, axis=0)  # 计算有偏方差
            m = self.input.shape[0]  # 给的数据的数量
            # 使用EMA算法更新运行过程中的均值和无偏方差
            self.running_mean = self.momentum * self.mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * (m / (m - 1)) * self.var + (1 - self.momentum) * self.running_var
        else:  # 测试阶段使用全局的均值与无偏方差
            self.mean = self.running_mean
            self.var = self.running_var
        # 进行归一化变换
        self.input_hat = (self.input - self.mean) / np.sqrt(self.var + self.eps)
        gamma = self.weight[0]
        beta = self.weight[1]
        # 进行仿射变换
        output = self.input_hat * gamma + beta
        return output

    def backward(self, delta):
        """反向传播"""
        # 计算梯度(累积梯度)
        self.grad += np.vstack((np.sum(delta * self.input_hat, axis=0), np.sum(delta, axis=0)))
        m = delta.shape[0]  # 数据的数量
        gamma = self.weight[0]
        # 计算要用到的导数
        diff_input_hat = delta * gamma
        diff_var = np.sum(diff_input_hat * (self.input - self.mean) * -0.5 * (self.var + self.eps) ** (-3 / 2), axis=0)
        diff_mean = np.sum(diff_input_hat * -1 / np.sqrt(self.var + self.eps), axis=0)
        diff_input = (diff_input_hat / np.sqrt(self.var + self.eps)
                      + diff_var * 2 * (self.input - self.var) / m
                      + diff_mean / m)
        delta_next = diff_input
        # 将梯度传递到上一层网络
        return delta_next


class BatchNorm2d(BatchNorm):
    """批归一化层(用于卷积网络中)"""

    def __init__(self, num_features, eps=1.e-5, momentum=0.1):
        super().__init__(num_features, eps, momentum)

    def forward(self, input_):
        """前向传播"""
        # input_size: (num_data/batch_size, in_channels, height, width) (NCHW格式)
        if input_.ndim != 4:
            raise ValueError("BatchNorm2d can only handle 4-dim data")
        # 获取输入矩阵的形状
        num_data, channel, height, width = input_.shape
        # 将通道数放在最后面后reshape: (N,H,W,C) => (N*H*W, C)
        input_reshape = input_.transpose(0, 2, 3, 1).reshape(-1, channel)
        output = super().forward(input_reshape)
        # 需要将输出再还原为原来的形状
        output = output.reshape(num_data, height, width, channel).transpose(0, 3, 1, 2)
        return output

    def backward(self, delta):
        """反向传播"""
        # delta: (num_data/batch_size, in_channels, height, width) (NCHW格式)
        # 获取输入矩阵的形状
        num_data, channel, height, width = delta.shape
        # 将通道数放在最后面后reshape: (N,H,W,C) => (N*H*W, C)
        delta_reshape = delta.transpose(0, 2, 3, 1).reshape(-1, channel)
        delta_next = super().backward(delta_reshape)
        # 需要将输出再还原为原来的形状
        delta_next = delta_next.reshape(num_data, height, width, channel).transpose(0, 3, 1, 2)
        return delta_next


class ReLULayer(Layer):
    """ReLU激活层"""

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, input_):
        self.input = input_.copy()
        self.output = self.input * (self.input > 0)
        return self.output

    def backward(self, delta):
        return delta * (self.input > 0)


class SigmoidLayer(Layer):
    """Sigmoid激活层"""

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, input_):
        self.input = input_.copy()
        # 防止指数溢出
        indices_pos = np.nonzero(self.input >= 0)
        indices_neg = np.nonzero(self.input < 0)
        self.output = np.zeros_like(self.input)
        # y = 1 / (1 + exp(-x)), x >= 0
        # y = exp(x) / (1 + exp(x)), x < 0
        self.output[indices_pos] = 1 / (1 + np.exp(-self.input[indices_pos]))
        self.output[indices_neg] = np.exp(self.input[indices_neg]) / (1 + np.exp(self.input[indices_neg]))
        return self.output

    def backward(self, delta):
        return delta * (self.output * (1 - self.output))


class TanhLayer(Layer):
    """Tanh激活层"""

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, input_):
        self.input = input_.copy()
        # 防止指数溢出
        indices_pos = np.nonzero(self.input >= 0)
        indices_neg = np.nonzero(self.input < 0)
        self.output = np.zeros_like(self.input)
        # y = (1-exp(-2*x))/(1+exp(-2*x)), x >= 0
        # y = (exp(2*x)-1)/(1+exp(2*x)), x < 0
        self.output[indices_pos] = ((1 - np.exp(-2 * self.input[indices_pos]))
                                    / (1 + np.exp(-2 * self.input[indices_pos])))
        self.output[indices_neg] = ((np.exp(2 * self.input[indices_neg]) - 1)
                                    / (1 + np.exp(2 * self.input[indices_neg])))
        return self.output

    def backward(self, delta):
        return delta * (1 - self.output * self.output)


class SoftmaxLayer(Layer):
    """Softmax激活层"""

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, input_, dim=1):
        # 因为在求exp时，可能因为指数过大，出现溢出的情况
        # 而在softmax中，重要的是两个数字之间的差值，只要差值相同，softmax的结果就相同
        self.input = input_
        self.input -= np.max(self.input, axis=dim, keepdims=True)
        self.output = np.exp(self.input) / np.sum(np.exp(self.input), axis=dim, keepdims=True)
        return self.output

    def backward(self, delta):
        # Softmax的梯度反向传播集成在CrossEntropyWithSoftmax中了
        return delta
