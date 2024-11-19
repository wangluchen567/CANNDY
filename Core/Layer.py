import numpy as np


class Layer:
    """层级父类"""

    def __init__(self, input_size=None, output_size=None, activation=None, bias=False):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.bias = bias
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
    def padding_matrix(matrix: np.ndarray, padding):
        """对矩阵进行padding填充"""
        if isinstance(padding, int):  # 整数填充
            # 在最后两个维度四周均匀填充padding个0
            return np.pad(matrix, ((0, 0), (0, 0), (padding, padding),
                                   (padding, padding)), mode='constant')
        elif (isinstance(padding, tuple) or isinstance(padding, list)) and len(padding) == 2:  # 元组填充
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
        return self.weight

    def set_parameters(self, weight):
        assert self.weight.shape == weight.shape
        self.weight = weight

    def forward(self, input_):
        """前向传播"""
        # 计算batch大小
        self.batch_size = input_.shape[0]
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
            delta = grad * self.activation.backward(self.output.T)
        # 计算梯度(累计梯度) 取平均
        self.grad += (self.input_1 @ delta).T / self.batch_size
        # 将delta @ w传递到上一层网络
        if self.bias:
            # 偏置求导被消掉了无需参与反向传播
            delta_next = delta @ self.weight[:, :-1]
        else:
            delta_next = delta @ self.weight
        return delta_next


class Dropout(Layer):
    """随机失活层"""

    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.p = p  # 随机失活概率
        self.mask = None  # 记录被dropout的部分

    def forward(self, input_):
        """前向传播"""
        output = input_.copy()
        # 若失活概率非0且是训练模式则进行mask
        if self.p > 0 and self.training:
            self.mask = np.random.uniform(0, 1, size=output.shape) > self.p
            # 进行dropout操作
            output = output * self.mask / (1 - self.p)
        return output

    def backward(self, grad):
        """反向传播"""
        # 若失活概率非0且是训练模式则需要对梯度进行mask
        if self.p > 0 and self.training:
            return grad * self.mask / (1 - self.p)
        else:
            return grad


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
        return self.weight

    def set_parameters(self, weight):
        """设置参数值（权重值）"""
        assert self.weight.shape == weight.shape
        self.weight = weight
        self.weight_input = self.weight[:, :self.splice]
        self.weight_hidden = self.weight[:, self.splice:]

    def forward(self, input_, hidden):
        """前向传播"""
        # 计算batch大小
        self.batch_size = input_.shape[0]
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
            delta = grad * self.activation.backward(output.T)
        # 计算各个权重的梯度并取平均
        grad_input = (input_1 @ delta).T / self.batch_size
        grad_hidden = (hidden_1 @ delta).T / self.batch_size
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


class Conv2d(Layer):
    """卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None, bias=False):
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
        self.kh, self.kw = self.init_param_value(self.kernel_size)
        self.sh, self.sw = self.init_param_value(self.stride)
        # 初始化权重
        self.weight = np.zeros((self.co, self.ci + self.bias, self.kh, self.kw))
        # 何凯明的方法初始化权重
        self.weight = self.kaiming_uniform_(self.weight, mode='fan_out', gain=self.get_gain(), bias=bias)
        # 实例化激活函数
        if self.activation is not None:
            self.activation = self.activation()
        # 初始化梯度
        self.grad = np.zeros_like(self.weight)
        # 记录输出对权重和输出对输入的导数
        self.out_diff_weight, self.out_diff_input = None, None
        # 计算参数数量
        self.num_params = self.weight.size

    @staticmethod
    def init_param_value(param):
        """初始化给定参数值"""
        if isinstance(param, int):
            return param, param
        elif isinstance(param, tuple) or isinstance(param, list):
            if len(param) != 2:
                raise ValueError("param must be 2-dim tuples or lists")
            return param
        else:
            raise ValueError("param must be int or 2-dim tuples or lists")

    def zero_grad(self):
        self.grad = np.zeros_like(self.weight)

    def get_parameters(self):
        return self.weight

    def set_parameters(self, weight):
        assert self.weight.shape == weight.shape
        self.weight = weight

    def forward(self, input_):
        """前向传播"""
        # input_size: (num_data/batch_size, in_channels, height, width) (NCHW格式)
        if input_.ndim != 4:
            raise ValueError("Conv2d can only handle 2-dim data")
        input_1 = input_.copy()
        # 根据给定的padding得到加入padding后的输入
        self.input = self.padding_matrix(input_1, self.padding)
        # 调整形状为(height, width, num_data/batch_size, in_channels) (HWNC格式)
        # numpy多维数组是行优先，尽量使低维度的数据在运算时保持连续，就能减少cache不命中的次数，从而提高性能
        input_pad = self.input.transpose(2, 3, 0, 1)
        weight = self.weight.transpose(2, 3, 0, 1)
        # 得到输入的形状
        ih, iw, nd, _ = input_pad.shape
        self.batch_size = nd  # batch_size === num_data
        # 如果使用偏置则需要加入对应偏置的弥散矩阵
        if self.bias:
            bias_diffusion = np.ones((ih, iw, nd, 1))
            input_pad = np.concatenate((input_pad, bias_diffusion), axis=-1)
        # 下面正式计算卷积及其导数值
        # 重新计算形状(可能加入了偏置)
        ih, iw, nd, ci = input_pad.shape
        kh, kw, sh, sw, co = self.kh, self.kw, self.sh, self.sw, self.co
        # 计算输出特征图形状
        oh = int(np.floor((ih - kh + sh) / sh))
        ow = int(np.floor((iw - kw + sw) / sw))
        # 初始化输出的特征图，与输入形状相同(HWNC格式)
        output = np.zeros((oh, ow, nd, co))
        # 初始化input和weight的导数(六维矩阵)
        # 输出对权重的导数形状为: (oh, ow, kh, kw, nd, ci)
        self.out_diff_weight = np.zeros((oh, ow, kh, kw, nd, ci))
        # 输出对输入的导数形状为: (oh, ow, ih, iw, co, ci)
        self.out_diff_input = np.zeros((oh, ow, ih, iw, co, ci))
        # 进行卷积操作，同时记录导数情况
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (kh, kw, nd, ci)
                input_part = input_pad[i:i + kh, j:j + kw, :, :]
                # (nd, co) <= (kh, kw, ci, nd) dot (kh, kw, ci, co)
                output[h, w, :, :] = np.tensordot(input_part, weight, axes=([0, 1, 3], [0, 1, 3]))
                # 记录导数值
                self.out_diff_weight[h, w, :, :, :, :] = input_part
                self.out_diff_input[h, w, i:i + kh, j:j + kw, :, :] = weight
        # 然后将轴转换回(num_data/batch_size, out_channels, height, width)(NCHW格式)
        self.output = output.transpose(2, 3, 0, 1)
        return self.output

    def backward(self, grad):
        """反向传播"""
        delta = grad
        if self.activation is not None:
            delta = grad * self.activation.backward(self.output)
        # 改变形状为(oh, ow, nd, co)(HWNC格式)
        delta = delta.transpose(2, 3, 0, 1)
        # 根据之前记录的导数值计算梯度
        # 改变形状方便求点积
        # (oh, ow, nd, ci, kh, kw) <= (oh, ow, kh, kw, nd, ci)
        self.out_diff_weight = self.out_diff_weight.transpose(0, 1, 4, 5, 2, 3)
        # (co, ci, kh, kw) <= (oh, ow, nd, co) dot (oh, ow, nd, ci, kh, kw)
        self.grad = np.tensordot(delta, self.out_diff_weight, axes=([0, 1, 2], [0, 1, 2]))
        # 传递到上一层网络时偏置不传导梯度
        if self.bias:
            # 形状: (oh, ow, ih, iw, co, ci)
            self.out_diff_input = self.out_diff_input[:, :, :, :, :, :-1]
        # 将梯度传递到上一层网络
        # 改变形状方便求点积
        # (oh, ow, co, nd) <= (oh, ow, nd, co)
        delta = delta.transpose(0, 1, 3, 2)
        # (oh, ow, co, ci, ih, iw) <= (oh, ow, ih, iw, co, ci)
        self.out_diff_input = self.out_diff_input.transpose(0, 1, 4, 5, 2, 3)
        # (nd, ci, ih, iw) <= (oh, ow, co, nd) dot (oh, ow, co, ci, ih, iw)
        delta_next = np.tensordot(delta, self.out_diff_input, axes=([0, 1, 2], [0, 1, 2]))
        # 得到反向传播到上一层的形状(输入的形状)
        _, _, ih, iw = delta_next.shape
        # 还要考虑padding的梯度，由于是常数填充，所以直接裁剪掉padding
        delta_next = delta_next[:, :, self.padding:ih - self.padding, self.padding:iw - self.padding]
        return delta_next


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

    def backward(self, grad):
        """反向传播"""
        # 将梯度变为输入的形状反向传播到上一层
        return grad.reshape(self.input_shape)


class MaxPool2d(Layer):
    """最大池化层"""

    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 记录输入输出以及batch大小以方便反向传播
        self.input, self.output, self.batch_size = None, None, 1
        # 根据给定参数初始化相关参数
        self.kh, self.kw = self.init_param_value(self.kernel_size)
        self.sh, self.sw = self.init_param_value(self.stride)
        self.max_index = None  # 记录最大池化时的位置

    @staticmethod
    def init_param_value(param):
        """初始化给定参数值"""
        if isinstance(param, int):
            return param, param
        elif isinstance(param, tuple) or isinstance(param, list):
            if len(param) != 2:
                raise ValueError("param must be 2-dim tuples or lists")
            return param
        else:
            raise ValueError("param must be int or 2-dim tuples or lists")

    def forward(self, input_):
        """前向传播"""
        # input_size: (num_data/batch_size, in_channels, height, width) (NCHW格式)
        if input_.ndim != 4:
            raise ValueError("Conv2d can only handle 2-dim data")
        input_1 = input_.copy()
        # 根据给定的padding得到加入padding后的输入
        self.input = self.padding_matrix(input_1, self.padding)
        # 调整形状为(height, width, num_data/batch_size, in_channels) (HWNC格式)
        # numpy多维数组是行优先，尽量使低维度的数据在运算时保持连续，就能减少cache不命中的次数，从而提高性能
        input_pad = self.input.transpose(2, 3, 0, 1)
        # 得到输入的形状
        ih, iw, nd, _ = input_pad.shape
        self.batch_size = nd  # batch_size === num_data
        # 下面正式计算最大池化
        # 重新计算形状(可能加入了偏置)
        ih, iw, nd, ci = input_pad.shape
        kh, kw, sh, sw, co = self.kh, self.kw, self.sh, self.sw, ci
        # 计算输出特征图形状
        oh = int(np.floor((ih - kh + sh) / sh))
        ow = int(np.floor((iw - kw + sw) / sw))
        # 初始化输出的特征图，与输入形状相同(HWNC格式)
        output = np.zeros((oh, ow, nd, co))
        # 初始化梯度mask下标
        self.max_index = np.zeros((oh, ow, nd * ci, 2), dtype=int)
        # 进行卷积操作，同时记录导数情况
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (kh, kw, nd, ci)
                input_part = input_pad[i:i + kh, j:j + kw, :, :]
                # 将后两维合并以方便操作 (kh, kw, nd*ci)
                input_part_ = input_part.reshape(kh, kw, -1)
                # 进行最大池化
                part_max = input_part_.max(axis=(0, 1))
                # 得到该区域最大池化的结果
                output[h, w, :, :] = part_max.reshape(nd, ci)
                # 得到该区域所有最大值的位置
                part_max_pos = np.where(input_part_ == part_max)
                # 得到该区域第一个最大值的下标的位置
                _, unique_index = np.unique(part_max_pos[-1], return_index=True)
                # 得到该区域所有最大值的下标
                part_max_index = np.vstack(part_max_pos).T[unique_index, :-1]
                # 保存这些最大值下标
                self.max_index[h][w] = part_max_index
        # 然后将轴转换回(num_data/batch_size, out_channels, height, width)(NCHW格式)
        self.output = output.transpose(2, 3, 0, 1)
        return self.output

    def backward(self, grad):
        """反向传播"""
        # 调整形状为(height, width, num_data/batch_size, in_channels) (HWNC格式)
        delta = grad.transpose(2, 3, 0, 1).copy()
        # 获取梯度的形状(与输出的形状相同)
        oh, ow, nd, co = delta.shape
        # 初始化反向传播到上一层的梯度，并转换为HWNC格式
        delta_next = np.zeros_like(self.input).transpose(2, 3, 0, 1)
        # 核函数大小和步长大小
        kh, kw, sh, sw = self.kh, self.kw, self.sh, self.sw
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (kh, kw, nd, ci)
                delta_next_part = delta_next[i:i + kh, j:j + kw, :, :]
                # 将后两维合并并调整轴以方便操作 (kh, kw, nd*ci)
                delta_next_part = delta_next_part.reshape(kh, kw, -1).transpose(2, 0, 1)
                # 获取该部分的最大值下标
                part_max_index = self.max_index[h][w]
                # 将前面的梯度利用mask传递到上一层
                delta_next_part[np.arange(len(part_max_index)), part_max_index[:, 0], part_max_index[:, 1]] \
                    += delta[h][w].flatten()
        # 将传播到上一层的梯度转换为NCHW格式
        delta_next = delta_next.transpose(2, 3, 0, 1)
        # 得到反向传播到上一层的形状(输入的形状)
        _, _, ih, iw = delta_next.shape
        # 还要考虑padding的梯度，由于是常数填充，所以直接裁剪掉padding
        delta_next = delta_next[:, :, self.padding:ih - self.padding, self.padding:iw - self.padding]
        return delta_next


class MeanPool2d(Layer):
    """平均池化层"""

    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 记录输入输出以及batch大小以方便反向传播
        self.input, self.output, self.batch_size = None, None, 1
        # 根据给定参数初始化相关参数
        self.kh, self.kw = self.init_param_value(self.kernel_size)
        self.sh, self.sw = self.init_param_value(self.stride)

    @staticmethod
    def init_param_value(param):
        """初始化给定参数值"""
        if isinstance(param, int):
            return param, param
        elif isinstance(param, tuple) or isinstance(param, list):
            if len(param) != 2:
                raise ValueError("param must be 2-dim tuples or lists")
            return param
        else:
            raise ValueError("param must be int or 2-dim tuples or lists")

    def forward(self, input_):
        """前向传播"""
        # input_size: (num_data/batch_size, in_channels, height, width) (NCHW格式)
        if input_.ndim != 4:
            raise ValueError("Conv2d can only handle 2-dim data")
        input_1 = input_.copy()
        # 根据给定的padding得到加入padding后的输入
        self.input = self.padding_matrix(input_1, self.padding)
        # 调整形状为(height, width, num_data/batch_size, in_channels) (HWNC格式)
        # numpy多维数组是行优先，尽量使低维度的数据在运算时保持连续，就能减少cache不命中的次数，从而提高性能
        input_pad = self.input.transpose(2, 3, 0, 1)
        # 得到输入的形状
        ih, iw, nd, _ = input_pad.shape
        self.batch_size = nd  # batch_size === num_data
        # 下面正式计算最大池化
        # 重新计算形状(可能加入了偏置)
        ih, iw, nd, ci = input_pad.shape
        kh, kw, sh, sw, co = self.kh, self.kw, self.sh, self.sw, ci
        # 计算输出特征图形状
        oh = int(np.floor((ih - kh + sh) / sh))
        ow = int(np.floor((iw - kw + sw) / sw))
        # 初始化输出的特征图，与输入形状相同(HWNC格式)
        output = np.zeros((oh, ow, nd, co))
        # 进行卷积操作，同时记录导数情况
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (kh, kw, nd, ci)
                input_part = input_pad[i:i + kh, j:j + kw, :, :]
                # 调整形状以便操作
                input_part = input_part.transpose(2, 3, 0, 1)
                # 取这部分的平均值作为平均池化的输出结果
                output[h, w, :, :] = input_part.mean(axis=(2, 3))
        # 然后将轴转换回(num_data/batch_size, out_channels, height, width)(NCHW格式)
        self.output = output.transpose(2, 3, 0, 1)
        return self.output

    def backward(self, grad):
        """反向传播"""
        # 调整形状为(height, width, num_data/batch_size, in_channels) (HWNC格式)
        delta = grad.transpose(2, 3, 0, 1).copy()
        # 获取梯度的形状(与输出的形状相同)
        oh, ow, nd, co = delta.shape
        # 初始化反向传播到上一层的梯度，并转换为HWNC格式
        delta_next = np.zeros_like(self.input).transpose(2, 3, 0, 1)
        # 核函数大小和步长大小
        kh, kw, sh, sw = self.kh, self.kw, self.sh, self.sw
        for h in range(oh):
            i = h * sh
            for w in range(ow):
                j = w * sw
                # 形状: (kh, kw, nd, ci)
                delta_next_part = delta_next[i:i + kh, j:j + kw, :, :]
                # 调整形状以便操作
                delta_next_part = delta_next_part.transpose(2, 3, 0, 1).reshape(nd, -1, (kh * kw))
                # 计算对应元素的平均梯度值
                delta_mean = delta[h][w] / (kh * kw)
                # 将前面的梯度传递到上一层
                delta_next_part += np.repeat(delta_mean[:, :, np.newaxis], (kh * kw), axis=-1)
        # 将传播到上一层的梯度转换为NCHW格式
        delta_next = delta_next.transpose(2, 3, 0, 1)
        # 得到反向传播到上一层的形状(输入的形状)
        _, _, ih, iw = delta_next.shape
        # 还要考虑padding的梯度，由于是常数填充，所以直接裁剪掉padding
        delta_next = delta_next[:, :, self.padding:ih - self.padding, self.padding:iw - self.padding]
        return delta_next


class BatchNorm2d(Layer):
    """批归一化层"""

    def __init__(self, eps=1.e-8):
        super().__init__()
        self.eps = eps
        self.gamma, self.beta = None, None
        # 需要记录全局的均值与方差
        self.mean, self.var = 0, 0
        self.weight = 0

        self.gard = None

    def zero_grad(self):
        """梯度置为0矩阵"""
        pass

    def get_parameters(self):
        """获取该层的权重参数"""
        pass

    def set_parameters(self, *args, **kwargs):
        """设置该层的权重参数"""
        pass

    def forward(self, *args, **kwargs):
        """该层前向传播"""
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        """该层反向传播"""
        raise NotImplementedError


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

    def backward(self, grad):
        return grad * (self.input > 0)


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

    def backward(self, grad):
        return grad * (self.output * (1 - self.output))


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

    def backward(self, grad):
        return grad * (1 - self.output * self.output)


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

    def backward(self, grad):
        # Softmax的梯度反向传播集成在CrossEntropyWithSoftmax中了
        return grad
