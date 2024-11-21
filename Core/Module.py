import numpy as np
from Core.Activation import Sigmoid, ReLU, Tanh, Softmax
from Core.Layer import Linear, GCNConv, Dropout, RNN, Conv2d, MaxPool2d, Flatten, ReLULayer, BatchNorm2d


class Module:
    def __init__(self, input_size=None, output_size=None, hidden_sizes=None,
                 hidden_activation=None, out_activation=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.Layers = None
        self.num_params = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        """设置为训练模式"""
        for layer in self.Layers:
            layer.training = True

    def eval(self):
        """设置为验证模式"""
        for layer in self.Layers:
            layer.training = False

    def get_num_params(self):
        num_params = 0
        for layer in self.Layers:
            num_params += layer.num_params
        return num_params

    def get_parameters(self):
        """获取模型的参数字典"""
        depth = 0  # 网络深度
        params_dict = dict()
        for layer in self.Layers:
            if hasattr(layer, 'weight') and layer.weight is not None:
                key = str(depth) + '_' + layer.__class__.__name__
                params_dict[key] = layer.get_parameters()
                depth += 1
        return params_dict

    def set_parameters(self, params_dict):
        """根据给定参数字典设置模型的参数"""
        depth = 0  # 网络深度
        for layer in self.Layers:
            if hasattr(layer, 'weight') and layer.weight is not None:
                key = str(depth) + '_' + layer.__class__.__name__
                layer.set_parameters(params_dict[key])
                depth += 1


class MLP(Module):
    """全连接模型"""

    def __init__(self, input_size, output_size, hidden_sizes, hidden_activation=Sigmoid, out_activation=None):
        """
        全连接模型
        :param input_size: 输入层大小
        :param output_size: 输出层大小
        :param hidden_sizes: 隐藏层大小(多个)
        :param hidden_activation: 隐藏层激活函数
        :param out_activation: 输出层激活函数
        """
        super().__init__(input_size, output_size, hidden_sizes, hidden_activation, out_activation)
        self.num_hidden = len(self.hidden_sizes)
        # 初始化第一层
        self.Layers = [Linear(self.input_size, self.hidden_sizes[0], self.hidden_activation)]
        # 加入中间隐层
        if self.num_hidden > 1:
            self.Layers.extend(
                [Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1], self.hidden_activation)
                 for i in range(self.num_hidden - 1)])
        # 加入最后一层
        self.Layers.append(Linear(self.hidden_sizes[-1], self.output_size, self.out_activation))
        # 计算参数数量
        self.num_params = self.get_num_params()

    def forward(self, input_):
        hidden = input_.copy()
        for fc in self.Layers:
            hidden = fc(hidden)
        output = hidden
        return output


class GCN(Module):
    """图卷积模型"""

    def __init__(self, adj_mat, input_size, output_size, hidden_sizes,
                 hidden_activation=ReLU, out_activation=None, dropout=False):
        """
        图卷积模型
        :param adj_mat: 图的邻接矩阵
        :param input_size: 输入层大小
        :param output_size: 输出层大小
        :param hidden_sizes: 隐藏层大小(多个)
        :param hidden_activation: 隐藏层激活函数
        :param out_activation: 输出层激活函数
        """
        super().__init__(input_size, output_size, hidden_sizes, hidden_activation, out_activation)
        self.adj_mat = adj_mat
        self.dropout = dropout
        self.num_hidden = len(self.hidden_sizes)
        # 计算度矩阵
        degree_mat = self.adj_mat.sum(axis=1)
        degree_pow_neg_half = np.diag(np.power(degree_mat, -0.5))
        self.adj_norm = degree_pow_neg_half @ self.adj_mat @ degree_pow_neg_half
        # 初始化第一层
        self.Layers = [GCNConv(self.input_size, self.hidden_sizes[0], self.adj_norm, self.hidden_activation),
                       Dropout(p=self.dropout)]  # 加入dropout层
        # 加入中间隐层
        if self.num_hidden > 1:
            for i in range(self.num_hidden - 1):
                self.Layers.append(GCNConv(self.hidden_sizes[i], self.hidden_sizes[i + 1],
                                           self.adj_norm, self.hidden_activation))
                self.Layers.append(Dropout(p=self.dropout))  # 加入dropout层
        # 加入最后一层
        self.Layers.append(GCNConv(self.hidden_sizes[-1], self.output_size, self.adj_norm, self.out_activation))
        # 计算参数数量
        self.num_params = self.get_num_params()
        self.num_layers = len(self.hidden_sizes) + 1

    def forward(self, input_):
        hidden = input_
        for gc in self.Layers:
            hidden = gc(hidden)
        output = hidden
        return output


class RNNModel(Module):
    """循环神经网络模型"""

    def __init__(self, input_size, rnn_hidden_size, num_layers, linear_hidden_sizes, output_size,
                 rnn_activation=Tanh, bias=True, batch_first=False, hidden_activation=ReLU, out_activation=None):

        super().__init__(input_size, output_size, None, hidden_activation, out_activation)
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.rnn_activation = rnn_activation
        self.linear_hidden_sizes = linear_hidden_sizes
        self.bias = bias
        # 根据参数构造RNN网络
        self.Rnn = RNN(self.input_size, self.rnn_hidden_size, self.num_layers,
                       self.rnn_activation, self.bias, self.batch_first)
        # 这里必须将Layers初始化为RNN的Layer，以便梯度下降
        self.Layers = [self.Rnn]
        # 初始化线性层(至少有一层)
        self.Linear_Layers = []
        if self.linear_hidden_sizes is None or len(self.linear_hidden_sizes) == 0:
            self.Linear_Layers.append(Linear(self.rnn_hidden_size, self.output_size, self.out_activation))
        else:
            # 要添加所有隐藏层
            self.Linear_Layers.append(Linear(self.rnn_hidden_size, self.linear_hidden_sizes[0], self.hidden_activation))
            for i in range(len(self.linear_hidden_sizes) - 1):
                self.Linear_Layers.append(Linear(self.linear_hidden_sizes[i],
                                                 self.linear_hidden_sizes[i + 1],
                                                 self.hidden_activation))
            # 然后添加最终输出层
            self.Linear_Layers.append(Linear(self.linear_hidden_sizes[-1], self.output_size, self.out_activation))
        # 然后将线性层接入到RNN网络之后
        self.Layers.extend(self.Linear_Layers)

    def forward(self, inputs, state=None):
        """前向传播"""
        # 先通过RNN网络
        outputs, state = self.Rnn(inputs, state)
        # 判断是否是batch_first,若不是则需修改形状
        if not self.batch_first:
            outputs = outputs.transpose(1, 0, 2)
        # 取最后一个元素作为线性层的输入
        output = outputs[-1, :, :]
        for fc in self.Linear_Layers:
            output = fc(output)
        return output


class LeNet5(Module):
    def __init__(self):
        super().__init__()
        self.Layers = [
            Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(num_features=6),
            ReLULayer(),
            MaxPool2d(kernel_size=2, stride=2, padding=0),
            Dropout(p=0.2),
            Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            BatchNorm2d(num_features=16),
            ReLULayer(),
            MaxPool2d(kernel_size=2, stride=2, padding=0),
            Dropout(p=0.2),
            Flatten(),
            Linear(input_size=400, output_size=120, activation=ReLU),
            Linear(input_size=120, output_size=84, activation=ReLU),
            Linear(input_size=84, output_size=10, activation=Softmax),
        ]

    def forward(self, input_):
        hidden = input_.copy()
        for layer in self.Layers:
            hidden = layer(hidden)
        output = hidden
        return output
