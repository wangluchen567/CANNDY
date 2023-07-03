from Core.Layer import Linear
from Core.Activation import Sigmoid, ReLU


class MLP():
    def __init__(self, input_size, output_size, hidden_sizes, out_act=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = Sigmoid
        self.num_hidden = len(hidden_sizes)
        self.num_layers = len(hidden_sizes) + 1

        self.Layers = [Linear(input_size, hidden_sizes[0], self.activation)]
        if self.num_hidden > 1:
            self.Layers.extend(
                [Linear(hidden_sizes[i], hidden_sizes[i + 1], self.activation) for i in
                 range(self.num_hidden - 1)])
        # 最后层是否加激活函数
        if out_act is None:
            self.Layers.append(Linear(hidden_sizes[-1], output_size))
        else:
            self.Layers.append(Linear(hidden_sizes[-1], output_size, out_act))
        # 计算参数数量
        self.num_params = self.get_num_params()

    def forward(self, input):
        # 必须对输入进行转置
        # 考虑到实现层的反向传播
        hidden = input.T
        for fc in self.Layers:
            hidden = fc.forward(hidden)
        output = hidden
        return output

    def get_num_params(self):
        num_params = 0
        for fc in self.Layers:
            num_params += fc.num_param
        return num_params