from Core.Layer import Linear
from Core.Module import Module
from Core.Activation import Tanh, Softmax


class Model(Module):
    def __init__(self, input_size, output_size, hidden_sizes, hidden_activation=Tanh, out_activation=Softmax):
        super().__init__(input_size, output_size, hidden_sizes, hidden_activation, out_activation)
        self.num_hidden = len(hidden_sizes)
        self.num_layers = len(hidden_sizes) + 1

        self.Layers = [Linear(self.input_size, self.hidden_sizes[0], self.hidden_activation)]
        if self.num_hidden > 1:
            self.Layers.extend(
                [Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1], self.hidden_activation) for i in
                 range(self.num_hidden - 1)])
        # 最后一层加激活函数
        self.Layers.append(Linear(self.hidden_sizes[-1], self.output_size, self.out_activation))
        # 计算参数数量
        self.num_params = self.get_num_params()

    def forward(self, input_):
        hidden = input_
        for fc in self.Layers:
            hidden = fc.forward(hidden)
        output = hidden
        return output

