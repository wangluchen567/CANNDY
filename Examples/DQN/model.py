from Core.Layer import Linear
from Core.Module import Module
from Core.Activation import ReLU


class Model(Module):
    def __init__(self, input_size, output_size, hidden_sizes, hidden_activation=ReLU, out_activation=None):
        super().__init__(input_size, output_size, hidden_sizes, hidden_activation, out_activation)
        self.num_hidden = len(self.hidden_sizes)
        self.num_layers = len(self.hidden_sizes) + 1
        # 创建网络层
        self.Layers = [Linear(self.input_size, self.hidden_sizes[0], self.hidden_activation)]
        if self.num_hidden > 1:
            self.Layers.extend(
                [Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1], self.hidden_activation) for i in
                 range(self.num_hidden - 1)])
        self.Layers.append(Linear(self.hidden_sizes[-1], self.output_size))
        # 计算参数数量
        self.num_params = self.get_num_params()

    def forward(self, input_):
        hidden = input_
        for fc in self.Layers:
            hidden = fc.forward(hidden)
        output = hidden
        return output

    def load_params_list(self, params_list):
        for i in range(self.num_layers):
            self.Layers[i].set_parameters(params_list[i])

    def get_params_list(self):
        params_list = []
        for i in range(self.num_layers):
            params_list.append(self.Layers[i].get_parameters())
        return params_list
