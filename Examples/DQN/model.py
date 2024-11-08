from Core.Layer import Linear
from Core.Activation import ReLU


class Model():
    def __init__(self, input_size, output_size, hidden_sizes):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = ReLU
        self.num_hidden = len(hidden_sizes)
        self.num_layers = len(hidden_sizes) + 1

        self.Layers = [Linear(input_size, hidden_sizes[0], self.activation)]
        if self.num_hidden > 1:
            self.Layers.extend(
                [Linear(hidden_sizes[i], hidden_sizes[i + 1], self.activation) for i in
                 range(self.num_hidden - 1)])
        # 最后一层加激活函数
        self.Layers.append(Linear(hidden_sizes[-1], output_size))
        # 计算参数数量
        self.num_params = self.get_num_params()

    def forward(self, input):
        hidden = input
        for fc in self.Layers:
            hidden = fc.forward(hidden)
        output = hidden
        return output

    def get_num_params(self):
        num_params = 0
        for fc in self.Layers:
            num_params += fc.num_param
        return num_params

    def load_params_list(self, params_list):
        for i in range(self.num_layers):
            self.Layers[i].set_parameters(params_list[i])

    def get_params_list(self):
        params_list = []
        for i in range(self.num_layers):
            params_list.append(self.Layers[i].get_parameters())
        return params_list
