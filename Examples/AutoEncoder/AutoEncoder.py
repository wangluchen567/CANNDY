from Core.Layer import Linear
from Core.Activation import Sigmoid, ReLU


class AutoEncoder():
    def __init__(self):
        self.Encoder = [
            Linear(784, 256, ReLU),
            Linear(256, 64, ReLU),
            Linear(64, 20, ReLU)
        ]
        self.Decoder = [
            Linear(20, 64, ReLU),
            Linear(64, 256, ReLU),
            Linear(256, 784, Sigmoid)
        ]
        self.Layers = self.Encoder
        self.Layers.extend(self.Decoder)
        self.num_layers = len(self.Layers)
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
