from Core.Layer import Linear
from Core.Module import Module
from Core.Activation import Sigmoid, ReLU


class AutoEncoder(Module):
    def __init__(self):
        super().__init__()
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

    def forward(self, input_):
        hidden = input_
        for fc in self.Layers:
            hidden = fc(hidden)
        output = hidden
        return output
