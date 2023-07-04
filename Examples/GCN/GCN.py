import numpy as np
from Core.Layer import GraphConv
from Core.Activation import ReLU


class GCN():
    def __init__(self, adj_mat, input_size, output_size, hidden_sizes, out_act=None):
        self.adj_mat = adj_mat
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = ReLU
        self.num_hidden = len(hidden_sizes)
        self.num_layers = len(hidden_sizes) + 1

        degree_mat = self.adj_mat.sum(axis=1)
        degree_pow_neg_half = np.diag(np.power(degree_mat, -0.5))
        self.adj_norm = degree_pow_neg_half @ self.adj_mat @ degree_pow_neg_half

        self.Layers = [GraphConv(input_size, hidden_sizes[0], self.adj_norm, self.activation)]
        if self.num_hidden > 1:
            self.Layers.extend(
                [GraphConv(hidden_sizes[i], hidden_sizes[i + 1], self.adj_norm, self.activation) for i in
                 range(self.num_hidden - 1)])
        # 最后层是否加激活函数
        if out_act is None:
            self.Layers.append(GraphConv(hidden_sizes[-1], output_size, self.adj_norm))
        else:
            self.Layers.append(GraphConv(hidden_sizes[-1], output_size, self.adj_norm, out_act))

    def forward(self, input):
        hidden = input.T
        for gc in self.Layers:
            hidden = gc.forward(hidden)
        output = hidden
        return output

