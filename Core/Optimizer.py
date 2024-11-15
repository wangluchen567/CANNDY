import numpy as np


class Optimizer:
    """优化器父类"""

    def __init__(self, model, learning_rate, weight_decay):
        self.model = model
        assert learning_rate >= 0
        assert weight_decay >= 0
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # 获取整个模型中每层的对象(将嵌套展平)
        self.layer_list = self.get_grad_layers(self.model.Layers)
        # 获取模型的层数以方便优化
        self.num_layers = len(self.layer_list)
        # 记录优化步数
        self.steps = 0

    def get_grad_layers(self, layers):
        """获取整个模型中每层带梯度的对象(将嵌套展平)"""
        layer_list = []
        for layer in layers:
            if hasattr(layer, 'Layers') and layer.Layers is not None and len(layer.Layers):
                # 如果给定元素中有Layers变量且变量不为空则需要嵌套放置到list中
                layer_list.extend(self.get_grad_layers(layer.Layers))
            elif hasattr(layer, 'grad') and layer.grad is not None:
                # 否则检查该层是否有梯度并且梯度不为空，若存在则加入list
                layer_list.append(layer)
            else:
                # 否则不添加
                pass
        return layer_list

    def zero_grad(self):
        """所有网络层梯度置0"""
        for i in range(self.num_layers):
            self.layer_list[i].zero_grad()

    def step(self):
        """更新一次权重"""
        raise NotImplementedError

    def update(self, layer):
        """更新梯度更新速度"""
        # 先进行weight_decay
        layer.grad = layer.grad + layer.weight * self.weight_decay


class GradientDescent(Optimizer):
    def __init__(self, model, learning_rate=0.01, weight_decay=0):
        super(GradientDescent, self).__init__(model, learning_rate, weight_decay)
        # 记录梯度更新速度
        self.v = dict()
        for i in range(self.num_layers):
            self.zero_v(self.layer_list[i])

    def step(self):
        """每层网络更新一次权重"""
        self.steps += 1  # 更新步数
        for i in range(self.num_layers):
            # 先更新梯度更新速度v
            self.update_v(self.layer_list[i])
            # 再更新权重
            self.update(self.layer_list[i])

    def zero_v(self, layer):
        """梯度更新速度置零"""
        self.v[layer] = 0

    def update_v(self, layer):
        """更新梯度更新速度"""
        self.v[layer] = self.learning_rate * layer.grad

    def update(self, layer):
        """更新权重"""
        super().update(layer)
        next_weight = layer.weight - self.v[layer]
        layer.set_parameters(next_weight)


class Momentum(Optimizer):
    def __init__(self, model, learning_rate=0.01, momentum=0.9, weight_decay=0):
        super(Momentum, self).__init__(model, learning_rate, weight_decay)
        self.momentum = momentum
        # 记录梯度更新速度
        self.v = dict()
        for i in range(self.num_layers):
            self.zero_v(self.layer_list[i])

    def step(self):
        """每层网络更新一次权重"""
        self.steps += 1  # 更新步数
        for i in range(self.num_layers):
            # 先更新梯度更新速度v
            self.update_v(self.layer_list[i])
            # 再更新权重
            self.update(self.layer_list[i])

    def zero_v(self, layer):
        """梯度更新速度置零"""
        self.v[layer] = 0

    def update_v(self, layer):
        """更新梯度更新速度"""
        self.v[layer] = self.momentum * self.v[layer] + self.learning_rate * layer.grad

    def update(self, layer):
        """更新权重"""
        super().update(layer)
        next_weight = layer.weight - self.v[layer]
        layer.set_parameters(next_weight)


class AdaGrad(Optimizer):
    def __init__(self, model, learning_rate=0.01, weight_decay=0):
        super(AdaGrad, self).__init__(model, learning_rate, weight_decay)
        # 记录梯度各分量的平方
        self.s = dict()
        for i in range(self.num_layers):
            self.zero_s(self.layer_list[i])

    def step(self):
        """每层网络更新一次权重"""
        self.steps += 1  # 更新步数
        for i in range(self.num_layers):
            # 先更新梯度各分量的平方s
            self.update_s(self.layer_list[i])
            # 再更新权重
            self.update(self.layer_list[i])

    def zero_s(self, layer):
        """梯度各分量平方速度置零"""
        self.s[layer] = 0

    def update_s(self, layer):
        """更新梯度各分量平方更新速度"""
        self.s[layer] = self.s[layer] + layer.grad * layer.grad

    def update(self, layer):
        """更新权重"""
        super().update(layer)
        next_weight = layer.weight - self.learning_rate * layer.grad / np.sqrt(self.s[layer] + 1e-8)
        layer.set_parameters(next_weight)


class RMSProp(Optimizer):
    def __init__(self, model, learning_rate=0.01, beta=0.99, weight_decay=0):
        super(RMSProp, self).__init__(model, learning_rate, weight_decay)
        # 衰减系数
        assert 0.0 < beta < 1.0
        self.beta = beta
        # 记录梯度各分量的平方
        self.s = dict()
        for i in range(self.num_layers):
            self.zero_s(self.layer_list[i])

    def step(self):
        """每层网络更新一次权重"""
        self.steps += 1  # 更新步数
        for i in range(self.num_layers):
            # 先更新梯度各分量的平方s
            self.update_s(self.layer_list[i])
            # 再更新权重
            self.update(self.layer_list[i])

    def zero_s(self, layer):
        """梯度各分量平方速度置零"""
        self.s[layer] = 0

    def update_s(self, layer):
        """更新梯度各分量平方更新速度"""
        self.s[layer] = self.beta * self.s[layer] + (1 - self.beta) * layer.grad * layer.grad

    def update(self, layer):
        """更新权重"""
        super().update(layer)
        next_weight = layer.weight - self.learning_rate * layer.grad / np.sqrt(self.s[layer] + 1e-8)
        layer.set_parameters(next_weight)


class Adam(Optimizer):
    def __init__(self, model, learning_rate=0.01, beta_1=0.9, beta_2=0.999, weight_decay=0, ams_grad=False, eps=1e-8):
        super(Adam, self).__init__(model, learning_rate, weight_decay)
        # 历史梯度衰减系数
        assert 0.0 < beta_1 < 1.0
        self.beta_1 = beta_1
        # 历史梯度各分量平方衰减系数
        assert 0.0 < beta_2 < 1.0
        self.beta_2 = beta_2
        assert 0.0 < eps < 1.0
        self.eps = eps
        # 是否使用ams_grad
        self.ams_grad = ams_grad
        # 初始化一阶矩估计
        self.v = dict()
        for i in range(self.num_layers):
            self.zero_v(self.layer_list[i])
        # 初始化二阶矩估计
        self.s = dict()
        for i in range(self.num_layers):
            self.zero_s(self.layer_list[i])
        # 初始化二阶矩估计历史最大值（AMS_Grad）
        if self.ams_grad:
            self.max_s = dict()
            for i in range(self.num_layers):
                self.zero_max_s(self.layer_list[i])

    def step(self):
        """每层网络更新一次权重"""
        self.steps += 1  # 更新步数
        for i in range(self.num_layers):
            # 先更新一阶矩估计
            self.update_v(self.layer_list[i])
            # 再更新二阶矩估计
            self.update_s(self.layer_list[i])
            # 再更新权重
            self.update(self.layer_list[i])

    def zero_v(self, layer):
        """一阶矩估计置零"""
        self.v[layer] = np.zeros_like(layer.grad)

    def zero_s(self, layer):
        """二阶矩估计置零"""
        self.s[layer] = np.zeros_like(layer.grad)

    def zero_max_s(self, layer):
        """最大二阶矩估计置零"""
        self.max_s[layer] = np.zeros_like(layer.grad)

    def update_v(self, layer):
        """更新一阶矩估计"""
        self.v[layer] = self.beta_1 * self.v[layer] + (1 - self.beta_1) * layer.grad

    def update_s(self, layer):
        """更新二阶矩估计"""
        self.s[layer] = self.beta_2 * self.s[layer] + (1 - self.beta_2) * layer.grad * layer.grad
        if self.ams_grad:
            # 更新最大二阶矩估计
            self.max_s[layer] = np.maximum(self.max_s[layer], self.s[layer])

    def update(self, layer):
        """更新权重"""
        super().update(layer)
        # 进行偏差校正
        v_layer = self.v[layer] / (1 - self.beta_1 ** self.steps)
        if self.ams_grad:
            s_layer = self.max_s[layer] / (1 - self.beta_2 ** self.steps)
        else:
            s_layer = self.s[layer] / (1 - self.beta_2 ** self.steps)
        # 得到更新后的权重
        next_weight = layer.weight - self.learning_rate * v_layer / np.sqrt(s_layer + self.eps)
        layer.set_parameters(next_weight)
