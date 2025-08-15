## 🚀 快速开始

### 模型构建

以一个简单的二分类任务为例，假设我们需要构建一个模型用于对同心圆数据进行分类。我们可以使用全连接神经网络（多层感知机）来实现。以下是一个输入维度为2、输出维度为2、隐藏层维度为3的模型构建示例。

```python
from Core.Module import Module, Linear  # 导入Module和Linear类
from Core.Activation import Sigmoid, Softmax  # 导入激活函数

class MLPDemo(Module):
    def __init__(self):
        # 调用父类Module的构造方法来初始化父类的属性
        # 子类会继承父类的属性和方法
        super().__init__()
        # 构建一个简单的全连接神经网络
        self.Layers = [
            Linear(2, 3, Sigmoid),
            Linear(3, 2, Softmax)
        ]
    
    def forward(self, input_):
        # 覆写前向传播函数
        hidden = input_.copy()
        for layer in self.Layers:
            hidden = layer(hidden)
        output = hidden
        return output
```
**注意**：为了使模型能够进行反向传播训练，模型必须继承 [Core.Module](../Core/Module.py) 下的 `Module` 父类，并且需要将需要训练的网络部分放到 `Layers` 中。`Module` 父类会通过 `backward` 函数按照 `Layers` 中的层级架构逐步反向传播梯度。此外，必须覆写 `forward` 函数。

可以看到，模型的构建方式与 PyTorch 类似，但本项目通过调用 [Core.Layer](../Core/Layer.py) 下的 `Linear` 来实现线性层，并且额外支持为每个线性层指定激活函数。

如果想学习构建更通用的全连接神经网络模型，可以参考 [Core.Module](../Core/Module.py) 下的 `MLP` 的实现。

### 模型训练

模型构建完成后，可以使用梯度优化器对模型进行反向传播和梯度下降训练。首先，需要实现一个训练一轮（epoch）的函数：

```python
import numpy as np
from Core.Loss import CrossEntropyWithSoftmax  # 导入损失函数

def train_epoch(model, optimizer, X, Y, batch_size):
    """训练一轮(训练一个epoch)"""
    train_loss = 0  # 训练损失初始化
    for i in np.arange(0, len(X), batch_size):
        # 提取一个batch的输入数据和真实数据
        input_ = X[i:i + batch_size, :]
        truth = Y[i:i + batch_size, :]
        # 得到模型的预测输出
        output = model.forward(input_)
        # 初始化模型的预测损失
        Loss = CrossEntropyWithSoftmax(model, truth, output)
        # 前向传播损失并计算模型的预测损失
        ces_loss = Loss.forward()
        # 记录训练损失
        train_loss += ces_loss
        # 优化器梯度置0
        optimizer.zero_grad()
        # 将损失进行反向传播
        Loss.backward()
        # 优化器优化一步
        optimizer.step()
        
    return model, optimizer, train_loss
```

为了评估模型的准确率，还需要额外定义一个评估函数：

```python
import numpy as np

def valid_model(model, input_, truth):
    """对模型准确率进行评估"""
    # 得到模型的预测输出
    output = model.forward(input_)
    predict = np.argmax(output, axis=1)
    # 计算模型的预测准确率
    accuracy = np.array(predict == truth.flatten(), dtype=int).sum() / len(truth)
    return accuracy
```

接下来，可以对模型进行训练。其中，`make_circles` 函数的实现可以参考 [MLP_Classifier](../Models/MLP/MLP_Classifier.py)：

```python
from Core.Optimizer import Adam

# 获取同心圆状分布数据，X的每行包含两个特征，y是1/0类别标签
X, Y = make_circles(600, noise=0.12, factor=0.2)
Y = Y.reshape(-1, 1)  # 重整真实数据形状为(n, 1)
# 初始化之前定义的模型
model = MLPDemo()
num_epochs = 20  # 训练的总轮数
batch_size = 16  # 批处理的大小
# 初始化梯度优化器：使用Adam优化器，学习率设置为0.05
optimizer = Adam(model=model, learning_rate=0.05)
for epoch in range(num_epochs):
    # 训练模型，训练一轮（训练一个epoch）
    model, optimizer, train_loss = train_epoch(model, optimizer, X, Y, batch_size)
    # 对模型准确率进行一个评估
    accuracy = valid_model(model, X, Y)
    # 打印每轮训练的结果
    print("epoch: [{:d}/{:d}], loss: {:.3f}, accuracy: {:.3f}".format(epoch + 1, num_epochs, train_loss, accuracy))
```

通过以上步骤，即可以快速构建并训练一个简单的全连接神经网络模型。