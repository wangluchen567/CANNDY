import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt

from Core.Layer import Linear
from Core.Optimizer import Adam
from Core.Activation import ReLU, Softmax
from Core.Loss import CrossEntropyWithSoftmax
from Core.Module import Module, MLP, ResidualFC


class ResNet(Module):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.Layers = [
            Linear(self.input_size, 128, ReLU),  # 降维
            ResidualFC(128, 128, [128], ReLU, ReLU),
            ResidualFC(128, 128, [128], ReLU, ReLU),
            ResidualFC(128, 128, [128], ReLU, ReLU),
            Linear(128, 64, ReLU),  # 降维
            ResidualFC(64, 64, [64], ReLU, ReLU),
            ResidualFC(64, 64, [64], ReLU, ReLU),
            ResidualFC(64, 64, [64], ReLU, ReLU),
            Linear(64, 32, ReLU),  # 降维
            ResidualFC(32, 32, [32], ReLU, ReLU),
            ResidualFC(32, 32, [32], ReLU, ReLU),
            ResidualFC(32, 32, [32], ReLU, ReLU),
            Linear(32, 16, ReLU),  # 降维
            ResidualFC(16, 16, [16], ReLU, ReLU),
            ResidualFC(16, 16, [16], ReLU, ReLU),
            ResidualFC(16, 16, [16], ReLU, ReLU),
            Linear(16, self.output_size, Softmax)
        ]

    def forward(self, input_):
        hidden = input_.copy()
        for layer in self.Layers:
            hidden = layer(hidden)
        output = hidden
        return output


class DeepMLP(Module):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.Layers = [
            Linear(self.input_size, 128, ReLU),  # 降维
            MLP(128, 128, [128], ReLU, ReLU),
            MLP(128, 128, [128], ReLU, ReLU),
            MLP(128, 128, [128], ReLU, ReLU),
            Linear(128, 64, ReLU),  # 降维
            MLP(64, 64, [64], ReLU, ReLU),
            MLP(64, 64, [64], ReLU, ReLU),
            MLP(64, 64, [64], ReLU, ReLU),
            Linear(64, 32, ReLU),  # 降维
            MLP(32, 32, [32], ReLU, ReLU),
            MLP(32, 32, [32], ReLU, ReLU),
            MLP(32, 32, [32], ReLU, ReLU),
            Linear(32, 16, ReLU),  # 降维
            MLP(16, 16, [16], ReLU, ReLU),
            MLP(16, 16, [16], ReLU, ReLU),
            MLP(16, 16, [16], ReLU, ReLU),
            Linear(16, self.output_size, Softmax)
        ]

    def forward(self, input_):
        hidden = input_.copy()
        for layer in self.Layers:
            hidden = layer(hidden)
        output = hidden
        return output


def load_data(data_path):
    """加载数据集"""
    with gzip.open(data_path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    x_train, y_train, x_valid, y_valid = map(
        np.array, (x_train, y_train, x_valid, y_valid)
    )

    return x_train, y_train, x_valid, y_valid


def train_epoch(model, optimizer, X, Y, batch_size):
    """训练一个epoch"""
    train_loss = 0
    for i in np.arange(0, len(X), batch_size):
        input_ = X[i:i + batch_size, :]
        truth = Y[i:i + batch_size, :]
        output = model.forward(input_)
        Loss = CrossEntropyWithSoftmax(model, truth, output)
        ces_loss = Loss.forward()
        train_loss += ces_loss
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
    return model, optimizer, train_loss


def train_model(model, train_data, train_label, valid_data, valid_label, num_epochs):
    """训练模型"""
    batch_size = 64
    optimizer = Adam(model=model, learning_rate=1e-3)
    loss_list, acc_list = [], []
    for epoch in range(num_epochs):
        model, optimizer, train_loss = train_epoch(model, optimizer, train_data, train_label, batch_size)
        accuracy = valid_model(model, valid_data, valid_label)
        print("epoch: [{:d}/{:d}], loss: {:.3f}, accuracy: {:.3f}".
              format(epoch + 1, num_epochs, train_loss, accuracy))
        loss_list.append(train_loss)  # 记录损失
        acc_list.append(accuracy)  # 记录准确率
    return model, loss_list, acc_list


def valid_model(model, input_, truth):
    output = model.forward(input_)
    predict = np.argmax(output, axis=1)
    accuracy = np.array(predict == truth.flatten(), dtype=int).sum() / len(truth)
    return accuracy


if __name__ == '__main__':
    np.random.seed(0)
    data_path = "../../Dataset/mnist.pkl.gz"
    x_train, y_train, x_valid, y_valid = load_data(data_path)
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    train_size = 10000
    valid_size = 1000
    # 设置训练epoch数
    num_epoch = 20

    # 创建深度MLP模型
    model = DeepMLP(784, 10)
    # 训练模型
    print("training deep-mlp...")
    model, mlp_loss, mlp_acc = train_model(model,
                                           x_train[:train_size], y_train[:train_size],
                                           x_valid[:valid_size], y_valid[:valid_size],
                                           num_epoch)
    accuracy = valid_model(model, x_valid, y_valid)
    print("full dataset accuracy: {:.3f} %".format(accuracy * 100))

    # 创建ResNet模型
    model = ResNet(784, 10)
    # 训练模型
    print("training res-net...")
    model, resnet_loss, resnet_acc = train_model(model,
                                                 x_train[:train_size], y_train[:train_size],
                                                 x_valid[:valid_size], y_valid[:valid_size],
                                                 num_epoch)
    accuracy = valid_model(model, x_valid, y_valid)
    print("full dataset accuracy: {:.3f} %".format(accuracy * 100))

    # 将对比结果进行绘图展示
    plt.figure(0)
    num_loss = len(mlp_loss)
    plt.plot(np.arange(num_loss)+1, mlp_loss, marker='o', label='deep-mlp loss')
    plt.plot(np.arange(num_loss)+1, resnet_loss, marker='o', label='res-net loss')
    plt.title('Loss Comparison')
    plt.legend()
    plt.grid()

    plt.figure(1)
    num_acc = len(mlp_acc)
    plt.plot(np.arange(num_acc)+1, mlp_acc, marker='o', label='deep-mlp accuracy')
    plt.plot(np.arange(num_acc)+1, resnet_acc, marker='o', label='res-net accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.grid()

    plt.show()
