import gzip
import pickle
import numpy as np

from MLP import MLP
from Core.Optimizer import Adam
from Core.Activation import Softmax
from Core.Loss import CrossEntropyWithSoftmax


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
        input = X[i:i + batch_size, :].T
        truth = Y[i:i + batch_size, :].T
        optimizer.zero_grad()
        output = model.forward(input)
        Loss = CrossEntropyWithSoftmax(model, truth, output)
        ces_loss = Loss.forward()
        train_loss += ces_loss
        Loss.backward()
        optimizer.step()
    return model, optimizer, train_loss


def train_onebyone(model, optimizer, X, Y, batch_size):
    """训练一个epoch(一个一个的训练后累加)"""
    # 批计数清零
    batch_count = 0
    # 累计loss置零
    train_loss = 0
    for i in range(len(X)):
        input = X[i, :].reshape(-1, 1)
        truth = Y[i, :].reshape(-1, 1)
        output = model.forward(input)
        Loss = CrossEntropyWithSoftmax(model, truth, output)
        ces_loss = Loss.forward()
        train_loss += ces_loss
        Loss.backward()
        batch_count += 1
        if batch_count >= batch_size:
            # 更新一次梯度
            optimizer.step()
            # 梯度置零
            optimizer.zero_grad()
            # 批计数清零
            batch_count = 0
    return model, optimizer, train_loss


def train_model(model, train_data, train_label, valid_data, valid_label):
    """训练模型"""
    batch_size = 64
    optimizer = Adam(model=model, learning_rate=0.001)
    for epoch in range(30):
        model, optimizer, train_loss = train_epoch(model, optimizer, train_data, train_label, batch_size)
        accuracy = valid_model(model, valid_data, valid_label)
        print("epoch: {:d}, loss: {:.3f}, accuracy: {:.3f}".format(epoch + 1, train_loss, accuracy))
    return model


def valid_model(model, X, Y):
    input = X.T
    truth = Y.T
    output = model.forward(input)
    predict = np.argmax(output, axis=0)
    accuracy = np.array(predict == truth, dtype=int).sum() / len(Y)
    return accuracy


if __name__ == '__main__':
    data_path = "../Dataset/mnist.pkl.gz"
    x_train, y_train, x_valid, y_valid = load_data(data_path)
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    train_size = 5000
    valid_size = 100
    # 创建模型
    model = MLP(784, 10, [100, 20], out_act=Softmax)
    model = train_model(model, x_train[:train_size], y_train[:train_size], x_valid[:valid_size], y_valid[:valid_size])
    accuracy = valid_model(model, x_valid, y_valid)
    print("full dataset accuracy: {:.3f} %".format(accuracy * 100))
    accuracy = valid_model(model, x_valid[:valid_size], y_valid[:valid_size])
    print("val dataset accuracy: {:.3f} %".format(accuracy * 100))
