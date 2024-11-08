import numpy as np
import pandas as pd

from MLP import MLP
from Core.Optimizer import Adam
from Core.Activation import Softmax
from Core.Loss import CrossEntropyWithSoftmax


def train_epoch(model, optimizer, X, Y, batch_size):
    """训练一个epoch"""
    train_loss = 0
    for i in np.arange(0, len(X), batch_size):
        input = X[i:i + batch_size, :]
        truth = Y[i:i + batch_size, :]
        output = model.forward(input)
        Loss = CrossEntropyWithSoftmax(model, truth, output)
        ces_loss = Loss.forward()
        train_loss += ces_loss
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
    return model, optimizer, train_loss


def train_one_by_one(model, optimizer, X, Y, batch_size):
    """训练一个epoch(一个一个的训练后累加)"""
    # 批计数清零
    batch_count = 0
    # 累计loss置零
    train_loss = 0
    for i in range(len(X)):
        input = X[i, :].reshape(1, -1)
        truth = Y[i, :].reshape(1, -1)
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


def train_model(model, train_data, train_label, valid_data, valid_label, num_epochs=100):
    """训练模型"""
    batch_size = 10
    optimizer = Adam(model=model, learning_rate=0.02)
    for epoch in range(num_epochs):
        model, optimizer, train_loss = train_epoch(model, optimizer, train_data, train_label, batch_size)
        accuracy = valid_model(model, valid_data, valid_label)
        print("epoch: [{:d}/{:d}], loss: {:.3f}, accuracy: {:.3f}".
              format(epoch + 1, num_epochs, train_loss, accuracy))
    return model


def valid_model(model, input, truth):
    output = model.forward(input)
    predict = np.argmax(output, axis=1)
    accuracy = np.array(predict == truth.flatten(), dtype=int).sum() / len(truth)
    return accuracy


if __name__ == '__main__':
    # 读取数据集
    data = pd.read_csv("../Dataset/Iris.csv")
    # 将数据集中的每种花换成整数0, 1, 2
    data.iloc[np.where(data['Species'] == 'Iris-setosa')[0], -1] = 0
    data.iloc[np.where(data['Species'] == 'Iris-versicolor')[0], -1] = 1
    data.iloc[np.where(data['Species'] == 'Iris-virginica')[0], -1] = 2
    # 将Species列的数据设置类型为int
    data['Species'] = data['Species'].astype(int)
    # 数据集特征
    features = data[['SepalLengthCm',
                     'SepalWidthCm',
                     'PetalLengthCm',
                     'PetalWidthCm']].values
    # 数据集标签
    labels = data[['Species']].values
    # 打乱数据集
    random_index = np.arange(len(features))
    np.random.shuffle(random_index)
    features = features[random_index]
    labels = labels[random_index]

    # 划分训练集和验证集
    train_size = 100
    train_dataset = features[:train_size]
    train_labels = labels[:train_size]
    valid_dataset = features[train_size:]
    valid_labels = labels[train_size:]

    # 创建模型
    model = MLP(4, 3, [10, 10], out_act=Softmax)
    model = train_model(model, train_dataset, train_labels, valid_dataset, valid_labels)
    accuracy = valid_model(model, features, labels)
    print("full dataset accuracy: {:.3f} %".format(accuracy * 100))
    accuracy = valid_model(model, valid_dataset, valid_labels)
    print("val dataset accuracy: {:.3f} %".format(accuracy * 100))
