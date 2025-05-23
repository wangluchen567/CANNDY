import time
import json
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from Core.Module import LeNet5
from Core.Optimizer import Adam
from Core.Loss import CrossEntropyWithSoftmax


def load_data(data_path):
    """加载数据集"""
    with gzip.open(data_path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    x_train, y_train, x_valid, y_valid = map(
        np.array, (x_train, y_train, x_valid, y_valid)
    )

    return x_train, y_train, x_valid, y_valid


def standardize_data(data, mean=None, std=None):
    """对数据进行标准化"""
    if mean is None or std is None:
        mean = np.mean(data)
        std = np.std(data)
    return (data - mean) / std, mean, std


def train_epoch(model, optimizer, X, Y, batch_size):
    """训练一个epoch"""
    # 设置为训练模式
    model.train()
    train_loss = 0
    for i in tqdm(np.arange(0, len(X), batch_size)):
        input_ = X[i:i + batch_size]
        truth = Y[i:i + batch_size]
        output = model.forward(input_)
        Loss = CrossEntropyWithSoftmax(model, truth, output)
        ces_loss = Loss.forward()
        train_loss += ces_loss
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
    return model, optimizer, train_loss


def train_model(model, train_data, train_label, valid_data, valid_label,
                num_epochs=20, batch_size=64, save_checkpoint=True):
    """训练模型"""
    optimizer = Adam(model=model, learning_rate=1.e-3)
    for epoch in range(num_epochs):
        start = time.time()
        model, optimizer, train_loss = train_epoch(model, optimizer, train_data, train_label, batch_size)
        accuracy = valid_model(model, valid_data, valid_label)
        print("epoch: [{:d}/{:d}], loss: {:.3f}, accuracy: {:.5f}, time(s): {:.3f}".
              format(epoch + 1, num_epochs, train_loss, accuracy, time.time() - start))
        if save_checkpoint:
            # 保存每个epoch的模型参数
            params_dict = model.get_parameters()
            with open('LeNet-5_Params_' + str(epoch + 1) + '.json', 'w') as f:
                json.dump(params_dict, f)
    return model


def valid_model(model, input_, truth):
    # 设置为评估模式
    model.eval()
    output = model.forward(input_)
    predict = np.argmax(output, axis=1)
    accuracy = np.array(predict == truth.flatten(), dtype=int).sum() / len(truth)
    return accuracy


def show_image(image, label):
    plt.imshow(image.reshape(28, 28))
    plt.title(f'Label: {label}')
    plt.show()


if __name__ == '__main__':
    np.random.seed(6)
    data_path = "../../Dataset/mnist.pkl.gz"
    x_train, y_train, x_valid, y_valid = load_data(data_path)
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_valid = x_valid.reshape(-1, 1, 28, 28)
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    # 只训练一部分的设置
    # train_size = 10000
    # valid_size = 1000
    # 数据集全部训练的设置
    train_size = 50000
    valid_size = 10000

    # 展示前几个数据
    for i in range(3):
        show_image(x_train[i], y_train[i][0])

    # 对数据进行标准化以提升效果
    # print('standardize dataset...')
    # x_train, mean, std = standardize_data(x_train)
    # x_valid, _, _ = standardize_data(x_valid, mean, std)
    # print(f'mean: {mean}, std: {std}')

    # 创建模型
    model = LeNet5()
    # 开始训练
    start = time.time()
    model = train_model(model, x_train[:train_size], y_train[:train_size], x_valid[:valid_size], y_valid[:valid_size],
                        num_epochs=30, batch_size=64, save_checkpoint=True)
    train_time = time.time() - start
    accuracy = valid_model(model, x_valid, y_valid)
    print("Test accuracy: {:.3f} %".format(accuracy * 100))
    print("running time: {:.3f}".format(train_time))
    # 保存每个epoch的模型参数
    params_dict = model.get_parameters()
    with open('LeNet-5_Params_.json', 'w') as f:
        json.dump(params_dict, f)
