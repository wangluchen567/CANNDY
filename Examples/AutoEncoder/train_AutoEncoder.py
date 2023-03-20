import time
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from Core.Loss import MSELoss
from Core.Optimizer import Adam
from AutoEncoder import AutoEncoder


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
    for i in tqdm(np.arange(0, len(X), batch_size)):
        input = X[i:i + batch_size, :].T
        truth = Y[i:i + batch_size, :].T
        optimizer.zero_grad()
        output = model.forward(input)
        Loss = MSELoss(model, truth, output)
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
    for i in tqdm(range(len(X))):
        input = X[i, :].reshape(-1, 1)
        truth = Y[i, :].reshape(-1, 1)
        output = model.forward(input)
        Loss = MSELoss(model, truth, output)
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


def val_epoch(model, X, Y, batch_size):
    """验证一个epoch"""
    val_loss = 0
    for i in np.arange(0, len(X), batch_size):
        input = X[i:i + batch_size, :].T
        truth = Y[i:i + batch_size, :].T
        output = model.forward(input)
        Loss = MSELoss(model, truth, output)
        ces_loss = Loss.forward()
        val_loss += ces_loss
        Loss.backward()
    return val_loss


def train_AutoEncoder(x_train, x_valid):
    model = AutoEncoder()
    batch_size = 64
    optimizer = Adam(model=model, learning_rate=0.001)
    for epoch in range(10):
        start = time.time()
        model, optimizer, train_loss = train_epoch(model, optimizer, x_train, x_train, batch_size)
        val_loss = val_epoch(model, x_valid, x_valid, batch_size)
        epoch_time = time.time() - start
        print("epoch:{:d}  train_loss:{:.5f}  val_loss:{:.5f}  epoch_time:{:.5f} s".format(
            epoch + 1, train_loss, val_loss, epoch_time))
        AutoEncoder_test(model, x_valid, index=0, pause=True)
    return model


def AutoEncoder_test(model, x_valid, index=0, pause=True):
    """输入某个验证集数据, 查看效果"""
    index = index  # 要查看效果的数据下标
    x = x_valid[index].reshape(-1, 1)
    x_hat = model.forward(x)
    fig = plt.figure(0)
    fig.add_subplot(1, 2, 1)
    plt.imshow(x.reshape((28, 28)), cmap="gray")
    fig.add_subplot(1, 2, 2)
    plt.imshow(x_hat.reshape((28, 28)), cmap="gray")
    if pause:
        plt.pause(0.1)
    else:
        plt.show()


def reconstruct_show(model, x_valid):
    """输入多个验证集数据进行重构, 查看效果"""
    x_sample = x_valid[10:60]  # 数据范围可自行选择
    x_reconstruct = model.forward(x_sample.T)
    x_reconstruct = x_reconstruct.T
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_path = "../../Dataset/mnist.pkl.gz"
    x_train, y_train, x_valid, y_valid = load_data(data_path)
    model = train_AutoEncoder(x_train[:5000], x_valid[:100])
    AutoEncoder_test(model, x_valid, index=0, pause=False)
    reconstruct_show(model, x_valid)

