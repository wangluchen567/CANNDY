import numpy as np
import matplotlib.pyplot as plt

from MLP import MLP
from Core.Loss import MSELoss
from Core.Optimizer import Adam


def learning_all(model, X, Y):
    """一口气学习所有数据"""
    optimizer = Adam(model=model, learning_rate=0.01)
    for epoch in range(5000):
        optimizer.zero_grad()
        input, truth = X.T, Y.T
        output = model.forward(input)
        Loss = MSELoss(model, truth, output)
        mse_loss = Loss.forward()
        print(epoch + 1, mse_loss)
        Loss.backward()
        optimizer.step()
    return model


def learning_batch(model, X, Y):
    """一个batch一个batch的学习"""
    batch_size = 10
    optimizer = Adam(model=model, learning_rate=0.01)
    for epoch in range(5000):
        for i in np.arange(0, len(X), batch_size):
            input = X[i:i + batch_size, :].T
            truth = Y[i:i + batch_size, :].T
            optimizer.zero_grad()
            output = model.forward(input)
            Loss = MSELoss(model, truth, output)
            mse_loss = Loss.forward()
            print(epoch + 1, mse_loss)
            Loss.backward()
            optimizer.step()
    return model


def learning_single(model, X, Y):
    """按照batch_size一个一个学习(累计梯度)"""
    batch_size = 10
    optimizer = Adam(model=model, learning_rate=0.01)
    for epoch in range(1000):
        # 批计数清零
        batch_count = 0
        # 累计loss置零
        sum_loss = 0
        for i in range(len(X)):
            input = X[i].reshape(1, -1)
            truth = Y[i].reshape(1, -1)
            output = model.forward(input)
            Loss = MSELoss(model, truth, output)
            mse_loss = Loss.forward()
            sum_loss += mse_loss
            Loss.backward()
            batch_count += 1
            if batch_count >= batch_size:
                # 更新一次梯度
                optimizer.step()
                # 梯度置零
                optimizer.zero_grad()
                # 批计数清零
                batch_count = 0
        print(epoch + 1, sum_loss)
    return model


if __name__ == '__main__':
    X = np.random.uniform(0, 10, size=(100, 1))
    Y = np.sin(X)
    model = MLP(1, 1, [8, 16, 8])
    # model = learning_all(model, X, Y)
    model = learning_batch(model, X, Y)
    # model = learning_single(model, X, Y)

    X = np.arange(0, 10, 0.01).reshape(1, -1)
    Y = np.sin(X)
    output = model.forward(X)
    plt.figure()
    plt.plot(X.flatten(), Y.flatten(), c='red')
    plt.plot(X.flatten(), output.flatten(), c='blue')
    plt.show()
