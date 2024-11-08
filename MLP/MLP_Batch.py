import numpy as np
import matplotlib.pyplot as plt

from MLP import MLP
from Core.Loss import MSELoss
from Core.Optimizer import Adam


def learning_all(model, X, Y, num_epochs=1000):
    """一口气学习所有数据"""
    optimizer = Adam(model=model, learning_rate=0.01)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        input, truth = X, Y
        output = model.forward(input)
        Loss = MSELoss(model, truth, output)
        mse_loss = Loss.forward()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {mse_loss.item()}')
        Loss.backward()
        optimizer.step()
    return model


def learning_batch(model, X, Y, num_epochs=1000):
    """一个batch一个batch的学习"""
    batch_size = 10
    optimizer = Adam(model=model, learning_rate=0.01)
    for epoch in range(num_epochs):
        for i in np.arange(0, len(X), batch_size):
            input = X[i:i + batch_size, :]
            truth = Y[i:i + batch_size, :]
            output = model.forward(input)
            Loss = MSELoss(model, truth, output)
            mse_loss = Loss.forward()
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Batch:{int(i / batch_size) + 1}, '
                  f'Loss: {mse_loss.item()}')
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
    return model


def learning_single(model, X, Y, num_epochs=1000):
    """按照batch_size一个一个学习(累计梯度，等价于batch)"""
    batch_size = 10
    optimizer = Adam(model=model, learning_rate=0.01)
    for epoch in range(num_epochs):
        # 批计数清零
        batch_count = 0
        # 累计loss置零
        sum_loss = 0
        for i in range(len(X)):
            input = X[i].reshape(-1, 1)
            truth = Y[i].reshape(-1, 1)
            output = model.forward(input)
            Loss = MSELoss(model, truth, output)
            mse_loss = Loss.forward()
            sum_loss += mse_loss
            Loss.backward()
            batch_count += 1
            if batch_count >= batch_size:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Batch:{int((i+1) / batch_size)}, '
                      f'Loss: {sum_loss}')
                # 这里梯度是累计梯度所以需要取平均
                for layer in model.Layers:
                    layer.grad /= batch_size
                # 更新一次梯度
                optimizer.step()
                # 梯度置零
                optimizer.zero_grad()
                # 批计数清零
                batch_count = 0
                # 累计loss置零
                sum_loss = 0
    return model


if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.uniform(0, 10, size=(100, 1))
    Y = np.sin(X)
    model = MLP(1, 1, [8, 16, 8])

    # model = learning_all(model, X, Y)
    model = learning_batch(model, X, Y)
    # model = learning_single(model, X, Y)

    X = np.arange(0, 10, 0.01).reshape(-1, 1)
    Y = np.sin(X)
    output = model.forward(X)
    plt.figure()
    plt.plot(X.flatten(), Y.flatten(), c='red')
    plt.plot(X.flatten(), output.flatten(), c='blue')
    plt.show()
