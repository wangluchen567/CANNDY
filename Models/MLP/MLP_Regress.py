import numpy as np
import matplotlib.pyplot as plt

from Core.Module import MLP
from Core.Loss import MSELoss
from Core.Optimizer import Adam


def plot_sin(model, pause=True):
    """绘制模型的回归预测表现"""
    x = np.arange(0, 10, 0.01).reshape(-1, 1)
    y = np.sin(x)
    output = model.forward(x)
    Loss = MSELoss(model, y, output)
    mse_loss = Loss.forward()

    plt.figure(0)
    # 画图中文显示会有问题，需要这两行设置默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.clf()
    plt.plot(x.flatten(), y.flatten(), c='red', label='truth')
    plt.plot(x.flatten(), output.flatten(), c='blue', label='predict')
    plt.title('sin函数拟合结果', fontsize=18)
    info = '损失值:%.3f' % (mse_loss)
    plt.ylim([-1.2, 1.2])
    plt.xlabel(info)
    plt.grid(True)
    plt.legend()
    if pause:
        plt.pause(0.001)
    else:
        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    num_samples = 100
    X = np.random.uniform(0, 10, size=(num_samples, 1))
    Y = np.sin(X)
    model = MLP(1, 1, [8, 16, 8])

    optimizer = Adam(model=model, learning_rate=0.01)
    num_epochs = 1000
    for epoch in range(num_epochs):
        output = model.forward(X)
        Loss = MSELoss(model, Y, output)
        mse_loss = Loss.forward()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {mse_loss.item()}')
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        plot_sin(model)

    plot_sin(model, pause=False)
