import copy
import numpy as np
import matplotlib.pyplot as plt

from Core.Loss import MSELoss
from Core.Optimizer import Adam
from Core.Module import RNNModel


def plot_future(model, test_previous, gap, x_range, mse_loss, pause=True):
    Model = copy.deepcopy(model)
    test_previous = test_previous.reshape(1, -1, 1)
    y_plot = test_previous.flatten().copy()
    for i in range(200):
        test_next = Model(test_previous)
        y_plot = np.append(y_plot, test_next[0][0])
        # 数据偏移并加入预测元素
        test_previous[:, :-1, :] = test_previous[:, 1:, :]
        test_previous[:, -1, :] = test_next
    x_plot = (np.arange(len(y_plot)) * gap)
    truth_plot = 2 * np.sin(x_plot)

    # 画图中文显示会有问题，需要这两行设置默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.clf()
    plt.plot(x_plot, truth_plot, c='red', label='truth')
    plt.plot(x_plot, y_plot, c='blue', label='predict')
    plt.title('sin函数拟合结果', fontsize=18)
    info = f'损失值:{mse_loss:.6f}'
    plt.ylim([-2.5, 2.5])
    plt.xlabel(info)
    # 在x=20处添加一条竖线
    plt.axvline(x=x_range[1], color='r', linestyle='--')
    plt.text(15, -2.3, '训练数据', fontsize=12, color='red', ha='center')
    plt.text(25, -2.3, '预测数据', fontsize=12, color='blue', ha='center')
    plt.grid(True)
    plt.legend(loc='upper right')
    if pause:
        plt.pause(0.001)
    else:
        plt.show()


def get_predict_data(num_samples, num_steps, x_range):
    """获取预测数据(前n步预测下一步)"""
    X = np.linspace(x_range[0], x_range[1], num_samples + num_steps + 1)
    X = [X[i:i + num_steps + 1] for i in range(num_samples)]
    gap = X[0][1] - X[0][0]
    Y = 2 * np.sin(X).reshape(num_samples, -1)
    data_previous = Y[:, :-1].reshape(-1, num_steps, 1)
    data_next = Y[:, -1].reshape(-1, 1)
    return data_previous, data_next, gap


if __name__ == '__main__':
    np.random.seed(0)
    # 获取数据集
    num_samples, num_steps, x_range = 100, 10, [0, 20]
    data_previous, data_next, gap = get_predict_data(num_samples, num_steps, x_range)
    # 初始化模型
    model = RNNModel(input_size=1,
                     rnn_hidden_size=20,
                     num_layers=1,
                     linear_hidden_sizes=None,
                     output_size=1,
                     batch_first=True)
    # 初始化梯度优化器
    optimizer = Adam(model=model, learning_rate=0.01)
    # 对模型进行优化
    num_epochs = 500
    for epoch in range(num_epochs):
        output = model.forward(data_previous)
        Loss = MSELoss(model, data_next, output)
        mse_loss = Loss.forward()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {mse_loss.item()}')
        # 绘制优化过程中的模型表现
        plot_future(model, data_previous[0].copy(), gap, x_range, mse_loss, pause=True)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
