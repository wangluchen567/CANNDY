import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_classifier(model, X, Y, accuracy, n_iter=None, max_iter=None, plot_mode=0, pause=True, pause_time=0.1):
    """
    绘制分类结果（0：硬边缘，1：软边缘，2: 3d效果）
    :param model: 给定模型
    :param X: 训练数据
    :param Y: 训练标签
    :param accuracy: 当前准确率
    :param n_iter: 当前迭代次数
    :param max_iter: 最大迭代次数
    :param plot_mode: 绘图方式（0：硬边缘，1：软边缘，2: 3d效果）
    :param pause: 是否暂停
    :param pause_time: 暂停时间
    """
    if n_iter is not None and max_iter is not None and n_iter == max_iter:
        # 最后一次迭代不再使用停顿展示
        pause = False
    if plot_mode == 0:
        plot_classifier_edge(model, X, Y, accuracy, n_iter, pause, pause_time)
    elif plot_mode == 1:
        plot_classifier_soft(model, X, Y, accuracy, n_iter, pause, pause_time)
    elif plot_mode == 2:
        plot_classifier_3d(model, X, Y, accuracy, n_iter, pause, pause_time)
    else:
        raise ValueError(f"There is no such plotting mode: {plot_mode}")


def plot_classifier_edge(model, X, Y, accuracy, n_iter=None, pause=True, pause_time=0.1):
    """绘制分类结果（边缘明显）"""
    # plt.figure(0)
    plt.clf()
    # 画图中文显示会有问题，需要这两行设置默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    Data = X  # 将数据放入Data
    Label = Y  # 将标签放入Label
    N, M = 300, 300  # 横纵各采样多少个值
    x1_min, x1_max = np.min(Data[:, 0], axis=0), np.max(Data[:, 1], axis=0)  # 第0列的范围
    x2_min, x2_max = np.min(Data[:, 0], axis=0), np.max(Data[:, 1], axis=0)  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1_mesh, x2_mesh = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test = np.stack((x1_mesh.flat, x2_mesh.flat), axis=1)  # 得到测试点
    y_hat = model.forward(x_test)
    # 将标签转化为0/1
    y_hat = y_hat.argmax(axis=1)
    y_hat = y_hat.reshape(x1_mesh.shape)  # 使之与输入的形状相同
    # cm_light = matplotlib.colors.ListedColormap(['#FF8080', '#A0A0FF', '#77E0A0'])  # 三种背景颜色
    # cm_dark = matplotlib.colors.ListedColormap(['r', 'b', 'g'])  # 三种样本显示颜色
    cm_light = matplotlib.colors.ListedColormap(['#A0A0FF', '#FF8080'])  # 两种背景颜色
    cm_dark = matplotlib.colors.ListedColormap(['b', 'r'])  # 两种样本显示颜色
    plt.pcolormesh(x1_mesh, x2_mesh, y_hat, shading='auto', cmap=cm_light)  # 预测值的显示
    plt.scatter(Data[:, 0], Data[:, 1], c=Label.flat, edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
    # 固定大小以美观
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('同心圆分类结果', fontsize=18)
    # 打印准确率
    info = ""
    info += f"迭代次数: {n_iter}, " if n_iter is not None else ""
    info += f"准确率: {(accuracy * 100) :.2f} %"
    plt.xlabel(info)
    # plt.text(5, 0.5, info, weight="bold")
    plt.grid(True)
    if pause:
        plt.pause(pause_time)
    else:
        plt.show()


def plot_classifier_soft(model, X, Y, accuracy, n_iter=None, pause=True, pause_time=0.1):
    """绘制分类结果（无边缘，绘制分类概率）"""
    # plt.figure(0)
    plt.clf()
    # 画图中文显示会有问题，需要这两行设置默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    Data = X  # 将数据放入Data
    Label = Y  # 将标签放入Label
    N, M = 300, 300  # 横纵各采样多少个值
    x1_min, x1_max = np.min(Data[:, 0], axis=0), np.max(Data[:, 1], axis=0)  # 第0列的范围
    x2_min, x2_max = np.min(Data[:, 0], axis=0), np.max(Data[:, 1], axis=0)  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1_mesh, x2_mesh = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test = np.stack((x1_mesh.flat, x2_mesh.flat), axis=1)  # 得到测试点
    y_hat = model.forward(x_test)
    # 直接映射为实数
    y_hat_p = y_hat[:, 1] - y_hat[:, 0]
    y_hat_p = y_hat_p.reshape(x1_mesh.shape)  # 使之与输入的形状相同
    cm_dark = matplotlib.colors.ListedColormap(['b', 'r'])  # 两种样本显示颜色
    plt.pcolormesh(x1_mesh, x2_mesh, y_hat_p, shading='auto', cmap=plt.get_cmap("rainbow"))  # 预测值的显示
    plt.scatter(Data[:, 0], Data[:, 1], c=Label.flat, edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
    # 固定大小以美观
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('同心圆分类结果', fontsize=18)
    # 打印准确率
    info = ""
    info += f"迭代次数: {n_iter}, " if n_iter is not None else ""
    info += f"准确率: {(accuracy * 100) :.2f} %"
    plt.xlabel(info)
    # plt.text(5, 0.5, info, weight="bold")
    plt.grid(True)
    if pause:
        plt.pause(pause_time)
    else:
        plt.show()


def plot_classifier_3d(model, X, Y, accuracy, n_iter=None, pause=True, pause_time=0.1):
    """绘制分类结果（绘制 3d 分类效果）"""
    # plt.figure(0)
    plt.clf()
    # 画图中文显示会有问题，需要这两行设置默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax = plt.subplot(111, projection='3d')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
    x_data = X
    y_data = Y
    # 绘制面
    x1_grid, x2_grid = np.meshgrid(np.linspace(x_data[:, 0].min(), x_data[:, 0].max(), 100),
                                   np.linspace(x_data[:, 1].min(), x_data[:, 1].max(), 100))
    # 得到模型预测值
    y_hat = model.forward(np.stack((x1_grid.flat, x2_grid.flat), axis=1))
    # x_grid_b = np.stack((x1_grid, x2_grid, np.ones_like(x1_grid)), axis=-1)
    ax.plot_surface(x1_grid, x2_grid, y_hat[:, 1].reshape(x1_grid.shape), alpha=0.5, cmap='viridis')
    # 绘制点
    positive, negative = np.array(y_data == 1).flatten(), np.array(y_data == 0).flatten()
    ax.scatter(x_data[positive, 0], x_data[positive, 1], y_data[positive, 0], marker="o", c="red")
    ax.scatter(x_data[negative, 0], x_data[negative, 1], y_data[negative, 0], marker="o", c="blue")
    ax.view_init(elev=30)
    # if n_iter is not None:
    #     ax.view_init(azim=(n_iter * 2) % 360)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # 打印准确率
    info = ""
    info += f"迭代次数: {n_iter}, " if n_iter is not None else ""
    info += f"准确率: {(accuracy * 100) :.2f} %"
    plt.xlabel(info)
    if pause:
        plt.pause(pause_time)
    else:
        plt.show()
