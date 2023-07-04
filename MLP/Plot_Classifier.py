import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_classifier(model, X, Y, accuracy, pause=True):
    # 画图并保存图像
    plt.figure(0)
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
    y_hat = y_hat.argmax(axis=0)
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
    info = '准确率:%.2f%%' % (accuracy * 100)
    plt.xlabel(info)
    # plt.text(5, 0.5, info, weight="bold")
    plt.grid(True)
    if pause:
        plt.pause(0.1)
    else:
        plt.show()

def plot_classifier_soft(model, X, Y, accuracy, pause=True):
    # 画图并保存图像
    plt.figure(0)
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
    y_hat_p = y_hat[0, :] - y_hat[1, :]
    y_hat_p = y_hat_p.reshape(x1_mesh.shape)  # 使之与输入的形状相同
    cm_dark = matplotlib.colors.ListedColormap(['b', 'r'])  # 两种样本显示颜色
    plt.pcolormesh(x1_mesh, x2_mesh, y_hat_p, shading='auto', cmap=plt.get_cmap("rainbow"))  # 预测值的显示
    plt.scatter(Data[:, 0], Data[:, 1], c=Label.flat, edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
    # 固定大小以美观
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('同心圆分类结果', fontsize=18)
    # 打印准确率
    info = '准确率:%.2f%%' % (accuracy * 100)
    plt.xlabel(info)
    # plt.text(5, 0.5, info, weight="bold")
    plt.grid(True)
    if pause:
        plt.pause(0.1)
    else:
        plt.show()
