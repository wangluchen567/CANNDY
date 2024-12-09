import json
import gzip
import pickle
import numpy as np
# 需要有Pillow库读取图像
from PIL import Image
from Core.Module import LeNet5
import matplotlib.pyplot as plt


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


def valid_model(model, input_, truth):
    model.eval()
    input_ = input_.reshape(-1, 1, 28, 28)
    output = model.forward(input_)
    predict = np.argmax(output, axis=1)
    accuracy = np.array(predict == truth.flatten(), dtype=int).sum() / len(truth)
    return accuracy


def load_image(data_path):
    """读取图像用来测试"""
    # 读取测试图像
    image = Image.open(data_path)
    # 将图片转换为数组
    image_array = np.array(image)
    if image_array.ndim == 3:
        image_array = image_array.sum(-1)
    # 将图像数组归一化
    image_array = image_array.max() - image_array
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    return image_array


def plot_feature_maps(feature_map, title):
    width = feature_map.shape[-1]
    feature_map = feature_map.reshape(-1, width, width)
    N = feature_map.shape[0]
    if N == 6:
        rows, cols = 2, 3
    elif N == 16:
        rows, cols = 4, 4
    else:
        raise ValueError("Unsupported number of subplots")
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axs = axs.flatten()
    for i in range(N):
        axs[i].imshow(feature_map[i])
        axs[i].axis('off')  # 不显示坐标轴
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_path = "../../Dataset/mnist.pkl.gz"
    x_train, y_train, x_valid, y_valid = load_data(data_path)
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_valid = x_valid.reshape(-1, 1, 28, 28)
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    # 创建模型
    model = LeNet5()
    # 读取模型参数
    with open('LeNet-5_Params.json', 'r') as f:
        params_dict = json.load(f)
    model.set_parameters(params_dict)

    # 对数据集进行标准化
    # print('standardize dataset...')
    # x_train, mean, std = standardize_data(x_train)
    # x_valid, _, _ = standardize_data(x_valid, mean, std)
    # print(f'mean: {mean}, std: {std}')

    # 得到训练集与测试集的准确率
    # train_acc = valid_model(model, x_train, y_train)
    # print("train acc: {:.3f}".format(train_acc * 100))
    valid_acc = valid_model(model, x_valid, y_valid)
    print("Test Accuracy: {:.3f} %".format(valid_acc * 100))

    # 使用数据集中的数据
    # index = 0
    # test_data = x_valid[index]

    # 使用自己手写的测试数据

    data_path = '../../Dataset/Mnist_Test/1.png'
    test_data = load_image(data_path)

    # 绘制输入图像
    plt.figure()
    plt.imshow(test_data)
    plt.title('Input')
    plt.tight_layout()
    plt.show()

    # 调整为评估模式
    model.eval()
    # 对数据进行标准化
    # test_data, _, _ = standardize_data(test_data, mean, std)
    input_ = test_data.reshape(-1, 1, 28, 28)

    # 前向传播得到每步的结果
    hidden = input_.copy()
    for layer in model.Layers:
        hidden = layer(hidden)
        # 绘制特征图
        if layer.__class__.__name__ == 'Conv2d':
            plot_feature_maps(hidden, 'Conv2d')
        if layer.__class__.__name__ == 'MaxPool2d':
            plot_feature_maps(hidden, 'MaxPool2d')
    output = hidden
    predict = np.argmax(output, axis=1)[0]

    plt.figure()
    # 画图中文显示会有问题，需要这两行设置默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.imshow(test_data)
    plt.title('Result')
    plt.xlabel('识别结果：{:d}, 识别概率: {:.3f} %'.format(predict, output[0, predict] * 100))
    plt.show()
