import time
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from Core.Optimizer import Adam
from Core.Module import MiniVGG
from Core.Loss import CrossEntropyWithSoftmax


def unpickle(file):
    """读取其中一个数据"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_batch(file):
    """读取其中一个batch数据"""
    batch = unpickle(file)
    images = batch[b'data']
    labels = batch[b'labels']
    # images的形状为[10000, 3072]，需要转换为[10000, 3, 32, 32]
    images = images.reshape(-1, 3, 32, 32)
    labels = np.array(labels)
    return images, labels

def standardize_data(data, mean=None, std=None):
    """标准化数据"""
    # 先对数据归一化
    data = data / 255
    # 再对数据进行标准化
    data = data.transpose(0, 2, 3, 1)
    if mean is None or std is None:
        mean = np.mean(data, axis=(0, 1, 2))
        std = np.std(data, axis=(0, 1, 2))
    data = data - mean / std
    return data.transpose(0, 3, 1, 2), mean, std


def train_epoch(model, optimizer, X, Y, batch_size):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    for i in tqdm(np.arange(0, len(X), batch_size)):
        # 需要将input形状调整为(batch_size, in_channels, height, width)(NCHW格式)
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
        print("epoch: [{:d}/{:d}], loss: {:.3f}, accuracy: {:.3f}, time(s): {:.3f}".
              format(epoch + 1, num_epochs, train_loss, accuracy, time.time() - start))
        if save_checkpoint:
            # 保存每个epoch的模型参数
            params_dict = model.get_parameters()
            with open('MiniVGG_Params_' + str(epoch + 1) + '.json', 'w') as f:
                json.dump(params_dict, f)
    return model


def valid_model(model, input_, truth):
    model.eval()
    output = model.forward(input_)
    predict = np.argmax(output, axis=1)
    accuracy = np.array(predict == truth.flatten(), dtype=int).sum() / len(truth)
    return accuracy


def show_image(image, label):
    image = image.transpose(1, 2, 0)
    plt.imshow(image)
    plt.title(f'Label: {label}')
    plt.show()


if __name__ == '__main__':
    file_path = '../../Dataset/cifar-10-batches-py/'
    # 读取训练集与测试集
    x_train, y_train = np.zeros((0, 3, 32, 32)), np.zeros(0)
    for i in range(5):
        data_path = file_path + 'data_batch_' + str(i + 1)
        images, labels = load_batch(data_path)
        x_train = np.concatenate((x_train, images), axis=0)
        y_train = np.concatenate((y_train, labels))
    x_valid, y_valid = load_batch(file_path + 'test_batch')
    # 每类的名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # 展示前10个数据
    # for i in range(10):
    #     show_image(x_train[i], class_names[y_train[i]])
    # 若要全部训练需要设置为50k
    train_size = 10000
    valid_size = 1000
    # 对数据进行标准化以提升效果
    print('standardize dataset...')
    x_train, mean, std = standardize_data(x_train)
    x_valid, _, _ = standardize_data(x_valid, mean, std)
    print(f'mean: {mean}\nstd: {std}')
    # 创建模型
    model = MiniVGG()
    start = time.time()
    model = train_model(model, x_train[:train_size], y_train[:train_size], x_valid[:valid_size], y_valid[:valid_size],
                        num_epochs=30, batch_size=64, save_checkpoint=True)
    train_time = time.time() - start
    accuracy = valid_model(model, x_valid, y_valid)
    print("Test accuracy: {:.3f} %".format(accuracy * 100))
    print("running time: {:.3f}".format(train_time))
    # 保存每个epoch的模型参数
    params_dict = model.get_parameters()
    with open('MiniVGG_Params_.json', 'w') as f:
        json.dump(params_dict, f)
