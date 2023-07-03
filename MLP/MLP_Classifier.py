import numpy as np
from MLP import MLP
from Core.Optimizer import Adam
from Core.Activation import Softmax
from Core.Loss import CrossEntropyWithSoftmax
from Plot_Classifier import plot_classifier


def train_epoch(model, optimizer, X, Y, batch_size):
    """训练一个epoch"""
    train_loss = 0
    for i in np.arange(0, len(X), batch_size):
        input = X[i:i + batch_size, :]
        truth = Y[i:i + batch_size, :]
        optimizer.zero_grad()
        output = model.forward(input)
        Loss = CrossEntropyWithSoftmax(model, truth, output)
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
    for i in range(len(X)):
        input = X[i, :].reshape(1, -1)
        truth = Y[i, :].reshape(1, -1)
        output = model.forward(input)
        Loss = CrossEntropyWithSoftmax(model, truth, output)
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


def train_model(model, X, Y):
    """训练模型"""
    batch_size = 16
    optimizer = Adam(model=model, learning_rate=0.05)
    for epoch in range(30):
        model, optimizer, train_loss = train_epoch(model, optimizer, X, Y, batch_size)
        accuracy = valid_model(model, X, Y)
        print("epoch: {:d}, loss: {:.3f}, accuracy: {:.3f}".format(epoch + 1, train_loss, accuracy))
        plot_classifier(model, X, Y, accuracy)
    return model


def valid_model(model, X, Y):
    input = X
    truth = Y
    output = model.forward(input)
    predict = np.argmax(output, axis=0)
    accuracy = np.array(predict == truth.flatten(), dtype=int).sum() / len(Y)
    return accuracy


def make_circles(n_samples=100, noise=None, factor=.8, shuffle=True):
    """创建同心圆随机数据"""
    if factor > 1 or factor < 0:
        raise ValueError("'factor' has to be between 0 and 1.")
    linspace = np.linspace(0, 2 * np.pi, n_samples // 2 + 1)[:-1]
    outer_circ_x = np.cos(linspace)
    outer_circ_y = np.sin(linspace)
    inner_circ_x = outer_circ_x * factor
    inner_circ_y = outer_circ_y * factor

    X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(n_samples // 2, dtype=np.intp),
                   np.ones(n_samples // 2, dtype=np.intp)])
    Y = y.reshape(-1, 1)
    if shuffle:
        random_index = np.arange(n_samples)
        np.random.shuffle(random_index)
        X = X[random_index]
        Y = Y[random_index]

    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)

    return X, Y


if __name__ == '__main__':
    # 获取同心圆状分布数据，X的每行包含两个特征，y是1/0类别标签
    X, Y = make_circles(600, noise=0.12, factor=0.2)
    Y = Y.reshape(-1, 1)
    # 创建模型
    model = MLP(2, 2, [3], out_act=Softmax)
    model = train_model(model, X, Y)
    accuracy = valid_model(model, X, Y)
    print("full dataset accuracy: {:.3f} %".format(accuracy * 100))
    plot_classifier(model, X, Y, accuracy, pause=False)
