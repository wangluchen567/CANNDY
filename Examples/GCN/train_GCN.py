import gzip
import time
import numpy as np
import matplotlib.pyplot as plt

from GCN import GCN
from Core.Optimizer import Adam
from Core.Activation import Softmax
from Core.Loss import CrossEntropyWithSoftmaxMask

def load_data(data_path):
    print("loading data...")
    i = -1
    data = [[], [], [], []]
    with gzip.open(data_path, "rt") as f:
        lines = f.readlines()
        for l in lines:
            if '\x00' in l:
                i += 1
                d = l.split('\x00')[-1]
                d = d.strip()
                if len(d):
                    data[i].append(np.array(d.split(',')))
            else:
                data[i].append(np.array(l.split(',')))
    features = np.array(data[0], dtype=float)
    indices = np.array(data[1], dtype=int)
    labels = np.array(data[2], dtype=int)
    masks = np.array(data[3], dtype=bool)
    return features, indices, labels, masks

def format_data(features, indices, labels, masks, self_loop=True):
    # 对特征进行归一化
    features = features / features.sum(1).reshape(-1, 1)
    # 获取邻接矩阵
    adj_mat = np.zeros((features.shape[0], features.shape[0]))
    adj_mat[indices[0, :], indices[1, :]] = 1
    # 获取训练、验证和测试的mask矩阵
    train_mask = masks[0, :]
    val_mask = masks[1, :]
    test_mask = masks[2, :]
    # 每个节点加入自环边
    if self_loop:
        adj_mat = adj_mat + np.eye(adj_mat.shape[0])
    return features, labels, adj_mat, train_mask, val_mask, test_mask

def evaluate(model, features, labels, mask):
    """评价函数"""
    output = model.forward(features)  # 将特征输入模型查看结果
    output = output.T[mask]  # 获取某类数据的结果（train/val/test）
    labels = labels[mask]  # 获取某类数据的真实值
    indices = np.argmax(output, axis=1)  # 取结果中最大值为预测值
    correct = np.sum(indices == labels.flatten())  # 获取准确的个数
    # 返回准确率
    return correct.item() * 1.0 / len(labels)


def plot_train(loss, acc):
    # 画图显示loss和acc的变化
    plt.figure(0)
    plt.title("Loss")
    epoch = list(range(1, len(loss) + 1))
    plt.plot(epoch, loss, color='red', label="Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.figure(1)
    plt.title("Accuracy")
    plt.plot(epoch, acc, color='blue', label="Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()



if __name__ == '__main__':
    data_path = "../../Dataset/cora.tar.gz"

    features, indices, labels, masks = load_data(data_path)
    features, labels, adj_mat, train_mask, val_mask, test_mask = format_data(features, indices, labels, masks,
                                                                             self_loop=True)
    model = GCN(adj_mat=adj_mat,
                input_size=features.shape[1],
                output_size=np.max(labels) + 1,
                hidden_sizes=[16],
                out_act=Softmax)


    optimizer = Adam(model=model, learning_rate=0.001)

    num_epochs = 200
    # 训练过程
    dur = []  # 记录epoch时间
    train_loss = []  # 记录训练损失变化
    train_acc = []  # 记录训练准确率变化
    for epoch in range(num_epochs):
        t0 = time.time()
        # 梯度归零
        optimizer.zero_grad()
        # forward前向传播，使用交叉熵损失
        output = model.forward(features)
        Loss = CrossEntropyWithSoftmaxMask(model, labels, output, train_mask)
        # Loss = CrossEntropyWithSoftmax(model, labels, output)
        ces_loss = Loss.forward()
        Loss.backward()
        optimizer.step()
        # 保存训练时间
        dur.append(time.time() - t0)
        # 计算训练过程中验证集的准确率
        acc = evaluate(model, features, labels, val_mask)
        # 保存训练损失和准确率
        train_loss.append(ces_loss)
        train_acc.append(acc)
        # 打印相关信息
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f}"
              .format(epoch + 1, np.mean(dur), ces_loss, acc))

    print()
    # 计算测试集准确率
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))
    plot_train(train_loss, train_acc)

