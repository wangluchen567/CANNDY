# CANNDY
Language：[English](README_EN.md) | [中文](README.md)

## 项目名称
基于NumPy构建的人工神经网络框架<br>
Chen's Artificial Neural Network constructeD with numpY<br>

## 项目简介
本项目致力于为对神经网络和人工智能感兴趣的伙伴们提供一个学习和实践的平台。
本项目不旨在构建大规模模型或直接用于生产环境的模型，
但力求通过实现一系列经典的神经网络模型，深入探讨神经网络的底层原理。
本项目提供了详尽的实现步骤，以帮助学习者深入理解神经网络的构建过程与内部工作的机制。
此外，本项目采用的不是基于计算图的构建方式，而是通过公式推导构建的层级架构，以便更直观地剖析神经网络的工作机制。
希望本项目能够帮助感兴趣的伙伴们建立起对神经网络的深刻理解，从而在人工智能领域的进一步学习和研究打下坚实的基础。

**特别说明：`本代码仅供参考学习、竞赛和科学研究等非商业用途，在复制核心代码时请注明出处`**

## 安装教程
**1. 建议使用 `Anaconda` 创建 `Python` 环境**

  使用 Anaconda 创建环境可以方便地管理依赖包，避免版本冲突。建议从 [Anaconda 官网](https://www.anaconda.com/download/success) 下载并安装 Anaconda。如果需要特定版本，可以访问 [Anaconda所有版本下载地址](https://repo.anaconda.com/archive/)。

  安装完成后，运行以下命令创建 Python 环境：
  ```bash
  conda create --name my_env python=3.9
  conda activate my_env
  ```
  **注意**：本项目支持 Python 3.7 及以上版本，建议使用 Python 3.9 以获得最佳兼容性。请确保已安装 Python 3.7 或更高版本。

**2. 安装必要包**

  本项目依赖以下包:`numpy`、`matplotlib`、`tqdm`。请确保已安装 Python 3.7 或更高版本，运行以下命令一键安装必要包：
  
  ```bash
  pip install numpy matplotlib tqdm
  ```

**3. 安装可选包**

  在使用本项目中强化学习模型训练CartPole环境时需要安装游戏环境相关包。运行以下命令一键安装：
  
  ```bash
  pip install gym==0.22.0 pygame==2.2.0 pyglet==1.5.27
  ```
  
  在使用`gym`时可能会遇到如下报错:
  
  ```
  ImportError: cannot import name 'rendering' from 'gym.envs.classic_control'
  ```
  
  若遇到该问题请将rendering.py放到..\Anaconda\Lib\site-packages\gym\envs\classic_control\目录下

  参考网址：https://blog.csdn.net/qq_34666857/article/details/123551558

**4. 镜像源选择**

  如果在运行安装命令时发现下载速度较慢，可以尝试使用清华大学的镜像源进行安装。安装命令如下：
  ```bash
  pip install numpy matplotlib tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
  注意：如果无法访问上述镜像源，也可以选择其他可用的镜像源，例如中国科技大学、阿里云等。


## 核心实现

- **Activation: 激活函数**
  - ReLU/Sigmoid/Tanh/Softmax

- **Layers: 神经网络层**
  - Linear: 线性层(全连接单层)
  - Indentity: 恒等变换层
  - Dropout: 随机失活层
  - GCNConv: 图卷积层
  - RNNCell: 循环神经网络模块
  - RNN: 循环神经网络层
  - Flatten: 展平层
  - Conv1d: 一维卷积层
  - Conv2d: 二维卷积层
  - MaxPool1d: 一维最大池化层
  - MaxPool2d: 二维最大池化层
  - MeanPool1d: 一维平均池化层
  - MeanPool2d: 二维平均池化层
  - BatchNorm: 批归一化层
  - BatchNorm2d: 二维批归一化层
  - ReLULayer: ReLU激活层
  - SigmoidLayer: Sigmoid激活层
  - TanhLayer: Tanh激活层
  - SoftmaxLayer: Softmax激活层

- **Loss: 损失函数**
  - MSELoss: 均方误差损失
  - CrossEntropyWithSoftmax: 带Softmax的交叉熵损失
  - CrossEntropyWithSoftmaxMask: 带Softmax和Mask的交交叉熵损失

- **Module: 神经网络模型**
  - MLP: 全连接神经网络模型(多层感知机)
  - GCN: 图卷积神经网络模型
  - RNNModel: 循环神经网络模型
  - CNNTimeSeries: 一维卷积神经网络模型
  - LeNet5: LeNet-5卷积神经网络模型

- **Optimizer: 优化器(优化算法)**
  - GD/Momentum/AdaGrad/RMSProp/Adam

## 框架结构
```
CANNDY/
├── Core/                               # 人工神经网络框架核心
│   ├── Activation.py                   # 实现各种激活函数
│   ├── Layers.py                       # 实现各种神经网络层
│   ├── Loss.py                         # 实现各种损失函数
│   ├── Module.py                       # 实现各种神经网络模型
│   └── Optimizer.py                    # 实现各种优化器
├── Datasets/                           # 数据集
│   ├── Mnist_Test/                     # 手写数字识别数据集(单独生成)
│   ├── cora.tar.gz                     # 图神经网络数据集Cora
│   ├── iris.csv                        # 鸢尾花数据集
│   ├── mnist.pkl.gz                    # 手写数字识别数据集
│   └── titanic.csv                     # 泰坦尼克幸存者数据集
├── Examples/                           # 使用该框架实现的各种具体实例模型
│   ├── AutoEncoder/                    # 自编码器模型
│   │   ├── AutoEncoder.py              # 自编码器模型
│   │   └── train_AutoEncoder.py        # 运行自编码器模型实例
│   ├── DQN/                            # 深度Q-学习网络模型
│   │   ├── agent.py                    # DQN智能体
│   │   ├── algorithm.py                # DQN算法
│   │   ├── DQNLoss.py                  # DQN损失函数
│   │   ├── replay_memory.py            # 经验回放池
│   │   ├── train_DQN.py                # 训练DQN玩CartPole
│   │   └── train_DQN_Snake.py          # 训练DQN玩贪吃蛇
│   ├── PolicyGradient/                 # 策略梯度网络模型
│   │   ├── agent.py                    # PG智能体
│   │   ├── algorithm.py                # PG算法
│   │   ├── PGLoss.py                   # PG损失函数
│   │   ├── replay_memory.py            # 经验回放池
│   │   ├── train_PG.py                 # 训练PG玩CartPole
│   │   └── train_PG_Snake.py           # 训练PG玩贪吃蛇
│   ├── RL_Envs/                        # 强化学习环境(单独实现)
│   │   └── Snake.py                    # 贪吃蛇游戏环境
├── Models/                             # 使用该框架实现的几种经典的模型
│   ├── CNN/                            # 卷积神经网络
│   │   ├── CNN_Mnist.py                # 卷积神经网络学习手写数字识别分类
│   │   ├── CNN_Series.py               # 卷积神经网络学习周期数据回归
│   │   ├── CNN_Visualize.py            # 卷积神经网络学习手写数字识别结果可视化
│   │   ├── LeNet-5_Params.json         # 卷积神经网络学习手写数字识别所得参数
│   │   └── Plot_Contrast.py            # 卷积神经网络不同参数效果对比绘图
│   ├── GCN/                            # 图卷积神经网络
│   │   └── GCN_Cora.py                 # 图卷积神经网络学习Cora数据
│   └── MLP/                            # 全连接神经网络(多层感知机)
│   │   ├── MLP_Batch.py                # 全连接神经网络学习批数据测试
│   │   ├── MLP_Classifier.py           # 全连接神经网络简单分类测试
│   │   ├── MLP_Iris.py                 # 全连接神经网络对鸢尾花数据集分类测试
│   │   ├── MLP_Mnist.py                # 全连接神经网络学习手写数字识别分类
│   │   ├── MLP_Regress.py              # 全连接神经网络学习简单数据回归
│   │   └── Plot_Classifier.py          # 对分类结果绘图函数
│   ├── RNN/                            # 循环神经网络
│   │   └── RNN_Predict.py              # 循环神经网络学习周期数据回归
├── Notes/                              # 框架实现的细节笔记
├── Pics/                               # 框架实现的模型运行效果图
└── README.md                           # 项目文档
```

## 更新计划

- [x] 更新项目文档
- [ ] 更新算法笔记
- [ ] 尝试实现更复杂的模型

## 效果展示

- MLP模型对同心圆数据集分类效果/对sin函数回归拟合效果

    <img src="Pics/MLP_circle.gif" width="288" height="220" alt="分类训练"/>
    <img src="Pics/MLP_sin.gif" width="288" height="220" alt="回归拟合"/>
  
- 循环神经网络模型与一维卷积时间序列模型对sin函数未来趋势预测效果

    <img src="Pics/RNN_predict.gif" width="288" height="220" alt="RNN预测"/>
    <img src="Pics/CNN_Series.gif" width="288" height="220" alt="CNNT预测"/>
  
- 卷积神经网络LeNet-5模型手写数字的识别效果(测试集准确率超过99%)

1. 训练损失与测试准确率表现及不同trick下的模型表现

    <img src="Pics/CNN_train.png" width="288" height="230"/>
    <img src="Pics/Trick_contrast.png" width="288" height="230"/>
   
2. 输入图像与输出预测结果及概率

    <img src="Pics/Input.png" width="288" height="250"/>
    <img src="Pics/Result.png" width="288" height="250"/>
   
3. 第1层卷积层和第1层池化层得到的特征图

    <img src="Pics/Conv1.png" width="288" height="196"/>
    <img src="Pics/MaxPool1.png" width="288" height="196"/>
   
4. 第2层卷积层和第2层池化层得到的特征图

    <img src="Pics/Conv2.png" width="288" height="288"/>
    <img src="Pics/MaxPool2.png" width="288" height="288"/>
- 自编码器模型效果

    <img src="Pics/Res_AE.gif" width="396" height="220" alt="自编码器"/>

- DQN(深度Q学习网络)模型训练玩贪吃蛇最终效果

    <img src="Pics/Snake.gif" width="396" height="360" alt="后期效果"/>

- PG(策略梯度网络)模型训练玩CartPole最终效果

    <img src="Pics/PG_end.gif" width="396" height="260" alt="后期效果"/>

## 项目贡献与支持

**Author: Luchen Wang**<br>
<small>（如在使用过程中遇到任何问题，可随时联系邮箱：wangluchen567@qq.com）</small>


