##  📦 项目结构

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
│   │   ├── train_DQN_CartPole.py       # 训练DQN玩CartPole
│   │   └── train_DQN_Snake.py          # 训练DQN玩贪吃蛇
│   ├── PolicyGradient/                 # 策略梯度网络模型
│   │   ├── agent.py                    # PG智能体
│   │   ├── algorithm.py                # PG算法
│   │   ├── PGLoss.py                   # PG损失函数
│   │   ├── replay_memory.py            # 经验回放池
│   │   ├── train_PG_CartPole.py        # 训练PG玩CartPole
│   │   └── train_PG_Snake.py           # 训练PG玩贪吃蛇
│   └── RL_Envs/                        # 强化学习环境(单独实现)
│       ├── CartPole.py                 # CartPole环境
│       └── Snake.py                    # 贪吃蛇游戏环境
├── Models/                             # 使用该框架实现的几种经典的模型
│   ├── CNN/                            # 卷积神经网络
│   │   ├── CNN_Mnist.py                # 卷积神经网络学习手写数字识别分类
│   │   ├── CNN_Series.py               # 卷积神经网络学习周期数据回归
│   │   ├── CNN_Visualize.py            # 卷积神经网络学习手写数字识别结果可视化
│   │   ├── LeNet-5_Params.json         # 卷积神经网络学习手写数字识别所得参数
│   │   └── Plot_Contrast.py            # 卷积神经网络不同参数效果对比绘图
│   ├── GCN/                            # 图卷积神经网络
│   │   └── GCN_Cora.py                 # 图卷积神经网络学习Cora数据
│   ├── MLP/                            # 全连接神经网络(多层感知机)
│   │   ├── MLP_Batch.py                # 全连接神经网络学习批数据测试
│   │   ├── MLP_Classifier.py           # 全连接神经网络简单分类测试
│   │   ├── MLP_Iris.py                 # 全连接神经网络对鸢尾花数据集分类测试
│   │   ├── MLP_Mnist.py                # 全连接神经网络学习手写数字识别分类
│   │   ├── MLP_Regress.py              # 全连接神经网络学习简单数据回归
│   │   └── Plot_Classifier.py          # 对分类结果绘图函数
│   └── RNN/                            # 循环神经网络
│       └── RNN_Predict.py              # 循环神经网络学习周期数据回归
├── Notes/                              # 框架实现的细节笔记
├── Pics/                               # 框架实现的模型运行效果图
└── README.md                           # 项目文档
```