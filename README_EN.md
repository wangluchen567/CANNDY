# 🍬 CANNDY
💬 Language：[English](README_EN.md) | [中文](README.md)

Chen's Artificial Neural Network constructeD with numpY

## 🔖 Project Introduction
This project is dedicated to providing a learning and practice platform for those interested in neural networks and artificial intelligence. Although it is not intended to build large-scale models or models directly for production environments, it strives to delve into the underlying principles of neural networks by implementing a series of classic neural network models. The project offers detailed implementation steps to help learners gain a deep understanding of the construction process and the internal workings of neural networks.

To lower the barrier to entry, this project has adopted the model building and training style of PyTorch in its design, but focuses on the implementation of core functionalities, resulting in a relatively streamlined feature set. Unlike mature industrial-level frameworks, this project places greater emphasis on the transparency and readability of algorithm implementation, revealing the working principles of neural networks and deep learning models through handwritten underlying code. It should be noted that this project is more suitable for learning and experimental purposes. For production environments, it is recommended to use more feature-rich professional frameworks such as [PyTorch](https://github.com/pytorch/pytorch).

Additionally, this project does not adopt a computational graph-based approach but instead constructs a hierarchical architecture through formula derivation, allowing for a more intuitive analysis of the neural network's working mechanism. We hope that this project will help interested individuals build a profound understanding of neural networks, thereby laying a solid foundation for further study and research in the field of artificial intelligence.

**Special Note: This code is for reference only in non-commercial purposes such as learning, competition, and scientific research. Please indicate the source when copying the core code.**

## 📚 Installation Guide

**1. It is recommended to use `Anaconda` to create a `Python` environment**

  Creating an environment with Anaconda allows for convenient management of dependencies and avoids version conflicts. It is recommended to download and install Anaconda from the [Anaconda official website](https://www.anaconda.com/download/success). If a specific version is needed, you can visit the [Anaconda archive download page](https://repo.anaconda.com/archive/).

  After installation, run the following commands to create a Python environment:

  ```bash
  conda create --name my_env python=3.9
  conda activate my_env
  ```

  **Note**: This project supports Python 3.7 and above. It is recommended to use Python 3.9 for the best compatibility. <br>

**2. Install necessary packages**

  This project depends on the following packages: `numpy`, `matplotlib`, `tqdm`. Ensure that you have Python 3.7 or a higher version installed, and then run the following command to install the necessary packages:

  ```bash
  pip install numpy matplotlib tqdm
  ```

## 🎯 Core Components

- **Activation: Activation Functions**
  - ReLU/Sigmoid/Tanh/Softmax

- **Layers: Neural Network Layers**
  - Linear: Linear layer (fully connected single layer)
  - Identity: Identity transformation layer
  - Dropout: Dropout layer (random deactivation)
  - GCNConv: Graph convolutional layer
  - RNNCell: Recurrent neural network cell
  - RNN: Recurrent neural network layer
  - Flatten: Flattening layer
  - Conv1d: 1D convolutional layer
  - Conv2d: 2D convolutional layer
  - MaxPool1d: 1D max pooling layer
  - MaxPool2d: 2D max pooling layer
  - MeanPool1d: 1D average pooling layer
  - MeanPool2d: 2D average pooling layer
  - BatchNorm: Batch normalization layer
  - BatchNorm2d: 2D batch normalization layer
  - ReLULayer: ReLU activation layer
  - SigmoidLayer: Sigmoid activation layer
  - TanhLayer: Tanh activation layer
  - SoftmaxLayer: Softmax activation layer

- **Loss: Loss Functions**
  - MSELoss: Mean squared error loss
  - CrossEntropyWithSoftmax: Cross-entropy loss with Softmax
  - CrossEntropyWithSoftmaxMask: Cross-entropy loss with Softmax and Mask

- **Module: Neural Network Models**
  - MLP: Multilayer perceptron (fully connected neural network model)
  - GCN: Graph convolutional neural network model
  - RNNModel: Recurrent neural network model
  - CNNTimeSeries: 1D convolutional neural network model
  - LeNet5: LeNet-5 convolutional neural network model

- **Optimizer: Optimizers (Optimization Algorithms)**
  - GD/Momentum/AdaGrad/RMSProp/Adam

## 📝 Update Plan

- [x] Update project documentation
- [ ] Update algorithm notes
- [ ] Attempt to implement more complex models

## 🌈 Effect Demonstration

- Classification effect of the MLP model on concentric circle dataset / Regression fitting effect on the sin function

    <img src="Pics/MLP_circle.gif" width="288" height="220" alt="Classification Training"/>
    <img src="Pics/MLP_sin.gif" width="288" height="220" alt="Regression Fitting"/>
  
- Prediction effect of the RNN model and the 1D convolutional time series model on the future trend of the sin function

    <img src="Pics/RNN_predict.gif" width="288" height="220" alt="RNN Prediction"/>
    <img src="Pics/CNN_Series.gif" width="288" height="220" alt="CNN Prediction"/>
  
- Recognition effect of the Convolutional Neural Network LeNet-5 model on handwritten digits (Test set accuracy exceeds 99%)

1. Training loss and test accuracy performance, as well as model performance under different tricks

    <img src="Pics/CNN_train.png" width="288" height="230"/>
    <img src="Pics/Trick_contrast.png" width="288" height="230"/>
   
2. Input images and output prediction results with probabilities

    <img src="Pics/Input.png" width="288" height="250"/>
    <img src="Pics/Result.png" width="288" height="250"/>
   
3. Feature maps obtained from the first convolutional layer and the first pooling layer

    <img src="Pics/Conv1.png" width="288" height="196"/>
    <img src="Pics/MaxPool1.png" width="288" height="196"/>
   
4. Feature maps obtained from the second convolutional layer and the second pooling layer

    <img src="Pics/Conv2.png" width="288" height="288"/>
    <img src="Pics/MaxPool2.png" width="288" height="288"/>
- Effect of the Autoencoder model

    <img src="Pics/Res_AE.gif" width="396" height="220" alt="Autoencoder"/>

- Final effect of the DQN (Deep Q-Learning Network) model trained to play Snake

    <img src="Pics/Snake.gif" width="396" height="360" alt="Final Effect"/>

- Final effect of the PG (Policy Gradient Network) model trained to play CartPole

    <img src="Pics/PG_end.gif" width="396" height="260" alt="Final Effect"/>

## 🤝 Contributions

**Author: Luchen Wang**

## ✉️ Contact Us

**E-mail: wangluchen567@qq.com**