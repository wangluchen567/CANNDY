# Loss——损失函数

## Basic-基础知识

损失函数（Loss Function）是机器学习和深度学习中用于衡量模型预测值与真实值之间差异的关键工具。它的核心作用是量化模型的预测误差，通过优化损失函数的值，可以调整模型的参数，从而提高模型的性能。

损失函数的选择取决于具体任务的性质。常见的任务主要分为两大类：分类问题和回归问题。对于分类问题，常用的损失函数包括交叉熵损失（Cross-Entropy Loss）、焦点损失（Focal Loss）等；而对于回归问题，常见的损失函数则有均方误差（Mean Squared Error, MSE）、平均绝对误差（Mean Absolute Error, MAE）等。不同的任务类型决定了损失函数的不同形式和优化目标。

损失函数的优化是训练过程的核心。通过反向传播算法，损失函数的梯度被计算出来，并用于更新模型的参数，以逐步降低损失值。因此，损失函数不仅决定了模型的优化方向，还直接影响模型的收敛速度和最终性能。

## MSE-均方误差损失

均方误差损失（Mean Squared Error, MSE）是回归问题中极为常见的损失函数。它通过计算预测值与真实值之间差的平方来量化模型的误差，其数学表达式为：

$$
Loss_{\text{MSE}}(p,t) = \frac{1}{2} (p - t)^2
$$
其中 $p$ 为预测值(predict)，$t$ 为真实值(truth)。

### 前向传播公式

上述公式即为均方误差损失的前向传播公式，用于衡量模型预测值与真实值之间的差异。

### 反向传播公式

在反向传播过程中，我们需要计算均方误差损失函数关于预测值 $p$ 的梯度。其导数表达式为：
$$
\frac{\mathrm{d}}{\mathrm{d}p} Loss_{\text{MSE}}(p,t) = p - t
$$

这个导数公式表明，均方误差损失的梯度是预测值与真实值之间的差值。在训练过程中，这个梯度被用来更新模型的参数，从而逐步减小预测值与真实值之间的差距。

## 交叉熵损失

交叉熵损失（Cross-Entropy Loss）是分类问题中非常常用的损失函数，尤其适用于多分类任务。它通过衡量模型预测的概率分布与真实标签的概率分布之间的差异来量化模型的误差。其数学表达式为：
$$
Loss_{\text{CE}}(p, t) = - \sum_{i=1}^K t_i~log(p_i)
$$
其中，$K$ 是类别总数，$p_i$ 是模型对第 $i$ 个类别的预测概率，$t_i$ 是真实标签的 one-hot 编码的第 $i$ 个分量。 具体来说，$t_i = 1$ 表示第 $i$ 个类别是真实类别，否则 $t_i=0$。

例如，对于一个三分类的问题，向量 $\mathbf{t} = [t_1, t_2, t_3]$ 表示某样本的 one-hot 编码，若该样本属于第一类，则其 one-hot 编码为$[1, 0, 0]$；若属于第二类，则为$[0, 1, 0]$；若属于第三类，则为$[0, 0, 1]$。也就是说，若样本属于第 $k$ 类，则其 one-hot 编码中 $t_k=1$，其余分量全部为 0.

模型对第 $i$ 个类别的预测概率 $p_i$ 通常通过 Softmax 函数实现。假设模型原始输出为 $a_i$ ，则 $p_i$ 表达式为：
$$
p_i = \text{Softmax}(a_i)=\frac{e^{a_i}}{\sum_{j=1}^k e^{a_j}}
$$
对于一个二分类问题，由于二分类本质是多分类的一种特殊情况，所以其交叉熵的损失可以简化为：
$$
Loss_{\text{CE}}(p, t) = - [t~log(p) + (1-t)~log(1-p)]
$$

### 前向传播公式

上述公式即为交叉熵损失的前向传播公式，它通过计算真实标签与预测概率之间的对数损失来衡量模型的性能。

### 反向传播公式

在处理交叉熵损失的反向传播时，通常会将 Softmax 函数的梯度和交叉熵损失的梯度进行整合，以简化计算。

由于 $t_i$ 表示为 one-hot 编码向量中的第 $i$ 个分量，若样本属于第 $i$ 类时 $t_i = 1$，且$t_j=0 ~ (j \neq i)$；若样本不属于第 $i$ 类时，$t_i = 0$，且其中其他某个 $t_j=1 ~ (j \neq i)$ ，计算交叉熵损失函数关于模型原始输出值的其中一个分量 $a_s$ 的梯度，有：
$$
\frac{\partial Loss_{\text{CE}}(p, t)}{\partial a_s}=-\sum_{i=1}^K \frac{t_i}{p_i} \cdot \frac{\partial p_i}{\partial a_s}
$$

其中，$\frac{\partial p_i}{\partial a_s}$ 的计算可以分为 $i = s$ 和 $i \neq s$ 两种情况计算：

- 当 $i = s$ 时：

$$
\begin{align}
\frac{\partial p_i}{\partial a_s} &= \frac{\partial p_s}{\partial a_s} = \frac{\partial}{\partial a_s} (\frac{e^{a_s}}{\sum_{j=1}^K e^{a_j}}) \\ &= \frac{e^{a_s} \cdot (\sum_{j=1}^K e^{a_j}) - (e^{a_s})^2}{(\sum_{j=1}^K e^{a_j})^2} \\ &= \frac{e^{a_s}}{\sum_{j=1}^K e^{a_j}} - (\frac{e^{a_s}}{\sum_{j=1}^K e^{a_j}})^2 \\ &= p_s \cdot (1-p_s)
\end{align}
$$

- 当 $i \neq s$ 时：

$$
\frac{\partial p_i}{\partial a_s} = \frac{\partial}{\partial a_s}(\frac{e^{a_i}}{\sum_{j=1}^K e^{a_j}})=-\frac{e^{a_i}\cdot e^{a_s}}{(\sum_{j=1}^K e^{a_j})^2} = -p_i \cdot p_s
$$

将两种情况代入有：

$$
\begin{align}
\frac{\partial Loss_{\text{CE}}(p, t)}{\partial a_s} &=-\sum_{i=1}^K \frac{t_i}{p_i} \cdot \frac{\partial p_i}{\partial a_s} \\&= -\frac{t_s}{p_s} \cdot p_s \cdot (1 - p_s) + \sum_{i \neq s} \frac{t_i}{p_i} \cdot p_i \cdot p_s \\&= -t_s + (t_s \cdot p_s + \sum_{i \neq s} t_i \cdot p_s) \\&= p_s \cdot (\sum_{i=1}^K t_i) - t_s \\&= p_s - t_s
\end{align}
$$

由此可以看出，交叉熵损失对模型原始输出的某个分量 $a_s$ 的偏导数是预测概率 $p_s$ 与标签 $t_s$ 的差。当 $t_s=1$ 时，说明样本属于该类别，$p_s - 1$ 为负值，更新会促使 $a_s$ 增大； 当 $t_s=0$时， 说明样本不属于该类别，$p_s - 0$ 为正值，则更新会促使 $a_s$ 减小。

