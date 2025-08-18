# Optimizer——梯度优化器

梯度优化器是神经网络、机器学习和深度学习模型训练中的关键组件，用于**自动调整模型参数**（如权重和偏置），以最小化损失函数（即模型预测误差）。其本质是通过计算损失函数对参数的梯度（导数），沿梯度反方向迭代更新参数，逐步逼近最优解。

## SGD

### 核心思想

SGD（*Stochastic Gradient Decent*）是最基础的梯度优化器，通过**每次迭代随机选取一个样本（或一个小批量样本）**计算损失函数的梯度，并沿负梯度方向更新模型参数。其核心目标是逐步降低损失函数值，找到局部最优解（或全局最优解）。

### 数学原理

其参数更新规则为：
$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$
其中 $\theta_t$ 表示第 $t$ 次迭代的参数，$\eta$ 表示学习率（步长），$g_t$ 表示损失函数对参数的梯度。

### 变体形式

SGD有两种极端的变体情况，第一种是全量的 **Batch GD**，它使用全部数据计算梯度，这样计算梯度的方向基本一致，但计算开销很大，并且对学习率比较敏感。另一种是每次随机选取一个样本的计算梯度，这种方式计算量虽小，但梯度方向可能较为混乱，导致收敛比较缓慢。因此，比较常用的是 **Mini-batch SGD**，即每次选取一小批（batch）的样本计算梯度，它能平衡效率和稳定性。

### 核心特性

- 优点：实现简单，内存占用小，并且其收敛性在凸函数下有严格的数学证明，适用于简单的模型与小规模数据集。
- 缺点：需要手动调参，对学习率比较敏感，可能会存在收敛缓慢或者剧烈震荡的情况，并且容易陷入局部最优或者鞍点。

## Momentum

### 核心思想

Momentum优化器通过引入**“动量”**（历史梯度的指数加权平均）来加速SGD的收敛，并减少参数更新过程中的震荡。其灵感来源于物理学中的动量概念：梯度更新方向不仅依赖当前梯度，还会保留之前更新方向的惯性。

### 数学原理

- **动量累积**
  $$
  v_t = \gamma ~ v_{t-1} + \eta \cdot g_t
  $$

- **参数更新**
  $$
  \theta_{t+1} = \theta_t - v_t
  $$
  其中 $v_t$ 表示当前动量（累积梯度方向），$\gamma$ 表示动量系数（取值通常为0.9或0.99，以控制历史梯度的保留程度），$\theta_t$ 表示第 $t$ 次迭代的参数，$\eta$ 表示学习率（步长），$g_t$ 表示损失函数对参数的梯度。

### 核心特性

- 优点：相比于SGD，可以加速收敛并减少震荡，使收敛路径更平滑。
- 缺点：仍然需要调参，对参数仍然敏感，若动量过大可能会导致在最优解附近震荡甚至发散。

## AdaGrad

### 核心思想

AdaGrad（*Adaptive Gradient*）是首个广泛使用的**自适应学习率优化器**，其核心思想是**为每个参数分配独立的学习率**：对于频繁更新的参数（梯度大），降低其学习率；对于稀疏更新的参数（梯度小），保持较大的学习率。这一特性使其特别适合处理稀疏数据（如NLP中的词嵌入）。

### 数学原理

- **梯度平方累积**
  $$
  s_t = s_{t-1} + g_t^2
  $$

- **参数更新**
  $$
  \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t+\epsilon} } ~ g_t
  $$
  其中 $\eta$ 表示全局初始学习率，$\epsilon$ 表示平滑项（通常设置为1e-8，防止除零），$\theta_t$ 表示第 $t$ 次迭代的参数，$g_t$ 表示损失函数对参数的梯度。

### 核心特性

- 优点：自适应学习率，调参简单，比较适合稀疏数据，可以为大规模的参数分配不同的学习率。
- 缺点：内存占用较大；学习率递减，可能会导致过早衰减；对非凸问题可能会收敛到次优解。

## RMSPorp

### 核心思想

**RMSProp**（*Root Mean Square Propagation*）是针对AdaGrad的缺陷提出的改进算法，通过引入**指数加权移动平均**（*Exponentially Weighted Moving Average*）替代AdaGrad的梯度平方累加，解决其学习率过早衰减至零的问题。核心思想是：

- 短期记忆：更关注最近梯度的规模，而非所有历史梯度。
- 自适应学习率：为每个参数动态调整学习率，适合非平稳目标（如神经网络的损失函数）。

### 数学原理

- **梯度平方的指数平均**
  $$
  s_t = \beta s_{t-1} + (1-\beta) g_t^2
  $$
  （$\beta$ 为衰减率，通常取值为 0.9 或0.99）

- **参数更新**
  $$
  \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t+\epsilon} } ~ g_t
  $$
  其中 $\eta$ 表示全局初始学习率，$\epsilon$ 表示平滑项（通常设置为1e-8，防止除零），$\theta_t$ 表示第 $t$ 次迭代的参数，$g_t$ 表示损失函数对参数的梯度。

### 核心特性

- 优点：基于AdaGrad改进，解决了学习率过早衰减的情况，能一定程度上抑制震荡并加速收敛。
- 缺点：学习率和衰减率仍然需要根据任务调整，并且可能会收敛到尖锐的极小值导致泛化性不如SGD。

## Adam

### 核心思想

Adam（*Adaptive Moment Estimation*）是当前深度学习中最流行的优化器之一，它结合了**动量法（*Momentum*）**和**自适应学习率（*RMSProp*）**的优点：

- **动量机制**：跟踪梯度的**一阶矩（均值）**，加速收敛并减少震荡。
- **自适应学习率**：跟踪梯度的**二阶矩（方差）**，为不同参数分配不同的学习率。

这使得Adam在**训练深度神经网络**时表现优异，尤其适用于：

- 大规模数据集（如ImageNet）
- 高维参数空间（如Transformer）
- 非凸优化问题（如GAN训练）

### 数学原理

- **更新一阶矩（动量）**
  $$
  v_t = \beta_1 v_{t-1} + (1 - \beta_1)g_t
  $$
  $v_t$ 表示当前动量（类似Momentum）,$\beta_1$ 用于控制历史梯度的衰减率（默认0.9）

- **更新二阶矩（自适应学习率）**
  $$
  s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2
  $$
  $s_t$ 表示梯度平方的指数移动平均（类似RMSProp），$\beta_2$ 用于控制历史梯度平方的衰减率（默认0.999）

- **偏差修正**

  由于初始的 $v_0 = 0,~ s_0=0$，导致训练早期的矩估计会偏小，因此需要进行修正：
  $$
  \hat{v_t} = \frac{v_t}{1-\beta_1^t}, \quad \hat{s_t} = \frac{s_t}{1-\beta_2^t}
  $$

- **参数更新**
  $$
  \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{s_t} + \epsilon}}~\hat{v_t}
  $$
  其中 $\eta$ 表示全局初始学习率（默认0.001），$\epsilon$ 表示平滑项（通常设置为1e-8，防止除零），$\theta_t$ 表示第 $t$ 次迭代的参数，$g_t$ 表示损失函数对参数的梯度。

### 核心特性

**Adam 是目前最常用的优化器**，能适用于绝大多数深度学习任务。但需注意：

- **训练初期**：由于偏差修正，更新幅度较小。
- **训练后期**：可能不如SGD泛化性好（可尝试切换至SGD微调）。

## AMSGrad

**AMSGrad**（*Adam with Maximum Squared Gradients*）是Adam的一种改进变体，主要针对 **Adam** 在某些任务中可能出现的不收敛问题进行了优化。其核心改进在于**对二阶矩的计算方式进行了修改和调整**。

在**Adam**中，二阶矩的计算直接使用指数移动平均的形式：
$$
s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2
$$
而在**AMSGrad**中，则强调二阶矩单调不减：
$$
s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2\\
\hat{s_t} = \text{max}(\hat{s}_{t-1}, s_t)
$$
也就是说，AMSGrad维护了一个二阶矩的最大值，每次参数更新使用的是这个最大值进行更新。这样可以确保自适应学习率的分母（$\sqrt{\hat{s}_t+\epsilon}$）不会因为梯度变小而突然下降，从而避免Adam在某些情况下因为学习率过大而发散。并且理论上证明AMSGrad对凸优化问题有更强的保证，虽然在实际任务中一般差异较小，但在某些极端情况下表现更稳定。

## AdamW

在介绍 **AdamW** 之前，我们需要先了解 **权重衰减** 这一概念。权重衰减是一种正则化技术，通过在损失函数中添加 L2 惩罚项来防止模型过拟合，其表达式为：
$$
Loss_{L2} = Loss + \lambda \|\theta\|_2^2
$$
对于传统的 **SGD**，可以在梯度更新前直接加入衰减项，具体操作如下：
$$
\hat{g}_t = g_t + \lambda \theta_t\\
\theta_{t+1} = \theta_{t} - \eta \cdot \hat{g}_t
$$
即：
$$
\theta_{t+1} = \theta_{t} - \eta \cdot g_t - \eta \lambda \theta_t
$$
其中，$\lambda$ 是权重衰减系数。

然而，对于 **Adam** 优化器，若直接使用权重衰减，其计算公式会变为：
$$
v_t = \beta_1 v_{t-1} + (1 - \beta_1)(g_t + \lambda \theta_t)\\
s_t = \beta_2 s_{t-1} + (1-\beta_2) (g_t + \lambda \theta_t)^2\\
\hat{v_t} = \frac{v_t}{1-\beta_1^t}, \quad \hat{s_t} = \frac{s_t}{1-\beta_2^t}\\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{s_t} + \epsilon}}~\hat{v_t}
$$
这种操作会导致 **衰减效果被自适应学习率缩放**，从而干扰正则化的效果。

为了解决这一问题，Ilya 和 Frank 提出了 **AdamW**，**AdamW**将权重衰减**独立于自适应学习率**，直接作用于参数，将两者解耦，从而避免自适应学习率对正则化的干扰。具体来说，只需要对 **Adam** 的最后一步进行如下修改：
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{s_t} + \epsilon}}~\hat{v_t} - \eta \lambda \theta_t
$$
因此，**AdamW** 可以看作是 **Adam** 的一个 **微小改进版本**，它成功解决了权重衰减与自适应学习率之间的耦合问题。其核心意义在于：

- **理论正确性**：真正还原了 L2 正则化的本质。
- **实践价值**：在大规模模型训练中得到了广泛应用，例如在 BERT 和 GPT 等模型的训练中，**AdamW** 都发挥了重要作用。

## 参考文献

[1] [Optimization methods for large-scale machine learning](https://arxiv.org/pdf/1606.04838)

[2] [Understanding the role of momentum in stochastic gradient methods](https://proceedings.neurips.cc/paper/2019/hash/4eff0720836a198b6174eecf02cbfdbf-Abstract.html)

[3] [Adaptive subgradient methods for online learning and stochastic optimization](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

[4] RMSProp: Divide the Gradient by a Running Average of Its Recent Magnitude

[5] [Adam: A method for stochastic optimization](https://arxiv.org/abs/1412.6980)

[6] [On the convergence of adam and beyond](https://arxiv.org/abs/1904.09237)

[7] [Decoupled weight decay regularization](https://arxiv.org/abs/1711.05101)