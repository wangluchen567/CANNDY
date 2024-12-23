import numpy as np
import matplotlib.pyplot as plt
from Core.Activation import ReLU, Sigmoid, Tanh

# 输入取值
x = np.linspace(-10, 10, 1000)
# 实例化激活函数
relu = ReLU()
sigmoid = Sigmoid()
tanh = Tanh()
# 绘制激活函数图像
plt.figure(dpi=120)
plt.plot(x, tanh(x), c='green', label='Tanh')
plt.plot(x, relu(x), c='red', label='Relu')
plt.plot(x, sigmoid(x), c='blue', label='Sigmoid')
plt.grid()  # 加入网格
plt.legend()  # 加入图例
# 保证x轴和y轴的单位长度相等
plt.gca().set_aspect('equal', adjustable='box')
# 限制坐标轴范围
plt.xlim([-6, 6])
plt.ylim([-2, 6])
# 设置标题
plt.title('Activations')
# 图像边缘紧缩
plt.tight_layout()
plt.show()




