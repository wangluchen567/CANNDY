import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CartPole:
    """CartPole环境"""

    def __init__(self):
        # 观测维度
        self.observation_dim = 4
        # 决策维度（动作维度）
        self.action_dim = 2
        # 物理参数（与Gym的CartPole-v1保持一致）
        self.gravity = 9.8  # 重力加速度 (m/s^2)
        self.mass_cart = 1.0  # 小车质量 (kg)
        self.mass_pole = 0.1  # 杆的质量 (kg)
        self.total_mass = self.mass_pole + self.mass_cart
        self.length = 0.5  # 杆的半长 (m)
        self.pole_mass_length = self.mass_pole * self.length
        self.force_mag = 10.0  # 施加的力大小 (N)
        self.tau = 0.02  # 时间步长 (s)
        # 终止条件阈值 (杆角度阈值 (12度))
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4  # 小车位置阈值 (m)
        # 状态变量
        self.state = None
        # 用于跟踪终止后的步骤
        self.steps_beyond_done = None
        # 可视化相关变量
        self.fig = None  # 图形窗口
        self.ax = None  # 坐标轴
        self.cart = None  # 小车图形对象
        self.pole = None  # 杆图形对象
        self.track = None  # 轨道图形对象
        # 最大步数限制（与Gym一致，200步）
        self.max_episode_steps = 200
        self.elapsed_steps = 0

    def reset(self):
        """重置环境到初始状态"""
        # 随机初始化状态，范围在[-0.05, 0.05]之间
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.elapsed_steps = 0
        self.steps_beyond_done = None
        # 关闭已有的图形窗口
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        """执行一个时间步的环境动态"""
        assert action in [0, 1], "Invalid action, only 0 or 1"
        # 解包当前状态
        x, x_dot, theta, theta_dot = self.state
        # 根据动作决定力的方向
        force = self.force_mag if action == 1 else -self.force_mag
        # 计算三角函数值（优化性能）
        cost_heta = np.cos(theta)
        sin_theta = np.sin(theta)
        # 物理计算（来自经典倒立摆动力学模型）
        temp = (force + self.pole_mass_length * theta_dot ** 2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cost_heta * temp) / (
                self.length * (4.0 / 3.0 - self.mass_pole * cost_heta ** 2 / self.total_mass))
        xacc = temp - self.pole_mass_length * theta_acc * cost_heta / self.total_mass
        # 使用欧拉方法更新状态
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        self.state = (x, x_dot, theta, theta_dot)
        self.elapsed_steps += 1
        # 检查终止条件
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.elapsed_steps >= self.max_episode_steps
        )
        # 计算奖励
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # 杆刚刚倒下
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                warnings.warn("You are calling 'step()' even though this environment has already returned done=True. "
                              "You should always call 'reset()' once you receive 'done=True' -- "
                              "any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def render(self, pause_time=0.01):
        """使用matplotlib渲染环境"""
        if self.fig is None:
            # 初始化图形窗口
            plt.ion()  # 开启交互模式
            self.fig, self.ax = plt.subplots(figsize=(6, 4.6))
            # 调整子图的边距
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            # 设置坐标轴范围
            self.ax.set_xlim(-3.0, 3.0)
            self.ax.set_ylim(-0.5, 3.6)
            self.ax.set_aspect('equal')
            # 隐藏坐标轴和网格
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_facecolor('white')  # 白色背景
            # 绘制轨道
            self.track = self.ax.plot(
                [-self.x_threshold * 1.5, self.x_threshold * 1.5],
                [0, 0],
                'k-', linewidth=4  # 加粗轨道线条
            )
            # 初始化小车
            cart_width = 0.5  # 小车宽度
            cart_height = 0.3  # 小车高度
            x = self.state[0]
            # 创建小车
            self.cart = patches.Rectangle(
                (x - cart_width / 2, -cart_height / 2),
                cart_width, cart_height,
                linewidth=2, edgecolor='k', facecolor='steelblue'
            )
            self.ax.add_patch(self.cart)
            # 初始化杆
            pole_length = self.length * 2  # 杆的全长
            pole_end_x = x + pole_length * np.sin(self.state[2])
            pole_end_y = pole_length * np.cos(self.state[2])
            self.pole, = self.ax.plot(
                [x, pole_end_x], [0, pole_end_y],
                color='goldenrod', linewidth=6
            )
            plt.title('CartPole')
            plt.tight_layout()
            plt.show()
        else:
            # 更新小车和杆的位置
            x = self.state[0]
            cart_width = 0.5
            # 更新小车位置
            self.cart.set_xy((x - cart_width / 2, -0.3 / 2))  # type: ignore
            pole_length = self.length * 2
            pole_end_x = x + pole_length * np.sin(self.state[2])
            pole_end_y = pole_length * np.cos(self.state[2])
            # 更新杆的位置
            self.pole.set_data([x, pole_end_x], [0, pole_end_y])
            # 重绘图形
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            # 小延迟使动画更流畅
            plt.pause(pause_time)

    def close(self):
        """关闭图形窗口，清理资源"""
        if self.fig is not None:
            plt.close('all')  # 关闭所有图形窗口
            plt.ioff()  # 关闭交互模式
            self.fig = None


if __name__ == "__main__":
    env = CartPole()
    state = env.reset()
    done = False
    total_reward = 0

    # 运行最多200步（与Gym默认设置一致）
    for _ in range(200):
        # 简单启发式策略
        action = 0 if state[2] < 0 else 1

        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

        if done:
            break

    print(f"回合结束，总奖励: {total_reward}")
    env.close()
