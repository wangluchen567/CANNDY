import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Snake:
    def __init__(self):
        # 观测维度
        self.observation_dim = None
        # 决策维度（动作维度）
        self.action_dim = 4  # 上下左右不操作五种
        # 当前状态
        self.state = None
        # 蛇的状态
        self.snake = None
        # 苹果的位置
        self.apples = None
        # 苹果的数量
        self.num_apples = None
        # 当前运动方向
        self.direction = None
        # 运动方向编码
        self.direction_code = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=int)
        # 地图大小
        self.map_size = None
        # 初始化地图
        self.map = None
        # 初始化无动作步数
        self.no_act_step = 0
        # 初始化分数
        self.score = 0
        # 重置环境
        self.reset()

    def reset(self):
        """重置环境"""
        self.score, self.no_act_step = 0, 0
        self.direction = np.array([0, 1], dtype=int)
        self.map_size = 16
        self.num_apples = 1
        # 初始化地图
        self.map = np.zeros((self.map_size, self.map_size), dtype=int)
        # 初始化蛇的位置
        self.snake = np.array([[3, 1], [3, 2], [3, 3]], dtype=int)
        # 初始化苹果位置
        # self.apples = np.random.randint(1, np.array(self.map.shape) - 1, size=(self.num_apples, 2))
        self.apples = np.array([[3, 5]])
        # 根据蛇的位置和苹果的位置设置地图
        self.set_map()
        obs = self.get_obs()
        self.observation_dim = len(obs)
        return obs

    def get_obs(self):
        """获取环境"""
        self.set_map()
        # 得到蛇头位置
        snake_head = self.snake[-1]
        # 当前运动方向
        directions = np.array(self.direction == self.direction_code).all(axis=1)
        # 检查上下左右是否有东西阻挡，若有则为1
        if self.map[snake_head[0], snake_head[1]] > 3:
            # 如果已经撞墙了则全为 1
            up_bar, down_bar, left_bar, right_bar = 1, 1, 1, 1
        else:
            up_bar = 1 if (self.map[snake_head[0] - 1, snake_head[1]] == 1 or
                           self.map[snake_head[0] - 1, snake_head[1]] == 3) else 0
            down_bar = 1 if (self.map[snake_head[0] + 1, snake_head[1]] == 1 or
                             self.map[snake_head[0] + 1, snake_head[1]] == 3) else 0
            left_bar = 1 if (self.map[snake_head[0], snake_head[1] - 1] == 1 or
                             self.map[snake_head[0], snake_head[1] - 1] == 3) else 0
            right_bar = 1 if (self.map[snake_head[0], snake_head[1] + 1] == 1 or
                              self.map[snake_head[0], snake_head[1] + 1] == 3) else 0
        # 检查上下左右部分是否有食物
        up_apple = 1 if np.sum(self.map[:snake_head[0], :] == 2) > 0 else 0
        down_apple = 1 if np.sum(self.map[snake_head[0] + 1:, :] == 2) > 0 else 0
        left_apple = 1 if np.sum(self.map[:, :snake_head[1]] == 2) > 0 else 0
        right_apple = 1 if np.sum(self.map[:, snake_head[1] + 1:] == 2) > 0 else 0
        # 检查上下左右方向是否有自己的身体
        up_self = 0 if np.sum(self.map[:snake_head[0], :] == 3) > 0 else 1
        down_self = 0 if np.sum(self.map[snake_head[0] + 1:, :] == 3) > 0 else 1
        left_self = 0 if np.sum(self.map[:, :snake_head[1]] == 3) > 0 else 1
        right_self = 0 if np.sum(self.map[:, snake_head[1] + 1:] == 3) > 0 else 1
        # up_self = 1 if np.sum(self.map[:snake_head[0], :] == 3) > 0 else 0
        # down_self = 1 if np.sum(self.map[snake_head[0] + 1:, :] == 3) > 0 else 0
        # left_self = 1 if np.sum(self.map[:, :snake_head[1]] == 3) > 0 else 0
        # right_self = 1 if np.sum(self.map[:, snake_head[1] + 1:] == 3) > 0 else 0
        # 获取上下左右方向的信息
        obs = np.array([up_bar, down_bar, left_bar, right_bar,
                        up_apple, down_apple, left_apple, right_apple,
                        up_self, down_self, left_self, right_self], dtype=float)
        obs = np.concatenate((directions, obs))
        return obs

    def set_map(self):
        """设置地图"""
        # 重置地图
        self.map = np.zeros((self.map_size, self.map_size), dtype=int)
        # 创建墙壁
        self.map[0, :] = self.map[-1, :] = 1
        self.map[:, 0] = self.map[:, -1] = 1
        # 创建蛇和苹果
        np.add.at(self.map, (self.snake[:, 0], self.snake[:, 1]), 3)
        np.add.at(self.map, (self.apples[:, 0], self.apples[:, 1]), 2)

    def set_direction(self, action):
        """将动作转换为方向变化情况"""
        self.direction = self.direction_code[action]

    def get_apple(self):
        """判断是吃了哪个苹果"""
        sum_state = np.sum(np.abs(self.snake[-1] - self.apples), axis=1)
        if np.sum(sum_state == 0) == 0:
            return -1
        else:
            return np.where(sum_state == 0)[0][0]

    def cal_reward(self, obs):
        """计算奖励"""
        # 当前运动方向上有食物则给正奖励，否则为负奖励
        if np.array((obs[0:4] == 1) + (obs[8:12] == 1) == 2).any():
            # 奖励计算为长度的倒数
            return 1 / len(self.snake)
        else:
            return -1 / len(self.snake)

    def step(self, action):
        # 根据动作转变运动方向
        self.set_direction(action)
        # 创建下一个蛇头
        self.snake = np.append(self.snake, [self.snake[-1] + self.direction], axis=0)
        # 如果蛇没有吃到苹果，删除蛇尾
        apple_id = self.get_apple()
        if apple_id == -1:
            self.snake = np.delete(self.snake, 0, 0)
            obs = self.get_obs()
            # 没吃到苹果，根据方向和确定奖励
            # reward = self.cal_reward(obs)
            reward = -1 / len(self.snake)
            self.no_act_step += 1
        # 否则得分并重新生成一个苹果位置
        else:
            # 蛇吃了苹果则给奖励，得分+1
            reward = len(self.snake) * 2
            self.score += 1
            self.no_act_step = 0  # 重置
            # 再次生成苹果时要避开蛇的位置
            grow = np.vstack(np.where(self.map == 0)).T
            # 检查是否还有生成位置
            if len(grow):
                grow_position = grow[np.random.randint(len(grow))]
                self.apples[apple_id] = grow_position
            else:
                # 若没有生成位置了则完美结束
                return self.get_obs(), 1000.0, True, None
            # 得到新的环境
            obs = self.get_obs()
        # 如果蛇撞到墙壁或者吃到自己 或者 无动作步数太多 则结束并给负奖励
        if np.sum(self.map > 3) or self.no_act_step > 300:
            done = True
            reward = -10.0
            # reward = -100.0/(len(self.snake))
        else:
            done = False
        return obs, reward, done, None

    def render(self, pause_time, file_name='', pic_id=''):
        """绘制当前状态"""
        # 定义颜色列表
        colors = [(0.0, 0.0, 0.0),  # 黑色
                  (0.0, 0.0, 1.0),  # 蓝色
                  (1.0, 0.0, 0.0),  # 红色
                  (0.0, 1.0, 0.0)]  # 绿色
        # 创建 ListedColormap 对象
        custom_cmap = ListedColormap(colors)
        plt.clf()
        fig = plt.gcf()  # 获取当前图形对象
        fig.patch.set_facecolor('black')  # 设置图形底色为黑色
        plt.imshow(self.map, cmap=custom_cmap, interpolation='none')
        plt.text(1, 0.3, 'Score: ' + str(self.score), ha='center', va='bottom', color='white', fontsize=12)
        plt.tight_layout()
        plt.axis('off')
        plt.pause(pause_time)
        # 保存文件
        # folder_name = "D:/PlotSnake/" + file_name
        # # 检查文件夹是否存在
        # if not os.path.exists(folder_name):
        #     # 文件夹不存在，则创建文件夹
        #     os.makedirs(folder_name)
        # plt.savefig(folder_name + '/' + str(pic_id) + '.png', dpi=160)


if __name__ == '__main__':
    env = Snake()
    obs = env.reset()
    for i in range(100):
        # action = np.random.randint(0, 5)
        action = int(input())
        obs, reward, done, info = env.step(action)
        env.render(0.5)
        print(reward)
        if done:
            break
