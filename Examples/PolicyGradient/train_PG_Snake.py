import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from Core.Module import MLP
from Core.Activation import Tanh, Softmax

from algorithm import PolicyGradient
from Examples.RL_Envs.Snake import Snake


class PGSnake:

    def __init__(self, learning_rate=1.e-3, max_episode=100000):
        """
        训练PG玩贪吃蛇
        :param learning_rate: 学习率
        :param max_episode: 训练最大回合数
        """
        self.learning_rate = learning_rate  # 每次agent学习率
        self.max_episode = max_episode  # 训练最大回合数
        self.env = Snake()  # 初始化游戏环境
        self.obs_dim = self.env.observation_dim  # 状态空间大小
        self.act_dim = self.env.action_dim  # 动作空间大小
        self.train_rewards = None  # 记录训练回报
        self.eval_rewards = None  # 记录验证回报
        # 构建agent
        self.model = MLP(input_size=self.obs_dim, output_size=self.act_dim, hidden_sizes=[64],
                         hidden_activation=Tanh, out_activation=Softmax)
        self.algorithm = PolicyGradient(self.model, lr=self.learning_rate)
        self.agent = Agent(self.algorithm, obs_dim=self.obs_dim, act_dim=self.act_dim)

    def train(self):
        """训练PG"""
        # 初始化训练回报和验证回报
        eval_reward = 0
        self.train_rewards = []
        self.eval_rewards = []
        for i in range(self.max_episode):
            obs_list, action_list, reward_list = self.run_episode(self.env, self.agent)
            self.train_rewards.append(sum(reward_list))
            self.eval_rewards.append(eval_reward)
            if i % 1000 == 0:
                print(f"episode: {i}, train reward: {sum(reward_list)}")

            batch_obs = np.array(obs_list)
            batch_action = np.array(action_list)
            # 根据单步收益，求未来总收益
            batch_reward = self.calc_reward_to_go(reward_list)

            self.agent.learn(batch_obs, batch_action, batch_reward)
            if (i + 1) % 10000 == 0:
                eval_reward = self.evaluate(self.env, self.agent, render=False)
                print(f"test reward: {eval_reward}")

    @staticmethod
    def run_episode(env, agent):
        """训练一个episode"""
        obs_list, action_list, reward_list = [], [], []
        obs = env.reset()
        while True:
            obs_list.append(obs)
            action = agent.sample(obs)
            action_list.append(action)

            obs, reward, done, info = env.step(action)
            reward_list.append(reward)

            if done:
                break
        return obs_list, action_list, reward_list

    @staticmethod
    def evaluate(env, agent, render=False):
        # 评估agent
        eval_reward = []
        for i in range(5):
            obs = env.reset()
            episode_reward = 0
            while True:
                action = agent.predict(obs)  # 预测动作，只选最优动作
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                if render:
                    env.render()
                if done:
                    break
            eval_reward.append(episode_reward)
        return np.mean(eval_reward)

    @staticmethod
    def calc_reward_to_go(reward_list, gamma=1.0):
        """使用每个step的reward，计算每个step的未来总收益"""
        # [r1, r2, r3, ...rt] -> [G0, G1, G2, ..., Gt-1]
        # G_t = r_t+1 + gamma * G_t+1
        reward_arr = np.array(reward_list)
        for i in range(len(reward_arr) - 2, -1, -1):
            # G_i = r_i+1 + γ·G_i+1
            reward_arr[i] += gamma * reward_arr[i + 1]
        # normalize episode rewards
        if len(reward_arr) == 1:
            # 若长度为1则可能会出现除以0的情况
            reward_arr /= np.abs(reward_arr)
        else:
            reward_arr -= np.mean(reward_arr)
            reward_arr /= np.std(reward_arr)
        return reward_arr

    def show_best(self):
        """展示模型效果"""
        # 为最优模型构建agent
        algorithm = PolicyGradient(self.model, lr=self.learning_rate)
        best_agent = Agent(
            algorithm,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )
        # 展示最优模型的效果
        eval_reward = self.evaluate(self.env, best_agent, render=True)
        print("best model test reward: ", eval_reward)

    def plot_reward(self):
        # 绘制reward的变化图
        plt.figure(0)
        plt.plot(np.arange(len(self.train_rewards)), self.train_rewards, c='skyblue', label='train reward')
        plt.plot(np.arange(len(self.eval_rewards)), self.eval_rewards, c='orangered', label='eval reward')
        plt.title('Reward')
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    demo = PGSnake()
    demo.train()
    demo.show_best()
    demo.env.close()
    demo.plot_reward()
