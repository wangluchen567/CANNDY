import copy
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from algorithm import DQN
from Core.Module import MLP
from Core.Activation import ReLU
from replay_memory import ReplayMemory
from Examples.RL_Envs.CartPole import CartPole


class DQNCartPole:
    def __init__(self,
                 batch_size=64,
                 learning_rate=1.e-3,
                 max_episode=2000,
                 learn_frequency=5,
                 memory_size=2000,
                 warmup_size=200,
                 gamma=0.99):
        """
        训练DQN玩贪吃蛇
        :param batch_size: 每次给agent学习的数据数量，从replay_memory随机里采样一批数据出来
        :param learning_rate: 每次agent学习率
        :param max_episode: 训练最大回合数
        :param learn_frequency: 训练频率，不需要每一个step都学习，攒一些新增经验后再学习，提高效率
        :param memory_size: replay_memory的大小，越大越占用内存
        :param warmup_size: replay_memory需要预存的经验数据量，再从里面采样一个batch的经验让agent去学习
        :param gamma: reward的衰减因子，一般取 0.9 到 0.999 不等
        """
        self.batch_size = batch_size  # 每次给agent学习的数据数量
        self.learning_rate = learning_rate  # 每次agent学习率
        self.max_episode = max_episode  # 训练最大回合数
        self.learn_frequency = learn_frequency  # 训练频率
        self.memory_size = memory_size  # replay_memory的大小
        self.warmup_size = warmup_size  # replay_memory需要预存的经验数据量
        self.gamma = gamma  # reward的衰减因子
        self.env = CartPole()  # 初始化游戏环境
        self.obs_dim = self.env.observation_dim  # 状态空间大小
        self.act_dim = self.env.action_dim  # 动作空间大小
        self.replay_memory = ReplayMemory(self.memory_size)  # 初始化经验回放池
        self.train_rewards = None  # 记录训练回报
        self.eval_rewards = None  # 记录验证回报
        self.best_e_greed = None  # 记录最优e_greed

        # 构建agent
        self.model = MLP(input_size=self.obs_dim, output_size=self.act_dim, hidden_sizes=[128], hidden_activation=ReLU)
        self.algorithm = DQN(self.model, act_dim=self.act_dim, gamma=self.gamma, lr=self.learning_rate)
        self.agent = Agent(
            self.algorithm,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            e_greed=0.1,  # 有一定概率随机选取动作，探索
            e_greed_decrement=1e-6,  # 随着训练逐步收敛，探索的程度慢慢降低
        )
        # 记录最优模型
        self.best_model = copy.deepcopy(self.model)

    def train(self):
        """训练DQN"""
        # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
        while len(self.replay_memory) < self.warmup_size:
            self.run_episode(self.env, self.agent, self.replay_memory)
        # 初始化训练回报和验证回报
        eval_reward = 0
        max_reward = 0
        self.best_e_greed = 0.1
        self.train_rewards = []
        self.eval_rewards = []
        # 开始训练
        episode = 0
        while episode < self.max_episode:
            # 训练部分
            for i in range(0, 50):
                total_reward = self.run_episode(self.env, self.agent, self.replay_memory)
                self.train_rewards.append(total_reward)
                self.eval_rewards.append(eval_reward)
                episode += 1

            # 测试部分
            eval_reward = self.evaluate(self.env, self.agent, render=False)
            # 记录最优模型
            if eval_reward > max_reward:
                max_reward = eval_reward
                self.best_e_greed = self.agent.e_greed
                self.best_model = copy.deepcopy(self.model)
            print(f"episode: {episode}, e_greed: {self.agent.e_greed}, test reward: {eval_reward}")

    def run_episode(self, env, agent, rpm):
        """训练一个episode"""
        step = 0
        total_reward = 0
        obs = env.reset()
        while True:
            step += 1
            # 采样动作，保证所有动作都有概率被尝试到
            action = agent.sample(obs)

            next_obs, reward, done, _ = env.step(action)
            # 将得到的信息加入经验回放池
            rpm.append((obs, action, reward, next_obs, done))

            # train model
            if (len(rpm) > self.warmup_size) and (step % self.learn_frequency == 0):
                (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(self.batch_size)
                # s, a, r, s', done
                train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
            total_reward += reward
            obs = next_obs
            if done:
                break
        return total_reward

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

    def show_best(self):
        """展示模型效果"""
        # 为最优模型构建agent
        algorithm = DQN(self.best_model, act_dim=self.act_dim, gamma=self.gamma, lr=self.learning_rate)
        best_agent = Agent(
            algorithm,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            e_greed=self.best_e_greed,
            e_greed_decrement=5e-7,
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
    demo = DQNCartPole()
    demo.train()
    demo.show_best()
    demo.env.close()
    demo.plot_reward()
