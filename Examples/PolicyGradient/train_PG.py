import gym
import pygame
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from Core.Module import MLP
from algorithm import PolicyGradient
from Core.Activation import Tanh, Softmax

# 设置帧速率以方便展示
clock = pygame.time.Clock()


# 训练一个episode
def run_episode(env, agent):
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


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                # 定义帧速率
                clock.tick(50)
                env.render()
                # 解决pygame window无响应问题
                for event in pygame.event.get():
                    pass
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


# 使用每个step的reward，计算每个step的未来总收益
# [r1, r2, r3, ...rt] -> [G0, G1, G2, ..., Gt-1]
# G_t = r_t+1 + gamma * G_t+1
def calc_reward_to_go(reward_list, gamma=1.0):
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_i = r_i+1 + γ·G_i+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


def main():
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 构建agent
    model = MLP(input_size=obs_dim, output_size=act_dim, hidden_sizes=[act_dim * 10],
                hidden_activation=Tanh, out_activation=Softmax)
    alg = PolicyGradient(model, lr=1e-3)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

    train_rewards = []
    eval_reward = 0
    eval_rewards = []
    for i in range(1000):
        obs_list, action_list, reward_list = run_episode(env, agent)
        train_rewards.append(sum(reward_list))
        eval_rewards.append(eval_reward)
        if i % 10 == 0:
            print("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)  # 根据单步收益，求未来总收益

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            eval_reward = evaluate(env, agent, render=True)
            print('Test reward: {}'.format(eval_reward))

    env.close()

    # 绘制reward的变化图
    plt.figure(0)
    plt.plot(np.arange(len(train_rewards)), train_rewards, c='skyblue', label='train reward')
    plt.plot(np.arange(len(eval_rewards)), eval_rewards, c='orangered', label='eval reward')
    plt.title('Reward')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
