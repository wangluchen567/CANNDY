import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from model import Model
from algorithm import PolicyGradient
from replay_memory import ReplayMemory
from Examples.RL_Envs.Snake import Snake

# 超参数设置
LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 2000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 64  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等

def run_episode(env, agent, rpm):
    # rpm: ReplayMemory 经验回放池
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到

        next_obs, reward, done, _ = env.step(action)
        # print(step, action, reward, done)
        # 将得到的信息加入经验回放池
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(BATCH_SIZE)
            batch_reward = calc_reward_to_go(batch_reward)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward)
        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render(0.1)
            if done:
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
    if len(reward_arr) == 1:
        # 若长度为1则可能会出现除以0的情况
        reward_arr /= np.abs(reward_arr)
    else:
        reward_arr -= np.mean(reward_arr)
        reward_arr /= np.std(reward_arr)
    return reward_arr


def main():
    env = Snake()
    obs_dim = env.observation_dim
    act_dim = env.action_dim
    print('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 构建agent
    model = Model(input_size=obs_dim, output_size=act_dim, hidden_sizes=[128])
    alg = PolicyGradient(model, lr=1e-3)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

    max_episode = 10000
    train_rewards = []
    eval_reward = 0
    eval_rewards = []
    # 开始训练
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # 训练部分
        for i in range(0, 50):
            total_reward = run_episode(env, agent, rpm)
            train_rewards.append(total_reward)
            eval_rewards.append(eval_reward)
            episode += 1

        # 测试部分
        if episode > 5000:
            eval_reward = evaluate(env, agent, render=True)
        else:
            eval_reward = evaluate(env, agent, render=False)
        print('episode:{}, test reward:{}'.format(episode, eval_reward))

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
