import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")
q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 1.0  # 探索率
epsilon_decay = 0.99

# 训练5000轮
training_episodes = 5000
display_episodes = 20

episode_rewards = []
episode_lengths = []

# 定义 Reward Shaping 相关的辅助奖励
def shaped_reward(old_state, new_state, reward):
    # 如果新的状态比旧的状态更接近目标，就给予额外的奖励
    # 比如：状态编号大的值可能意味着更接近目标
    proximity_bonus = 0
    if new_state > old_state:  # 简单的接近性条件
        proximity_bonus = 1  # 正奖励
    
    # 维持主要的环境奖励，同时增加额外的奖励
    return reward + proximity_bonus

# 训练智能体
for i in range(training_episodes):
    observasion, info = env.reset()
    done = False
    steps = 0
    total_rewards = 0
    
    while not done:
        # epsilon-greedy 策略选择动作
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[observasion])

        next_observasion, reward, done, truncated, info = env.step(action)
        
        # 使用 Reward Shaping 技巧
        shaped_r = shaped_reward(observasion, next_observasion, reward)
        
        # Q-learning 更新公式
        old_q = q_table[observasion, action]
        max_q = np.max(q_table[next_observasion])
        q_table[observasion, action] = old_q + alpha * (shaped_r + gamma * max_q - old_q)

        total_rewards += shaped_r
        steps += 1
        observasion = next_observasion

        # 减少 epsilon
        epsilon = max(0.1, epsilon * epsilon_decay)
    
    episode_rewards.append(total_rewards)
    episode_lengths.append(steps)
    
    if i % 1000 == 0:
        print(f"Episode: {i}, Total Reward: {total_rewards}")

print("训练完成")

# 训练过程可视化
# 累积奖励
plt.figure(figsize=(8, 5))
plt.plot(episode_rewards)
plt.title("Episode-rewards with Reward Shaping")
plt.xlabel("Episodes")
plt.ylabel("Total rewards")
plt.show()

# 累积步数
plt.figure(figsize=(8, 5))
plt.plot(episode_lengths)
plt.title("Episode-lengths with Reward Shaping")
plt.xlabel("Episodes")
plt.ylabel("Total steps")
plt.show()


total_epochs,total_penalties = 0,0
for _ in range(display_episodes):
    env = gym.make('Taxi-v3',render_mode = 'human')
    observasion,info = env.reset()
    epochs,penalties,reward = 0,0,0
    done = False

    while not done:
        action = np.argmax(q_table[observasion])
        observasion,reward,done,truncated,info = env.step(action)
        if reward == -10:
            penalties += 1

        epochs += 1

        
        print(f"Timestep: {epochs}")
        print(f"Obs: {observasion}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {display_episodes} episodes:")
print(f"平均步数: {total_epochs / display_episodes}")
print(f"平均惩罚: {total_penalties / display_episodes}")
