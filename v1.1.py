import gymnasium as gym
import numpy as np
import random

env = gym.make("Taxi-v3",render_mode = 'human')
q_table = np.zeros([env.observation_space.n,env.action_space.n])

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.99
training_episodes = 10000
display_episodes = 20
max_steps = 100


# 训练智能体
for i in range(training_episodes):
    observasion,info = env.reset()
    reward,penalty = 0,0
    done = False
    steps = 0

    while not done and steps < max_steps:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[observasion])
        
        next_observasion,reward,done,truncated,info = env.step(action)
        old_q = q_table[observasion,action]
        max_q = np.max(q_table[next_observasion])

        next_q = old_q + alpha*(reward+gamma*max_q - old_q)

        if reward == -10:
            penalty += 1
        
        steps += 1
        observasion = next_observasion
    epsilon = max{0.1,epsilon*epsilon_decay}

    if i % 1000 == 0:
        print(f"Episode:{i}")

print("训练完成")

total_epochs,total_penalties = 0,0
for _ in range(display_episodes):
    observasion,info = env.reset()
    epochs,penalties,reward = 0,0,0
    done = False

    while not done:
        action = np.argmax(q_table[observasion])
        observasion,reward,done,truncated,info = env.step(action)
        if reward == -10:
            penalties += 1

        epochs += 1

        env.render("human")
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        sleep(0.1)  

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {display_episodes} episodes:")
print(f"平均步数: {total_epochs / display_episodes}")
print(f"平均惩罚: {total_penalties / display_episodes}")
