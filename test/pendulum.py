import gym
import numpy as np
env = gym.make('Pendulum-v0')
for i_episode in range(1):
    observation = env.reset()
    reward = 0
    for t in range(10):
        env.render()
        print(observation,reward)
        action = [0.0]
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
