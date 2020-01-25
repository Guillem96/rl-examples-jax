import gym
import matplotlib.pyplot as plt

import numpy as np

from .agent import DeepQLearning
from rl_jax.utils import Transition


def moving_average(a, n=3) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def main():
    timesteps = []

    env = gym.make('CartPole-v1')
    agent = DeepQLearning(epsilon=.3, 
                          gamma=.99, 
                          memory_size=1e3)
    
    end_train_after_n_episodes = 250

    for i_episode in range(1000):
        state = env.reset()
        if i_episode > end_train_after_n_episodes:
            agent.end_train()

        for timestep in range(100):
            
            if i_episode > end_train_after_n_episodes:
                env.render()
            
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            
            t = Transition(state=state, next_state=next_state, 
                           reward=reward, is_terminal=done,
                           action=action)
            agent.update(t)
            
            state = next_state

            if done:
                break

        timesteps.append(timestep + 1)
        if len(timesteps) > 5:
            y = moving_average(timesteps, 5)
            x = range(y.shape[0])
            plt.plot(x, y)
            plt.xlabel('Episodes')
            plt.ylabel('Timesteps')
            plt.savefig('test.png')

        print("Episode finished after {} timesteps".format(timestep+1))
    
    env.close()


if __name__ == "__main__":
    main()