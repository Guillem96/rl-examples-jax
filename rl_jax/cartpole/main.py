
import gym

def main():
    env = gym.make('CartPole-v1')
    for i_episode in range(20):
        state = env.reset()
        for t in range(100):
            env.render()
            
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            state = next_state

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

if __name__ == "__main__":
    main()