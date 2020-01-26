import random
from functools import partial

import gym # RL environments
import jax # Autograd package
import jax.numpy as np # GPU NumPy :)

import rl_jax.nn as nn # Custom package

from collections import deque # Cyclic list with max capacity
from typing import *
from rl_jax.typing import JaxTensor


class Transition(NamedTuple):
    state: JaxTensor
    next_state: JaxTensor
    action: Union[JaxTensor, int]
    reward: int
    is_terminal: bool


class ReplayMemory(object):

    def __init__(self, size: int = 1e5):
        self.memory = deque(maxlen=int(size))
    
    def experience(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, sample_size: int) -> Sequence[Transition]:
        return random.sample(self.memory, k=sample_size)
    
    def __len__(self):
        return len(self.memory)


## Creating the model and backward step
def create_dqn(key):
    model =  nn.Sequential(
        nn.Linear(in_features=4, 
                  out_features=32,
                  activation=jax.nn.relu),
        nn.Linear(in_features=32, 
                  out_features=32,
                  activation=jax.nn.relu),
        nn.Linear(in_features=32, 
                  out_features=2))
    model.init(key)
    return model

random_key = jax.random.PRNGKey(0)

# We create the model using the same key 
# so they are initialized with the same parameters
dqn = create_dqn(random_key)
# Create a target model for training stability
target_dqn = create_dqn(random_key)

# Mean squared error
mse = lambda y1, y2: (y1 - y2) ** 2

@jax.grad # Differentiate the loss
def compute_loss(dqn, x, y, actions):
    # Get the q values corresponding to specified actions
    q_values = jax.vmap(dqn)(x) # Vectorized model 
    q_values = q_values[np.arange(x.shape[0]), actions]
    return np.mean(mse(y, q_values))

# Again, we compile the function with jit to improve performance
backward_fn = jax.jit(compute_loss)

# Declare an SGD optimizer
optimizer = nn.optim.simple_optimizer(learning_rate=1e-3)
optimizer = jax.jit(optimizer) # Compile

## Take action
def take_action(key, state):
    key, sk = jax.random.split(key)
    if jax.random.uniform(sk, shape=(1,)) < EPSILON:
        # Remember that we have only 2 actions (left, right)
        action = jax.random.randint(sk, shape=(1,), minval=0, maxval=2)
    else:
        # When invoking __call__ method of JaxModule the parameters are implicitly
        # passed as argument of the forward function of the model (refer to custom nn module implementation)
        # Compute state QValues
        q_values = dqn(state, training=False)
        # Pick the action that maximizes the value
        action = np.argmax(q_values)
    
    return int(action)


BATCH_SIZE = 32
GAMMA = .99
EPSILON = .3

def train():
    if len(memory) < BATCH_SIZE:
        return dqn # No train because we do not have enough experiences
    # Experience replay
    transitions = memory.sample(BATCH_SIZE)
        
    # Convert transition into tensors
    transitions = Transition(*zip(*transitions))
    states = np.array(transitions.state)
    next_states = np.array(transitions.next_state)
    actions = np.array(transitions.action)
    rewards = np.array(transitions.reward)
    is_terminal = np.array(transitions.is_terminal)

    # Compute the next Q values using the target parameters
    # We vectorize the model using vmap to work with batches
    next_Q_values = jax.vmap(target_dqn)(next_states)
    # Bellman equation
    yj = rewards + GAMMA * np.max(next_Q_values, axis=-1)
    # In case of terminal state we set a 0 reward
    yj = np.where(is_terminal, 0, yj)

    # Compute the Qvalues corresponding to the sampled transitions
    # and backpropagate the mse loss to compute the gradients
    gradients = backward_fn(dqn, states, yj, actions)
    
    # Update the policy gradients and return the
    # updated model
    return optimizer(dqn, gradients)
    

MAX_EPISODES = 300
MAX_EPISODE_STEPS = 100 # When agents holds the pole for 100 steps we are done :)

env = gym.make('CartPole-v1')

memory = ReplayMemory()

for i_episode in range(MAX_EPISODES):
    state = env.reset()
    for timestep in range(MAX_EPISODE_STEPS):
        env.render() # Display the environment
        
        # Take an action
        random_key, subkey = jax.random.split(random_key)
        action = take_action(subkey, state)
        next_state, reward, done, _ = env.step(action)
        
        # Generate a transition
        t = Transition(state=state, next_state=next_state, 
                    reward=reward, is_terminal=done,
                    action=action)
        memory.experience(t) # Store the transition
        dqn = train() # Update the agent with experience replay

        state = next_state

        if done: 
            break # The pole has felt
        
    # At the end of the episode we update the target parameters
    # Remember that during the episode we have updated the dqn parameters
    target_dqn = target_dqn.update(dqn.parameters)

if __name__ == "__main__":
    run()