import copy
import random
from functools import partial

import gym # RL environments
import jax # Autograd package
import jax.numpy as np # GPU NumPy :)

import haiku as hk

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


## Creating the model forward step
def forward_fn(observations):
    mlp = hk.Sequential([
        hk.Linear(300), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(2)
    ])
    return mlp(observations)

# We also declare a function to compute the loss given the model 
# parameters
def loss_fn(params, observations, q_values, actions):
    logits = model.apply(params, observations)
    # We pick the values of taken actions
    logits = logits[np.arange(logits.shape[0]), actions]
    return nn.losses.mse(q_values, logits, reduction='mean') 


random_key = jax.random.PRNGKey(0)

# When using haiku, model has to be initialized using mock inputs,
# they can be random
model = hk.transform(forward_fn)

random_key, model_key = jax.random.split(random_key)
dqn_params = model.init(
    model_key, 
    jax.random.normal(model_key, shape=(1, 4)))

# Target dqn contains a copy of dqn parameters
# This model will take care of computing the expected q values
target_dqn_params = copy.deepcopy(dqn_params)

# Define the backward step by just decorating the model's apply function
# To do so, we use the loss function in order to compute the gradients,
# wtr to the loss
backward_fn = jax.grad(jax.jit(loss_fn))

# Declare an SGD optimizer
optimizer = partial(nn.optim.simple_optimizer, learning_rate=1e-3)

## Take action
def take_action(key, state):
    key, sk = jax.random.split(key)
    if jax.random.uniform(sk, shape=(1,)) < EPSILON:
        # Remember that we have only 2 actions (left, right)
        action = jax.random.randint(sk, shape=(1,), minval=0, maxval=2)
    else:
        q_values = model.apply(dqn_params, state.reshape(1, -1))
        # Pick the action that maximizes the value
        action = np.argmax(q_values)
    
    return int(action)


BATCH_SIZE = 32
GAMMA = .99
EPSILON = .3

def train():
    if len(memory) < BATCH_SIZE:
        return dqn_params # No train because we do not have enough experiences
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
    next_Q_values = model.apply(target_dqn_params, next_states)
    # Bellman equation
    yj = rewards + GAMMA * np.max(next_Q_values, axis=-1)
    # In case of terminal state we set a 0 reward
    yj = np.where(is_terminal, 0, yj)

    # Compute the Qvalues corresponding to the sampled transitions
    # and backpropagate the mse loss to compute the gradients
    gradients = backward_fn(dqn_params, states, yj, actions)
    
    # Update the policy gradients and return the
    # updated model
    return jax.tree_multimap(optimizer, dqn_params, gradients)
    

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
        dqn_params = train() # Update the agent with experience replay

        state = next_state

        if done: 
            break # The pole has felt
        
    # At the end of the episode we update the target parameters
    # Remember that during the episode we have updated the dqn parameters
    target_dqn_params = copy.deepcopy(dqn_params)

if __name__ == "__main__":
    run()