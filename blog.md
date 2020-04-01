# CartPole with JAX

Welcome back guys, it's been a long since last blog post. I come with new interests and a lot of new knowledge to share with you. 
In this post, I am going to explain to you my brand new toy: **JAX** [1].

![JAX Logo](https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png)

*Figure 1. JAX Logo*

More precisely, I am going to demonstrate how we can apply JAX to a reinforcement learning (RL) framework. 
To properly understand this post, you should be familiar with:
Deep Learning fundamentals
At least one Deep Learning Framework (PyTorch recommended)
RL basics
Without further ado, let's start 🤗

## What JAX exactly is?

The following definition is how JAX team describes it with a single sentence: **JAX is NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research.**

But JAX is a lot more, with it you can differentiate Python code as-is, making the backpropagation implementation simpler than with any other previous framework. JAX also implements fully reproducible experiments by making the random seed (what they call a *key*) mandatory for every random operation.
To better understand how JAX works, let's see a quick example on how to compute the derivative of a simple function such as $ 2x^3 $

```python
import jax

# f(x) = 2x^3
f = lambda x: 2 * x ** 3

# f(2) = 2x2^3 = 16
assert f(2.) == 16.

# differentiate the function to compute the derivative
# I am pretty sure you know that the derivative of 
# 2 * x ** 3 is 6 * x ** 2 
# vf(x) = 6x^2 
df = jax.grad(f)
# vf(2) = 6x2^2 = 24
assert df(2.) == 24.
```

## Which RL algorithm are we going to train?

We are going to implement the straightforward algorithm Q-Learning but applying Neural Networks as a function estimator for the value function. This variation of the algorithm is called Deep Q-Learning, and it is described in this paper [2].

To train the agent, we are going to use gym package, more precisely the CartPole environment. 

At [OpenAI GYM](https://gym.openai.com/) web page CartPole is defined as follows:

> A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. 

![CartPole env](https://camo.githubusercontent.com/2e8fc61577c7c6f07ce0b37a8b952e68af756244/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a6f4d5367325f6d4b677541474b793143363455466c772e676966)

*Figure 2. CartPole Environment*

## RL framework and notation

During the post we are going to use the notation described at [2].

We are considering a task in which the agent interacts with an environment $ E $, in this case CartPole, in a sequence of observations ($ \phi $), actions and rewards. At each timestep the agent takes an action $ a_t $ belonging to a set of legal actions $ A = {0, 1} $, and receives a reward $ r_t $ depending on the action benefits. 

The goal of the agent is to maximize the future reward. 

## Deep QLearning Recap

In this section, we are going to describe Deep Q-Learning with few sentences. To better understand the proposed solution you should refer to [2].

The main idea behind Q-Learning is that if we have a function $ Q^*(\phi, a) $ that can tell us the expected utility given a state $ \phi $ and an action $ a $ we will be able to extract an optimal policy $ \pi $  that maximizes the utility or return over a sequence of states.

## Implementation

I hope you are ready for a JAX implementation 😎.

```python
import copy
import random
from typing import *
from functools import partial

import gym # RL environments
import jax # Autograd package
import jax.numpy as np # GPU NumPy :)

import haiku as hk

import rl_jax.nn as nn # Custom package
from rl_jax.typing import JaxTensor
```

To make life easier, instead of developing neural network layers from scratch, we are going to use the [Haiku](https://github.com/deepmind/dm-haiku) package. Haiku API is thought to be simple and intuitive, but it requires a mid-level understanding of higher-order functions.

This is because JAX is pretty functional and develop extension over it using classes or objects requires complex workarounds.

### Replay Memory

To train our agent, we are going to use a technique called *experience replay* where, at each timestep $ t $, we store the agent experiences $ e_t = (\phi_t, \phi_{t+1}, a_t, r_t, d_t)$ in a *replay memory*. We also call an experience a Transition. 

The elements of a transition are:
- $ \phi_t $: Current environment observation.
- $ a_t $: The action that the agent has just taken.
- $ \phi_{t+1} $: The environment observation after taking the action. It is usually called next state.
- $ r_t $: The achieved reward during the transition.
- $ d_t $: If the transition implies a terminal state. For example, in CartPole environment, a terminal state is the one in which the pole falls.

During the agent interaction, we are going to sample $ n $ instances of the replay memory to update our Q-Learning parameters.

Talking about code, we are going to need to classes, one to represent the*transitions*, and another one to implement the basic functionalities of a *Replay Memory*.

```python
from collections import deque # Cyclic list with max capacity

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
```

### The model

Our model will be a simple set of linear layers (if you come from a tensorflow or keras background, a set of Dense layers) with an output
of size equal to the number of possible actions, in this case 2 (move left or right), without any activation.

```python
def forward_fn(observations):
    mlp = hk.Sequential([
        hk.Linear(300), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(2)
    ])
    return mlp(observations)

random_key = jax.random.PRNGKey(0)


# Haiku models has two functions: init and apply
#  - init: Initializes the model's weights given a mock input
#  - apply: Does a forward step
model = hk.transform(forward_fn)

# When using haiku, model has to be initialized using mock inputs,
# they can be random
random_key, model_key = jax.random.split(random_key)
dqn_params = model.init(
    model_key, 
    jax.random.normal(model_key, shape=(1, 4)))
target_dqn_params = copy.deepcopy(dqn_params)
```

A part from the model we have to define the loss function with respect of the model
parameters, so later we can differentiate it and compute the gradients.

```python

@jax.grad
@jax.jit
def backward_fn(params, observations, q_values, actions):
    logits = model.apply(params, observations)
    # We pick the values of taken actions
    logits = logits[np.arange(logits.shape[0]), actions]
    return nn.losses.mse(q_values, logits, reduction='mean') 

# Declare an SGD optimizer
optimizer = partial(nn.optim.simple_optimizer, 
                    learning_rate=1e-3)
```

### Take the action

Our agent has to choose an action every timestep. To do so we use an $ \epsilon $-greedy
strategy. It means that every timestep we flip a biased coin to return true with an 
$ \epsilon $ probability and in this case we should take a random action, otherwise,
we use our value function estimator to pick the action that maximizes the value.

```python
def take_action(key, state):
    key, sk = jax.random.split(key)
    if jax.random.uniform(sk, shape=(1,)) < EPSILON:
        # Remember that we have only 2 actions (left, right)
        action = jax.random.randint(sk, shape=(1,), minval=0, maxval=2)
    else:
        # Compute state QValues
        q_values = model.apply(dqn_params, state.reshape(1, -1))
        # Pick the action that maximizes the value
        action = np.argmax(q_values)
    
    return int(action)
```

### Training step

The basic idea behind many RL algorithms is to estimate the action-value 
function by using th Bellman Equation as an iterative update [2]. 
$ Q_{i+1}(\phi, a) = r + \gamma max_{a'} Q_i(\phi', a') $.
In real-world examples, this is impractical due to large state spaces and time complexity. 
For this reason we use a function estimator $ Q^*(\phi, a) \approx Q(\phi, a; \theta) $,
where $ \theta $ are the Neural network weights.

Therefore, in the following training step, we will be modifying $ \theta $ until
convergence.

```python
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
    # Using tree_multimap we apply the optimizer to all parameters
    return jax.tree_multimap(optimizer, dqn_params, gradients)
```

### Mix implementations

Finally, we are going to mix all the previous implementations into a single loop
and understand how all the different modules interact with each other to achieve a
perfect agent.

```python
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
```

## Results

At *Figure 3* we can see how our agent is increasingly improving until it becomes
nearly perfect by holding the pole always for 100 timesteps.

![CartPole results](https://raw.githubusercontent.com/Guillem96/rl-examples-jax/master/reports/cartpole-results.png)

*Figure 3. CartPole results*


## Takeaways

Well, guys, that's all for today. I hope you liked it, if not at least now you know JAX 😁.

The important takeaways of this post are:

- JAX may become the future of Deep Learning frameworks
- JAX is flexible being capable of implementing custom and complex frameworks such RL is.
- Deep Q Learning is simple but the start of all the known deep RL techniques known nowadays.

## References

[1] JAX: composable transformations of Python+NumPy programs - http://github.com/google/jax

[2] Playing Atari with Deep Reinforcement Learning - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf