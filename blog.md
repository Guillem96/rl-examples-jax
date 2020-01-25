# Cartpole with JAX

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

We are going to implement the straightforward algorithm Q-Learning but applying Neural Networks as a function approximator for the value function. This variation of the algorithm is called Deep Q-Learning, and it is described in this paper [2].

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

The main idea behind Q-Learning is that if we have a function $ Q*(\phi, a) $ that can tell us the expected utility given a state $ \phi $ and an action $ a $ we will be able to extract an optimal policy $ \pi $  that maximizes the utility or return over a sequence of states.

## Implementation

I hope you are ready for a JAX implementation 😎.

```python
from functools import partial

import gym # RL environments
import jax # Autograd package
import jax.numpy as np # GPU NumPy :)

import rl_jax.nn as nn # Custom package
```

To make life easier, I developed a high-level API to prototype NN architectures. The API is thought to be simple and intuitive, but it requires a mid-level understanding of higher-order functions.

This is because JAX is pretty functional and develop extension over it using classes or objects requires complex workarounds.

To learn more about my custom nn package please refer to this [training example](https://github.com/Guillem96/rl-examples-jax/blob/master/test/train_mnist.py).

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
from typing import Union, NamedTuple
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
```

### The model

Our model will be a simple set of linear layers (if you come from a tensorflow or keras background, a set of Dense layers) with an output
of size equal to the number of possible actions, in this case 2 (move left or right), without any activation.

```python
random_key = jax.random.PRNGKey(0)
dqn = nn.sequential(
    random_key, # remember in jax the random key is mandatory
    partial(nn.linear, 
            in_features=4, 
            out_features=32,
            activation=jax.nn.relu),
    partial(nn.linear, 
            in_features=32, 
            out_features=32,
            activation=jax.nn.relu),
    partial(nn.linear, 
            in_features=32, 
            out_features=2))

# Vectorize the function to automatically work with batches
dqn_fn = jax.vmap(dqn.forward_fn, in_axes=(None, 0))
dqn_fn = jax.jit(dqn_fn) # Compile the function with JIT

# Create target parameters for training stability
target_params = dqn.parameters
```

A part from the model we have to define the loss function with respect of the model
parameters, so later we can differentiate it and compute the gradients.

```python
mse = lambda y1, y2: (y1 - y2) ** 2

@jax.grad # Differentiate the loss
def compute_loss(params, x, y, actions):
    # Get the q values corresponding to specified actions
    q_values = dqn_fn(params, x)
    q_values = q_values[np.arange(x.shape[0]), actions]
    return np.mean(mse(y, q_values))

# Again, we compile the function with jit to improve performanve
backward_fn = jax.jit(compute_loss)

# Declare an SGD optimizer
optimizer = nn.optim.simple_optimizer(learning_rate=1e-3)
optimizer = jax.jit(optimizer) # Compile
```

## References

[1] 

[2] Playing Atari with Deep Reinforcement Learning - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf