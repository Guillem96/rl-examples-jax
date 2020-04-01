import copy
import functools

import jax
import jax.numpy as np

import haiku as hk

import rl_jax.nn as nn
import rl_jax.utils as rl_utils
from rl_jax.typing import JaxTensor


class DeepQLearning(object):

    def __init__(self, 
                 epsilon: float = .2,
                 gamma: float= .99,
                 memory_size: int = 1e6,
                 batch_size: int = 32):
        
        self.training = True
        self.epsilon = epsilon
        self.gamma = gamma

        self.memory = rl_utils.ReplayMemory(size=memory_size)
        self.batch_size = batch_size
        self.random_key = jax.random.PRNGKey(0)
        
        # Create the forward pass of the model
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
            logits = self.model.apply(params, observations)
            # We pick the values of taken actions
            logits = logits[np.arange(logits.shape[0]), actions]
            return nn.losses.mse(q_values, logits, reduction='mean') 

        # When using haiku, model has to be initialized using mock inputs,
        # they can be random
        self.model = hk.transform(forward_fn)

        self.random_key, model_key = jax.random.split(self.random_key)
        self.dqn_params = self.model.init(
            model_key, 
            jax.random.normal(model_key, shape=(1, 4)))
        
        # Target dqn contains a copy of dqn parameters
        # This model will take care of computing the expected q values
        self.target_dqn_params = copy.deepcopy(self.dqn_params)
        
        # Define the backward step by just decorating the model's apply function
        # To do so, we use the loss function in order to compute the gradients,
        # wtr to the loss
        self.backward_fn = jax.grad(jax.jit(loss_fn))

        self.optimizer = functools.partial(nn.optim.simple_optimizer, 
                                           learning_rate=1e-3)
    
    def _train(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        
        # Convert transition into tensors
        transitions = rl_utils.Transition(*zip(*transitions))
        states = np.array(transitions.state)
        next_states = np.array(transitions.next_state)
        actions = np.array(transitions.action)
        rewards = np.array(transitions.reward)
        is_terminal = np.array(transitions.is_terminal)

        # Compute the next Q values using the target parameters
        next_Q_values = self.model.apply(self.target_dqn_params, next_states)
        yj = rewards + self.gamma * np.max(next_Q_values, axis=-1)
        # In case of terminal state we set a 0 rweward
        yj = np.where(is_terminal, 0, yj)

        # Compute the Qvalues corresponding to the sampled transitions
        # and backpropagate the mse loss to compute the gradients
        gradients = self.backward_fn(self.dqn_params, states, yj, actions)
        
        # Update the policy gradients
        self.dqn_params = jax.tree_multimap(
            self.optimizer, self.dqn_params, gradients)
        
    def update(self, t: rl_utils.Transition):
        if not self.training:
            return

        self.memory.experience(t)
        self._train()

        # Linear decay of exploration rate
        self.epsilon -= 2e-5

        # Update the target parameters with the "experiences" of 
        # the current episode
        if t.is_terminal:
            self.target_dqn_params = copy.deepcopy(self.dqn_params)

    def take_action(self, state: JaxTensor) -> int:
        self.random_key, sk = jax.random.split(self.random_key)
        if jax.random.uniform(sk, shape=(1,)) < self.epsilon:
            action = jax.random.randint(sk, shape=(1,), minval=0, maxval=2)
        else:
            q_values = self.model.apply(self.dqn_params, state.reshape(1, -1))
            action = np.argmax(q_values)
        
        return int(action)

    def end_train(self):
        self.epsilon = 0
        self.training = False
        self.memory.memory.clear()