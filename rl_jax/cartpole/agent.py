from functools import partial

import jax
import jax.numpy as np

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

        # Create the model that is going to be optimized during the episode
        # It also will take the actions
        self.dqn = self._create_model()

        # Vectorize the function to automatically work with batches
        self.dqn_fn = jax.vmap(self.dqn.forward_fn, 
                               in_axes=(None, 0))
        self.dqn_fn = jax.jit(self.dqn_fn) # Compile the function with JIT

        # Create target parameters for training stability
        self.target_params = self.dqn.parameters

        # Declare the mean squared error
        mse = lambda y_true, y_pred: (y_true - y_pred) ** 2 
        
        @jax.grad
        def forward_n_loss(params, x, y, actions):
            # Get the q values corresponding to specified actions
            q_values = self.dqn_fn(params, x)
            q_values = q_values[np.arange(x.shape[0]), actions]
            return np.mean(mse(y, q_values))
        
        self.backward_fn = jax.jit(forward_n_loss)
        
        self.optimizer = nn.optim.simple_optimizer(learning_rate=1e-3)
        self.optimizer = jax.jit(self.optimizer)
    
    def _create_model(self):
        self.random_key, sk = jax.random.split(self.random_key)
        return nn.sequential(
            sk,
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
        next_Q_values = self.dqn_fn(self.target_params, next_states)
        yj = rewards + self.gamma * np.max(next_Q_values, axis=-1)
        # In case of terminal state we set a 0 rweward
        yj = np.where(is_terminal, 0, yj)

        # Compute the Qvalues corresponding to the sampled transitions
        # and backpropagate the mse loss to compute the gradients
        gradients = self.backward_fn(
            self.dqn.parameters, 
            states, yj, actions)
        
        # Update the policy gradients
        new_params = self.optimizer(self.dqn.parameters, gradients)
        self.dqn = self.dqn.update(new_params)
        
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
            self.target_params = self.dqn.parameters
    
    def take_action(self, state: JaxTensor) -> int:
        self.random_key, sk = jax.random.split(self.random_key)
        if jax.random.uniform(sk, shape=(1,)) < self.epsilon:
            action = jax.random.randint(sk, shape=(1,), minval=0, maxval=2)
        else:
            q_values = self.dqn(state, training=False)
            action = np.argmax(q_values)
        
        return int(action)

    def end_train(self):
        self.epsilon = 0
        self.training = False
        self.memory.memory.clear()