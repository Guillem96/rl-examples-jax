from functools import partial

import jax
import jax.numpy as np

import rl_jax.nn as nn


key = jax.random.PRNGKey(0)

# Create the model
model = nn.sequential(
    key,
    nn.linear(in_features=128, out_features=64, activation=jax.nn.relu),
    nn.linear(in_features=64, out_features=32, activation=jax.nn.relu),
    nn.linear(in_features=32, out_features=1, activation=jax.nn.sigmoid))

# Select the function to minimize
criterion = partial(nn.bce, reduction='mean')

# Using the model and the criterion we create a function to compute
# both, the loss and gradients
backward_fn = nn.backward(model, criterion)

optimizer = nn.optim.simple_optimizer(learning_rate=1e-2)
optimizer = jax.jit(optimizer)

# Random inputs
# Train a network to always output ones
y_true = np.ones((10,))

for i in range(100):
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(10, 128))

    value, gradients = backward_fn(model.parameters, x, y_true)
    if i % 5 == 0:
        print('Loss:', value)

    parameters = optimizer(model.parameters, gradients)
    model = model.update(parameters)

key, subkey = jax.random.split(key)
x = jax.random.normal(subkey, shape=(128,))
print(model.forward_fn(model.parameters, x))