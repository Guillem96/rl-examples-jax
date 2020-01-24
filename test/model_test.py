from functools import partial

import jax
import jax.numpy as np

import rl_jax.nn as nn


key = jax.random.PRNGKey(0)

# Create the model
# Model recieves two type of parameters:
#  - A random key to initialize the partially evaluated layers
#  - A sequence of PartialJaxModules
# A PartialJaxModule is just a partially evaluated function which the only
# left parameter is a random key
# In this case nn.sequential will take care of spliting the key according to
# JAX good practices and initialize the PartialJaxModules
model = nn.sequential(
    key,
    partial(nn.linear, 
            in_features=128, 
            out_features=64, 
            activation=jax.nn.relu),
    partial(nn.linear, 
            in_features=64, 
            out_features=32, 
            activation=jax.nn.relu),
    partial(nn.linear, 
            in_features=32, 
            out_features=1, 
            activation=jax.nn.sigmoid))

# Select the function to minimize
# Now, criterion is a Criterion function, meaning that it computes
# a loss value given the ground truth and model predictions
criterion = partial(nn.bce, reduction='mean')

# Using the model and the criterion we create a function to compute
# both, the loss and gradients
# A BackwardFunction will take care of vectorizing the model and convert the
# Criterion to return the reverse-gradients
backward_fn = nn.backward(model, criterion)

# simple_optimizer creates a function that recieves two set of parameters
# the first one is the actual model's parameters, and the second one are 
# the gradients of each parameter
# The optimizer will take care of updating all model's parameters
optimizer = nn.optim.simple_optimizer(learning_rate=1e-2)
optimizer = jax.jit(optimizer) # Compile the optimizer

# Random inputs
# Train a network to always output ones
y_true = np.ones((10,))

for i in range(100):
    # Generate a random input
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(10, 128))
    
    # Apply backward function with the input and the expected output
    loss, gradients = backward_fn(model.parameters, x, y_true)
    
    # Report the loss
    if i % 5 == 0:
        print('Loss:', loss)
    
    # get the parameters that reduce the loss and update the model
    # with the new parameters
    parameters = optimizer(model.parameters, gradients)
    model = model.update(parameters)

  # Check if the model outputs a value close to one
key, subkey = jax.random.split(key)
x = jax.random.normal(subkey, shape=(128,))
print(model(x))
