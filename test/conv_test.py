from functools import partial

import jax
import jax.numpy as np

import rl_jax.nn as nn
from rl_jax.typing import JaxModule, JaxTensor, Criterion


key = jax.random.PRNGKey(0)

model = nn.Sequential(
    nn.Convolution2D(3, 32, 3, padding='VALID', activation=jax.nn.relu),
    nn.Convolution2D(32, 3, 3, padding='VALID', activation=jax.nn.relu))
model.init(key)

model_fn = jax.vmap(model)
key, subkey = jax.random.split(key)
rand_im = jax.random.normal(subkey, shape=(5, 3, 256, 256))
print(model(rand_im) == model_fn(rand_im).reshape((5, 3, 252, 252)))
