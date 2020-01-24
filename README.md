# Reinforcement learning with Jax

Simple reinforcement learning examples with [JAX](https://github.com/google/jax).

The aim of this repository is to provide simple examples on how to develop Deep Reinforcement learning using JAX.

## Intuitive NN interface

To make life easier, I developed a high-level API to prototype NN architectures. The API is thought to be simple and intuitive, but it requires a mid-level understanding of higher-order functions.
This is because JAX is pretty functional and develop extension over it using classes or objects requires complex workarounds.

```python
# Example: How to create a linear binary classifier
from functools import partial
import jax
import rl_jax.nn as nn

key = key = jax.random.PRNGKey(0)

model = nn.sequential(
    key,
    partial(nn.linear, 
            in_features=32, 
            out_features=1, 
            activation=jax.nn.sigmoid))

key, subkey = jax.random.split(key)
x_input = jax.random.uniform(subkey, shape=(32,))
prediction = model(x_input)
```

## References

- [JAX](http://github.com/google/jax) - Composable Transformations of Python+NumPy programs 2018
