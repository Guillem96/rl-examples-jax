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