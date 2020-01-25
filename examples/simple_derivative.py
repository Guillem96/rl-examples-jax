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