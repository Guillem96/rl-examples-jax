import functools

import jax
import jax.numpy as np

from ..typing import (ActivationFn, JaxTensor, 
                      JaxModule, Parameter)


def _linear_forward(params: Parameter, 
                    x: JaxTensor,
                    activation: ActivationFn) -> JaxTensor:
    W = params['W']
    b = params['bias']

    out = np.dot(W, x)
    if b is not None:
        out = out + b
    
    return activation(out)


def linear(key: JaxTensor,
           in_features: int,
           out_features: int, 
           activation: ActivationFn = lambda x: x,
           bias: bool = True) -> JaxModule:
    """
    Creates a JaxModule corsponding to a linear layer
    
    Parameters
    ----------
    key: JaxTensor
        Jax random key to randomly initialize the layer weights
    in_features: int
        Input feature size
    out_features: int
        Number of features after the linear transformation
    activation: ActivationFn, default a lniear activation (lambda x: x)
        Activation to apply after the linear transformation. e.g `jax.nn.relu`, `jax.nn.sigmoid`...
    bias: bool, default True
        Whether or not the linear layer has to add bias
    
    Returns
    -------
    JaxModule
    """
    W_key, b_key = jax.random.split(key)

    x_init = jax.nn.initializers.xavier_uniform() 
    norm_init = jax.nn.initializers.normal()

    W = x_init(W_key, shape=(out_features, in_features))
    b = None if not bias else norm_init(b_key, shape=())
    params = dict(W=W, bias=b)
    forward_fn = functools.partial(_linear_forward, 
                                activation=activation)

    return JaxModule(parameters=params, 
                    forward_fn=forward_fn)
  
