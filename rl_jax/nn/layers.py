import copy
import functools

import jax
import jax.numpy as np
import numpy as onp

from .utils import register_jax_module
from ..typing import (ActivationFn, JaxTensor, 
                      JaxModule, Parameter)


class Linear(JaxModule):
    """
    Creates a JaxModule corresponding to a linear layer
    
    Parameters
    ----------
    in_features: int
        Input feature size
    out_features: int
        Number of features after the linear transformation
    activation: ActivationFn, default a linear activation (lambda x: x)
        Activation to apply after the linear transformation. e.g `jax.nn.relu`, `jax.nn.sigmoid`...
    bias: bool, default True
        Whether or not the linear layer has to add bias
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 activation: ActivationFn = lambda x: x):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation = activation

    def init(self, random_key: JaxTensor):
        """
        Parameters
        ----------
        key: JaxTensor
            Jax random key to randomly initialize the layer weights
        """
        W_key, b_key = jax.random.split(random_key)

        x_init = jax.nn.initializers.xavier_uniform() 
        norm_init = jax.nn.initializers.normal()

        self.W = x_init(W_key, shape=(self.out_features, self.in_features))
        self.b = None if not self.bias else norm_init(b_key, shape=())
    
    @property
    def parameters(self):
        return dict(W=self.W, bias=self.b)
    
    def update(self, parameters):
        instance = copy.deepcopy(self)
        instance.W = parameters['W']
        instance.b = parameters['bias']
        return instance

    def forward(self, x: JaxTensor, training: bool = True):
        out = np.dot(self.W, x)
        if self.b is not None:
            out = out + self.b
        
        return self.activation(out)
    
    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False

        return all(k1 == k2 
                    for k1, k2 in zip(self.parameters, other.parameters))


class Dropout(JaxModule):
    """
    Creates a JaxModule corsponding to a linear layer
    
    Parameters
    ----------
    prob: float, default .5
    """
        

    def __init__(self, prob: float = .5):
        super(Dropout, self).__init__()
        self.prob = prob
        self.random_key = None

    def init(self, random_key: JaxTensor):
        """
        Parameters
        ----------
        key: JaxTensor
            Jax random key to randomly initialize the layer weights
        """
        key, subkey = jax.random.split(random_key)
        self.random_key = subkey
    
    @property
    def parameters(self):
        return []

    def update(self, parameters):
        return copy.deepcopy(self)

    def forward(self, x: JaxTensor, training: bool = True):
        if not training:
            return x

        self.random_key, subkey = jax.random.split(self.random_key)

        keep_mask = jax.random.uniform(subkey, shape=x.shape) 
        keep_mask = (keep_mask > self.prob).astype('float32')
        return np.multiply(keep_mask, x)

    def __eq__(self, other) -> bool:
        return self.prob == other.prob


register_jax_module(Linear)
register_jax_module(Dropout)