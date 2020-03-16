import copy
import functools
from typing import Union, Tuple

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
    def parameters(self) -> Parameter:
        return dict(W=self.W, bias=self.b)
    
    def update(self, parameters: Parameter) -> 'Linear':
        instance = copy.deepcopy(self)
        instance.W = parameters['W']
        instance.b = parameters['bias']
        return instance

    def forward(self, x: JaxTensor, training: bool = True):
        out = np.dot(self.W, x)
        if self.b is not None:
            out = out + self.b
        
        return self.activation(out)


class Dropout(JaxModule):
    """
    Creates a layer implementing the Dropout regularization

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
            Jax random key to randomly initialize the layer 
            internal random state
        """
        key, subkey = jax.random.split(random_key)
        self.random_key = subkey

    def forward(self, x: JaxTensor, training: bool = True):
        if not training:
            return x

        self.random_key, subkey = jax.random.split(self.random_key)

        keep_mask = jax.random.uniform(subkey, shape=x.shape) 
        keep_mask = (keep_mask > self.prob).astype('float32')
        return np.multiply(keep_mask, x)

    def __eq__(self, other) -> bool:
        return self.prob == other.prob


class Pool2D(JaxModule):

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]] = 2,
                 strides: Union[int, Tuple[int, int]] = 2,
                 padding: str = 'VALID'):
        
        self.kernel_size = _convert_to_tuple(kernel_size)
        self.strides = _convert_to_tuple(strides)
        self.padding = padding

        if self.padding not in {'VALID', 'SAME'}:
            raise ValueError('Padding must be either "VALID" or "SAME"')
        
        

class Convolution2D(JaxModule):
    """
    Creates a convolution layer

    Parameters
    ----------
    in_features: int
        Number of input image channels. Input images must have shape of
        (N, C, H, W)
    out_features: int
        Number of channels of the output feature map
    kernel_size: Tuple[int, int] or int
        Kernel dimensions in (H, W)
    strides: Tuple[int, int] or int, default 1
        Window strides of the convolution
    padding: str, wether VALID or SAME, default VALID
        If padding is same, the output feature map will have
        the same height and width as the input
    bias: bool, default True
    activation: ActivationFn, default lambda x: x
        Activation function at the end of the layer
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 strides: Union[int, Tuple[int, int]] = 1,
                 bias: bool = True,
                 padding: str = 'VALID',
                 activation: ActivationFn = lambda x: x):
        super(Convolution2D, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = _convert_to_tuple(kernel_size)
        self.strides = _convert_to_tuple(strides)
        self.padding = padding
        self.bias = bias
        self.activation = activation

        if self.padding not in {'VALID', 'SAME'}:
            raise ValueError('Padding must be either "VALID" or "SAME"')
        
        self.dn = jax.lax.conv_dimension_numbers(
            (1,) * 4, (1,) * 4, ('NCHW', 'IOHW', 'NCHW'))

    def init(self, key: JaxTensor):
        kernel_key, bias_key = jax.random.split(key)
        
        x_init = jax.nn.initializers.xavier_uniform()
        norm_init = jax.nn.initializers.normal()

        self.kernel = x_init(kernel_key, shape=(self.in_features, 
                                                self.out_features,
                                                *self.kernel_size))
        self.b = None if not self.bias else norm_init(bias_key, shape=())

    @property
    def parameters(self):
        return dict(kernel=self.kernel, bias=self.b)
    
    def update(self, parameters: Parameter) -> 'Convolution2D':
        instance = copy.deepcopy(self)
        instance.kernel = parameters['kernel']
        instance.b = parameters['bias']
        return instance

    def forward(self, x: JaxTensor, training: bool):
        x = x.reshape((-1, self.in_features, *x.shape[-2:]))
        
        feature_map = jax.lax.conv_general_dilated(
            x, self.kernel, self.strides, self.padding, (1, 1), (1, 1), self.dn) 
        
        if self.b is not None:
            feature_map = feature_map + self.b

        return self.activation(feature_map)


def _convert_to_tuple(t, elems: int = 2) -> Tuple:
    if isinstance(t, tuple):
        return t
    else:
        return (t,) * elems


register_jax_module(Linear)
register_jax_module(Dropout)
register_jax_module(Convolution2D)
