import copy
import functools
from typing import Callable, Sequence

import jax
from .utils import register_jax_module
from ..typing import (JaxModule, JaxTensor, ForwardFn, Parameter)


class Sequential(JaxModule):
    """
    Parameters
    ----------
    layers: Sequence[JaxModules]
        All the modules compoising the sequential model. The modules should
        not have been initialized.
    """
    def __init__(self, 
                 *layers: Sequence[JaxModule]):
        self.layers = layers

    def init(self, random_key: JaxTensor):
        """
        Parameters
        ----------
        key: JaxTensor
            Jax random key to randomly initialize the layer weights
        """
        layer_keys = jax.random.split(random_key, num=len(self.layers))
        for l, k in zip(self.layers, layer_keys):
            l.init(k)
    
    @property
    def parameters(self) -> Sequence[Parameter]:
        return [l.parameters for l in self.layers]

    def update(self, parameters: Sequence[Parameter]):
        instance = copy.deepcopy(self)
        instance.layers = [l.update(p) 
                           for l, p in zip(self.layers, 
                                           parameters)]
        return instance

    def forward(self, x: JaxTensor, training: bool = True):
        for l in self.layers:
            x = l(x, training=training)
    
        return x
    
    def __eq__(self, other: 'Sequential') -> bool:
        if not super().__eq__(other):
            return False
        
        return all(l1 == l2 for l1, l2 in zip(self.layers, other.layers))


register_jax_module(Sequential)