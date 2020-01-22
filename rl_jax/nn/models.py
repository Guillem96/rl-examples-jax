from typing import Callable, Sequence

import jax


class Sequential(list):

    def __init__(self, key, *layers):
        layer_keys = jax.random.split(key, num=len(layers))
        self.layers = layers
        for k, l in zip(layer_keys, self.layers):
            l.init(k)
    
    @property
    def parameters(self):
        return [l.parameters for l in self.layers]
    
    @parameters.setter
    def parameters(self, val):
        for l, p in zip(self.layers, val):
            l.parameters = p

    def __call__(self, params: Sequence[dict], x):
        for l, p in zip(self.layers, params):
            x = l(p, x)
        return x