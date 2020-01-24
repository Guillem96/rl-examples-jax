import functools
from typing import Callable, Sequence

import jax
from ..typing import (JaxModule, JaxTensor, 
                      ForwardFn, Parameter, PartialJaxModule)


class JaxModel(JaxModule):
    
    def __call__(self, 
                 x: JaxTensor, 
                 training: bool = True,
                 vectorize: bool = False):
        # Use this function during inference
        fn = functools.partial(self.forward_fn, training=training)

        if vectorize:
          fn = jax.vmap(fn, in_axes=(None, 0))
        
        return fn(self.parameters, x)
    
    def update(self, parameters: Sequence[Parameter]):
        return JaxModel(forward_fn=self.forward_fn, 
                        parameters=parameters)


def _sequential_forward(parameters: Sequence[Parameter], 
                        x: JaxTensor,
                        training: bool = True,
                        forward_fns: ForwardFn = None) -> JaxTensor:
    
    for p, f in zip(parameters, forward_fns):
        x = f(p, x, training=training)
    
    return x


def sequential(key, *layers: Sequence[PartialJaxModule]) -> JaxModel:
    # Randomly initialize layers
    layer_keys = jax.random.split(key, num=len(layers))
    layers = [l(k) for l, k in zip(layers, layer_keys)]

    parameters = [l.parameters for l in layers]
    forward_fns = [l.forward_fn for l in layers]
    model_forward_fn = functools.partial(_sequential_forward, 
                                         forward_fns=forward_fns)
   
    return JaxModel(parameters=parameters, 
                    forward_fn=model_forward_fn)
