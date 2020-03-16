import abc
import copy
from typing import (Callable, Dict, NamedTuple, 
                    Union, Sequence, Tuple, Mapping)

import jax
import numpy as np


# Type to define Jax data structures
JaxTensor = Union[np.ndarray, 'jax.xla.DeviceArray']

# Type of modules parameters
# We define parameters as a dictionary where the key is the 
# parameter name, p.e W, and the value is the real parameter 
# value
Parameter = Dict[str, JaxTensor]

# Type signature for performing forward step with a set of
# parameters
# forward:: Parameter -> JaxTensor -> bool -> JaxTensor 
ForwardFn = Callable[[JaxTensor, bool], JaxTensor]

# Type signature for function in charge of performing the backward
# step (compute partial derivatives of the loss wtr the model parameters)
# and also compute the loss value
# back_fn :: Sequence[Parameter] -> JaxTensor -> JaxTensor -> (JaxTenor, Paramer)
BackwardFn = Callable[[Sequence[Parameter], JaxTensor, JaxTensor], 
                       Tuple[Union[float, JaxTensor], Parameter]]

# Type signature for loss functions
# criterion :: JaxTensor -> JaxTensor -> float
# criterion y_true y_pred = loss
Criterion = Callable[[JaxTensor, JaxTensor], float]

# Type signature for Activation Function
# A function that simply maps a tensor to another tensor
ActivationFn = Callable[[JaxTensor], JaxTensor]


def _compare_keys(d1, d2) -> bool:
    return all(k1 == k2 for k1, k2 in zip(d1, d2))


def _compare_list_of_params(params1, params2) -> bool:
    eq_len = len(params1) == len(params2)
    if not eq_len:
        return False
    
    for p1, p2 in zip(params1, params2):
        if isinstance(p1, dict) and not _compare_keys(p1, p2):
            return False
        elif isinstance(p1, list) and not _compare_list_of_params(p1, p2):
            return False
    return True


# "Super Type" that will include any object created
# by our API. Similar to a Layer in keras or an nn.Module in PyTorch
class JaxModule(abc.ABC):

    @property
    def parameters(self) -> Union[Parameter, Sequence[Parameter]]:
        """
        Return a dict of the trainable tensors used on the forward step
        """
        return dict()

    def update(self, parameters: Parameter) -> 'JaxModule':
        """
        Create a copy of the current instance and update the parameters
        with the incoming ones. The incoming parameters are the parameters
        retrieved from the `parameters` property.
        Meaning that we have the same data structure.

        Examples
        --------
        >>> class MyModule(JaxModule):
        ...   # ...
        ...   @property
        ...   def parameters(self):
        ...     return dict(W=np.ones((10, 10)), bias=np.ones(()))
        ...   def update(self, parameters):
        ...     instance = copy.deepcopy(self)
        ...     instance.W, instance.bias = parameters['W'], parameters['bias']
        """
        return copy.deepcopy(self)
    
    def init(self, random_key: JaxTensor):
        pass

    @abc.abstractmethod
    def forward(self, x: JaxTensor, training: bool):
        raise NotImplemented

    def __call__(self, x: JaxTensor, training: bool = True):
        return self.forward(x, training)

    def __getitem__(self, key: Union[str, int]) -> Union[Parameter, JaxTensor]:
        return self.parameters[key]

    def __eq__(self, other: 'JaxModule') -> bool:
        if isinstance(self.parameters, list):
            return _compare_list_of_params(self.parameters, other.parameters)
        elif isinstance(self.parameters, dict):
            return _compare_keys(self.parameters, other.parameters)
        else:
            return True