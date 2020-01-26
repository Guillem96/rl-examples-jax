import abc
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
ForwardFn = Callable[[Parameter, JaxTensor, bool], JaxTensor]

# Type signarure for function in charge of performing the backward
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

# "Super Type" that will include any object created
# by our API. Similar to a Layer in keras or an nn.Module in PyTorch
class JaxModule(abc.ABC):

    @abc.abstractproperty
    def parameters(self) -> Union[Parameter, Sequence[Parameter]]:
        raise NotImplemented
    
    @abc.abstractmethod
    def init(self, random_key: JaxTensor):
        raise NotImplemented

    @abc.abstractmethod
    def forward(self, x: JaxTensor, training: bool):
        raise NotImplemented

    def __call__(self, x: JaxTensor, training: bool = True):
        return self.forward(x, training)

    def __getitem__(self, key: Union[str, int]) -> Union[Parameter, JaxTensor]:
        return self.parameters[key]

    def __eq__(self, other: 'JaxModule') -> bool:
        return len(self.parameters) == len(other.parameters)

