from typing import Callable, Sequence
from ..typing import JaxModule, Parameter


def simple_optimizer(
    learning_rate: float = 1e-3) -> Callable[[Parameter, Parameter], 
                                             Parameter]:

    def update(params, gradients):
        if isinstance(params, list):
            # In case parameters is a list apply
            # the update to each element of the list
            for i in range(len(params)):
                for k in gradients[i]:
                    params[i][k] = params[i][k] - learning_rate * gradients[i][k]
            return params
        else:
            # Otherwise we only have to update a single
            # set of parameters
            for k in gradients:
                params[k] = params[k] - learning_rate * gradients[k]
            return params

    return update
