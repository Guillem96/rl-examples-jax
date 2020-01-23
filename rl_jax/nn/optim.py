from typing import Callable, Sequence
from ..typing import JaxModule, Parameter


def simple_optimizer(
    learning_rate: float = 1e-3) -> Callable[[JaxModule, Sequence[Parameter]], 
                                             JaxModule]:

    def update(params, gradients):
        for i in range(len(params)):
            for k in gradients[i]:
                params[i][k] = params[i][k] - learning_rate * gradients[i][k]
        return params

    return update
