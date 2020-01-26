from typing import Callable, Sequence
from ..typing import JaxModule, Parameter


def simple_optimizer(
    learning_rate: float = 1e-3) -> Callable[[JaxModule, JaxModule], 
                                             JaxModule]:

    def update(module: JaxModule, module_grads: JaxModule) -> JaxModule:
        params = module.parameters
        gradients = module_grads.parameters

        if isinstance(gradients, list):
            # In case parameters is a list apply
            # the update to each element of the list
            for i in range(len(gradients)):
                for k in gradients[i]:
                    params[i][k] = params[i][k] - learning_rate * gradients[i][k]
        else:
            # Otherwise we only have to update a single
            # set of parameters
            for k in gradients:
                params[k] = params[k] - learning_rate * gradients[k]
        
        return module.update(params)

    return update
