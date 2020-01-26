from typing import Callable, Sequence
from ..typing import JaxModule, Parameter


def _optimize_single_param(param: Parameter,
                           gradient: Parameter,
                           learning_rate: float) -> Parameter:

    # Update a single parameter
    return {k: param[k] - gradient[k] * learning_rate for k in param}
    

def _optimize_list(params_list: Sequence[Parameter], 
                   gradients: Sequence[Parameter],
                   learning_rate: float) -> Sequence[Parameter]:
    # Optimizes a sequence of parameters. A sequence of parameters
    # can contain nested sequences, in this case, we recursively iterate over
    # them
    updated_parameters = []
    for i in range(len(params_list)):
        if isinstance(params_list[i], list):
            new_param = _optimize_list(
                params_list[i], 
                gradients[i], 
                learning_rate) 
        else:
            new_param = _optimize_single_param(
                params_list[i], 
                gradients[i], 
                learning_rate)

        updated_parameters.append(new_param)
    return updated_parameters


def simple_optimizer(
    learning_rate: float = 1e-3) -> Callable[[JaxModule, JaxModule], 
                                             JaxModule]:
    """
    Creates a simples optimizer that given the parameters and 
    the respegdive gradients apply the following update:
        parameter' = parameter - gradient * learning_rate
    
    Parameters
    ----------
    learning_rate: float, default 1e-3
        Step size to modify the parameters
    
    Returns
    -------
    A function of type Callable[[JaxModule, JaxModule], JaxModule]
    JaxModule -> JaxModule -> JaxModule
    """
    def update(module: JaxModule, 
               module_grads: JaxModule) -> JaxModule:
        params = module.parameters
        gradients = module_grads.parameters
        if isinstance(params, dict):
            new_parameters = _optimize_single_param(
                params, gradients, learning_rate)
        else:
            new_parameters = _optimize_list(
                params, gradients, learning_rate)

        return module.update(new_parameters)

    return update
