from ..typing import JaxTensor


def simple_optimizer(param: JaxTensor, 
                     gradient: JaxTensor, 
                     learning_rate: float = 1e-3) -> JaxTensor:
    """
    Creates a simples optimizer that given the parameters and 
    the respegdive gradients apply the following update:
        parameter' = parameter - gradient * learning_rate
    
    Parameters
    ----------
    param: JaxTensor
        Parameter to be updated
    gradient: JaxTensor
        Gradient to be subtracted
    learning_rate: float, default 1e-3
        Step size to modify the parameters
    
    Returns
    -------
    JaxTensor
        The updated parameter
    """
    return param - learning_rate * gradient
