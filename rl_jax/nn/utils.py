
import jax
from ..typing import BackwardFn, Criterion, JaxModule, Parameter


def backward(model: JaxModule, 
             criterion: Criterion) -> BackwardFn:
    
    # Vectorize the forward function of the model in order
    # to work with batches of data
    vmodel = jax.vmap(model.forward_fn, in_axes=(None, 0))

    # Create the forward function using the vectorized model as forward step
    def forward_n_loss(params, x, y):
        preds = vmodel(params, x)
        return criterion(y, preds)
    
    # Differentiate the forward and loss function
    # Reverse gradients :)
    return jax.value_and_grad(forward_n_loss)