import jax


def backward(model, criterion):
    vmodel = jax.vmap(model, in_axes=(None, 0))

    def _backward(params, x, y):
        preds = vmodel(params, x)
        return criterion(y, preds)
    
    return jax.value_and_grad(_backward)