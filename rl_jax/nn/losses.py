import jax
import jax.numpy as np


def bce(y_true, y_pred, reduction: str = None):
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    loss = -np.log(pt)
    
    if reduction is None or reduction == 'none':
        return loss
    elif reduction == 'sum':
        return np.sum(loss)
    elif reduction == 'mean':
        sum_over_batch = np.sum(loss, axis=-1)
        return np.mean(sum_over_batch)
    else:
        raise ValueError('Unexpected reduction type')