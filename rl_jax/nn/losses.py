from typing import Union

import jax
import jax.numpy as np

from ..typing import JaxTensor


_CriterionOut = Union[JaxTensor, float]

def bce(y_true: JaxTensor, 
        y_pred: JaxTensor, 
        reduction: str = None) -> _CriterionOut:
    y_pred = y_pred.reshape(-1)
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)

    y_true = y_true.astype('float32')

    # loss = y_true * np.log(y_pred + 1e-6)
    # loss += (1 - y_true) * np.log(1 - y_pred + 1e-6)
    # loss = -loss

    pt = np.where(y_true == 1, y_pred + 1e-6, 1 - y_pred + 1e-6)
    loss = -np.log(pt)
    
    if reduction is None or reduction == 'none':
        return loss
    elif reduction == 'sum':
        return np.sum(loss)
    elif reduction == 'mean':
        # sum_over_batch = np.sum(loss, axis=-1)
        return np.mean(loss)
    else:
        raise ValueError('Unexpected reduction type')


def ce(y_true: JaxTensor, 
       y_pred: JaxTensor, 
       reduction: str = None) -> _CriterionOut:

    """
    Computes cross entropy

    Parameters
    ----------
    y_true: JaxTensor of shape (N, C)
        One hot encoded tensor of the true labels. C is the number
        of classes
    y_pred: JaxTensor of shape (N, C)
        Probability distribution of the predicted labels
    
    Examples
    --------
    >>> target = jax.random.randint(key, shape=(10,), maxval=3, minval=0)
    >>> target = rl_jax.nn.utils.one_hot(target, 3)
    >>> predictions = jax.nn.softmax(jax.random.uniform(key, shape=(10, 3)))
    >>> loss = rl_jax.nn.ce(target, predictions, reduction='mean')
    """
    loss = y_true * -np.log(y_pred)

    if reduction is None or reduction == 'none':
        return loss
    elif reduction == 'sum':
        return np.sum(loss)
    elif reduction == 'mean':
        sum_over_batch = np.sum(loss, axis=-1)
        return np.mean(sum_over_batch)
    else:
        raise ValueError('Unexpected reduction type')