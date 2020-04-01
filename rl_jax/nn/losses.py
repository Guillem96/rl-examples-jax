from typing import Union

import jax
import jax.numpy as np

from ..typing import JaxTensor


_CriterionOut = Union[JaxTensor, float]

def bce(y_true: JaxTensor, 
        y_pred: JaxTensor, 
        reduction: str = None) -> _CriterionOut:
    """
    Computes the binary cross entropy
    
    Parameters
    ----------
    y_true: JaxTensor of shape (N,)
        Ground truths
    y_pred: JaxTensor of shape (N,)
        The probability of every prediction of belonging to the true 
        class
    reduction: str, 'sum', 'none' or 'mean'
        How to group the loss
    """
    # Clip the y_pred with an epsilon value to avoid nan when 
    # applying the log
    epsilon = 1e-6
    y_pred = y_pred.reshape(-1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    y_true = y_true.astype('float32')

    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    loss = -np.log(pt)
    
    if reduction is None or reduction == 'none':
        return loss
    elif reduction == 'sum':
        return np.sum(loss)
    elif reduction == 'mean':
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
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
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


def mse(y_true: JaxTensor, 
        y_pred: JaxTensor, 
        reduction: str = None) -> _CriterionOut:
    
    loss = np.square(y_true - y_pred)

    if reduction is None or reduction == 'none':
        return loss
    elif reduction == 'sum':
        return np.sum(loss)
    elif reduction == 'mean':
        sum_over_batch = np.sum(loss, axis=-1)
        return np.mean(sum_over_batch)
    else:
        raise ValueError('Unexpected reduction type')
