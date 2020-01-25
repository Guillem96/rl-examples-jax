
import jax
import jax.numpy as np
from ..typing import Parameter


def one_hot(x: JaxTensor, n_classes: int) -> JaxTensor:
    classes = np.arange(n_classes)
    return np.array(x[:, None] == classes).astype(x.dtype)