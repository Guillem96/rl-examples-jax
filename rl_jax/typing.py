from typing import Callable, Union

import jax
import numpy as np


# Type to define Jax data structures
JaxTensor = Union[np.ndarray, 'jax.xla.DeviceArray']

# Type signature for loss functions
# criterion :: JaxTensor -> JaxTensor -> float
# criterion y_true y_pred = loss
Criterion = Callable[[JaxTensor, JaxTensor], float]
