import functools
from typing import Type

import jax
import jax.numpy as np
from ..typing import Parameter, JaxTensor, JaxModule


def one_hot(x: JaxTensor, n_classes: int) -> JaxTensor:
    classes = np.arange(n_classes)
    return np.array(x[:, None] == classes).astype(x.dtype)


def register_jax_module(jax_module: Type[JaxModule]):
    jax.tree_util.register_pytree_node(
        jax_module,
        lambda instance: (instance.parameters, instance),
        lambda instance, xs: instance.update(xs))