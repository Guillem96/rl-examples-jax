from functools import partial
from typing import Sequence

import jax
import jax.numpy as np

import rl_jax.nn as nn
from rl_jax.typing import JaxTensor


label2idx = dict(setosa=0, versicolor=1, virginica=2)


def read_iris(path: str):
    data = []

    with open(path) as f:
        # Skip header
        lines = f.readlines()[1:]
        for l in lines:
            data.append(l.split(','))
    return data


def to_np(data: Sequence[Sequence[str]]) -> JaxTensor:
    tensor_data = np.zeros((len(data), len(data[0])))

    for i, d in enumerate(data):
        features, label = list(map(float, d[:-1])), d[-1].strip()
        label = label2idx[label]
        features.append(label)
        tensor_data = jax.ops.index_update(tensor_data, i, np.array(features))

    return tensor_data

key = jax.random.PRNGKey(0)

# Preprocess the data
data = read_iris('test/data/iris.csv')
data = to_np(data)
data = jax.random.shuffle(key, data, axis=0)
x = data[:, :-1]
y = data[:, -1].astype('int32')

# Define the model
model = nn.sequential(
    nn.linear(in_features=4, 
              out_features=65, 
              activation=jax.nn.relu),
    nn.linear(in_features=64, 
              out_features=64, 
              activation=jax.nn.relu),
    nn.linear(in_features=64, 
              out_features=32, 
              activation=jax.nn.relu),
    nn.linear(in_features=32, 
              out_features=3, 
              activation=jax.nn.softmax)
)

criterion = partial(nn.bce, reduction='mean')
backward_fn = nn.backward(model, criterion)
