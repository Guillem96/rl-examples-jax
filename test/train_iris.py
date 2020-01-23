import jax
import jax.numpy as np


label2idx = dict(setosa=0, versicolor=1, virginica=2)


def read_iris(path: str):
    data = []

    with open(path) as f:
        # Skip header
        lines = f.readlines()[1:]
        
        for l in lines:
            data.append(l.split(','))
    
    return data


def to_np(data):
    x = np.zeros((len(data), 4))
    y = np.zeros((len(data)))

    for i, d in enumerate(data):
        features, label = list(map(float, d[:-1])), d[-1].strip()
        x = jax.ops.index_update(x, i, np.array(features))
        y = jax.ops.index_update(y, i, label2idx[label])

    return x, y


print(to_np(read_iris('test/data/iris.csv'))) 
