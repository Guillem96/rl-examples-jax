from functools import partial
from typing import Sequence, Tuple

import jax
import jax.numpy as np

from sklearn.datasets import fetch_openml

import rl_jax.nn as nn
from rl_jax.typing import JaxTensor

EPOCHS = 5
BATCH_SIZE = 32
STEPS_PER_EPOCH = 256


def accuracy(model, x, y):
    y_pred = model(x, vectorize=True, training=False)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y, axis=1)

    n_instances = y.shape[0]
    true_instances = np.sum(y_pred == y_true)

    return true_instances / n_instances


def train_test_split(key: JaxTensor,
                     x: JaxTensor, y: JaxTensor):

    data = np.concatenate(
        [x, np.expand_dims(y, axis=-1)], axis=-1)[:5000]
    # print('Shuffling data...')
    # data = jax.random.shuffle(key, data, axis=0)
    train_size = int(data.shape[0] * .8)
    
    train = data[:train_size]
    test = data[train_size:]

    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    
    return x_train, x_test, y_train, y_test


def main():
    key = jax.random.PRNGKey(0)

    print('Downloading data...')
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    X = X.astype('float32')
    y = y.astype('float32')

    print('Generating train test splits...')
    x_train, x_test, y_train, y_test = train_test_split(key, X, y)

    print(y_train)
    y_train = nn.utils.one_hot(y_train, n_classes=10)
    y_test = nn.utils.one_hot(y_test, n_classes=10)

    # mean = np.mean(x_train, axis=0)
    # std = np.std(x_train, axis=0)
    # x_train = (x_train - mean) / std
    # x_test = (x_test - mean) / std

    # Define the model
    model = nn.sequential(
        key,
        partial(nn.linear,
                in_features=28 * 28, 
                out_features=512, 
                activation=jax.nn.sigmoid),
        # partial(nn.dropout, prob=.5),
        partial(nn.linear,
                in_features=512, 
                out_features=256, 
                activation=jax.nn.relu),
        # partial(nn.dropout, prob=.5),
        partial(nn.linear,
                in_features=256, 
                out_features=128, 
                activation=jax.nn.relu),
        partial(nn.linear,
                in_features=128, 
                out_features=10, 
                activation=jax.nn.softmax),
    )
    
    criterion = partial(nn.ce, reduction='mean')

    backward_fn = nn.backward(model, criterion)
    backward_fn = jax.jit(backward_fn)

    optimizer = nn.optim.simple_optimizer(learning_rate=1e-3)
    optimizer = jax.jit(optimizer)

    for epoch in range(EPOCHS):
        print(f'Epoch [{epoch}]')
        for step in range(STEPS_PER_EPOCH):
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.randint(subkey, 
                                           shape=(BATCH_SIZE,),
                                           minval=0, maxval=x_train.shape[0])
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            loss, gradients = backward_fn(model.parameters, x_batch, y_batch)
            new_parameters = optimizer(model.parameters, gradients)
            model = model.update(new_parameters)

            if step % 20 == 0:
                print(f'Epoch [{epoch}] loss: {loss:.4f}')

    print('Evaluating...')

    train_acc = accuracy(model, x_train, y_train)
    test_acc = accuracy(model, x_test, y_test)

    print(f'Train Accuracy: {train_acc:.3f}')
    print(f'Test Accuracy: {test_acc:.3f}')


if __name__ == "__main__":
    main()