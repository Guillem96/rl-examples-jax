from functools import partial
from typing import Sequence, Tuple

import jax
import jax.numpy as np

from sklearn.datasets import fetch_openml

import rl_jax.nn as nn
from rl_jax.typing import JaxTensor, JaxModule, Criterion, BackwardFn

EPOCHS = 8
BATCH_SIZE = 32
STEPS_PER_EPOCH = 256


def accuracy(y_true: JaxTensor, y_pred: JaxTensor):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    n_instances = y_true.shape[0]
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


def backward(model: JaxModule, 
             criterion: Criterion) -> BackwardFn:
    # Vectorize the model
    vectorized_model = jax.vmap(model.forward_fn, 
                                in_axes=(None, 0))
    
    # Create the forward function using the vectorized model as forward step
    def forward_n_loss(params, x, y):
        preds = vectorized_model(params, x)
        return criterion(y, preds)

    # Differentiate the forward and loss function
    # Reverse gradients :)
    return jax.value_and_grad(forward_n_loss)
    

def main():
    key = jax.random.PRNGKey(0)

    print('Downloading data...')
    # Download MNIST data using scikit-learn fetch openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    # Convert data to numeric types
    X = X.astype('float32')
    y = y.astype('float32')

    # Train test split
    print('Generating train test splits...')
    x_train, x_test, y_train, y_test = train_test_split(key, X, y)

    # As we are dealing with a multi class classification problem
    # we have to one hot encode the labels
    y_train = nn.utils.one_hot(y_train, n_classes=10)
    y_test = nn.utils.one_hot(y_test, n_classes=10)

    # Define a sequential model
    # The model receives two kind of parameters
    #    - A random jax key to initialize the layers weights
    #    - A set of JaxPartialModules, JaxPartialModule is a partially
    #      evaluated function that when is called with a random key as
    #      a parameter, it returns a fully functional JaxModule
    model = nn.sequential(
        key,
        partial(nn.linear,
                in_features=28 * 28, 
                out_features=512, 
                activation=jax.nn.sigmoid),
        partial(nn.linear,
                in_features=512, 
                out_features=256, 
                activation=jax.nn.relu),
        partial(nn.linear,
                in_features=256, 
                out_features=128, 
                activation=jax.nn.relu),
        partial(nn.dropout, prob=.3),
        partial(nn.linear,
                in_features=128, 
                out_features=10, 
                activation=jax.nn.softmax),
    )

    # The model is a JaxModel
    # JaxModel has two attributes
    #    - forward_fn: A function is a differentiable function 
    #      defining how a set of parameters should
    #      operated in order to compute a tensor. For example, 
    #      the forward function of an `nn.linear` is 
    #      `lambda params, x: np.dot(params['W'], x) + params['bias']` 
    #    - parameters: A group of parameters or weights
    
    # Create a CriterionFunction, in this case we use a cross entropy
    # loss with a mean reduction over the batch
    criterion = partial(nn.ce, reduction='mean')

    # Define the backward step of model to compute the derivatives of the
    # error wtr of the model.parameters
    backward_fn = backward(model, criterion)
    backward_fn = jax.jit(backward_fn)

    # Create an optimizer to update the model parameters
    optimizer = nn.optim.simple_optimizer(learning_rate=1e-3)
    optimizer = jax.jit(optimizer)

    for epoch in range(EPOCHS):
        print(f'Epoch [{epoch}]')
        for step in range(STEPS_PER_EPOCH):
            # Sample a batch
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.randint(subkey, 
                                           shape=(BATCH_SIZE,),
                                           minval=0, maxval=x_train.shape[0])
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            # Compute the loss and the gradients
            loss, gradients = backward_fn(model.parameters, x_batch, y_batch)
            # Apply gradients to reduce the loss
            new_parameters = optimizer(model.parameters, gradients)
            
            # JaxModels are immutable, so when updating the model with
            # new parameters, we are creating a new instance of a JaxModel
            # with the new parameters
            model = model.update(new_parameters)

            if step % 20 == 0:
                print(f'Epoch [{epoch}] loss: {loss:.4f}')

    print('Evaluating...')
    
    y_pred = model(x_train, vectorize=True, training=False)
    train_acc = accuracy(y_train, y_pred)
    
    y_pred = model(x_test, vectorize=True, training=False)
    test_acc = accuracy(y_test, y_pred)

    print(f'Train Accuracy: {train_acc:.3f}')
    print(f'Test Accuracy: {test_acc:.3f}')


if __name__ == "__main__":
    main()