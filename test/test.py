import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import jax


def read_iris(path: str):
    data = []

    with open(path) as f:
        # Skip header
        lines = f.readlines()[1:]
        for l in lines:
            data.append(l.split(','))
    return data

label2idx = dict(setosa=0, versicolor=1, virginica=2)

def to_np(data):
    tensor_data = np.zeros((len(data), len(data[0])))

    for i, d in enumerate(data):
        features, label = list(map(float, d[:-1])), d[-1].strip()
        label = label2idx[label]
        features.append(label)
        tensor_data = jax.ops.index_update(tensor_data, i, np.array(features))

    return tensor_data


def train_test_split(key,
                     data, 
                     train_prob: float = .8):
    train_size = int(data.shape[0] * train_prob)
    key, subkey = jax.random.split(key)
    x = jax.random.shuffle(subkey, data, axis=0)
    return x[:train_size], x[train_size:]


def accuracy(model, params, x, y):
    y_pred = model(params, x)[:, 0] > .5
    y_pred = y_pred.astype('int32')
    print(y_pred)
    print(y)
    print('-' * 80)
    n_instances = y.shape[0]
    true_instances = np.sum(y_pred == y)
    return true_instances / n_instances


def main():
    key = jax.random.PRNGKey(0)

    # Preprocess the data
    data = read_iris('test/data/iris.csv')
    data = to_np(data)
    train_data, test_data = train_test_split(key, data)

    x_train, x_test = train_data[:, :-1], test_data[:, :-1]
    y_train, y_test = train_data[:, -1], test_data[:, -1]
    y_train = (y_train == 0).astype('int32')
    y_test = (y_test == 0).astype('int32')

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    # Initialize all layers for a fully-connected neural network with sizes "sizes"
    def init_network_params(sizes, key):
        keys = random.split(key, len(sizes))
        return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    layer_sizes = [4, 512, 1]
    param_scale = 0.1
    step_size = 0.0001
    params = init_network_params(layer_sizes, random.PRNGKey(0)) 
    from jax.scipy.special import logsumexp

    def relu(x):
        return np.maximum(0, x)

    def predict(params, image):
        # per-example predictions
        activations = image
        for w, b in params[:-1]:
            outputs = np.dot(w, activations) + b
            activations = relu(outputs)
        
        final_w, final_b = params[-1]
        logits = np.dot(final_w, activations) + final_b
        return jax.nn.sigmoid(logits)
    
    batched_predict = vmap(predict, in_axes=(None, 0))

    def one_hot(x, k, dtype=np.float32):
        """Create a one-hot encoding of x of size k."""
        return np.array(x[:, None] == np.arange(k), dtype)

    def loss(params, images, targets):
        preds = batched_predict(params, images)
        pt = np.where(targets == 1, preds + 1e-6, 1 - preds + 1e-6)
        loss = -np.log(pt)
        return np.mean(loss)

    @jit
    def update(params, x, y):
        grads = grad(loss)(params, x, y)
        return [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params, grads)]
    

    for epoch in range(3):
        print(f'Epoch [{epoch}]')
        for step in range(100):
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.randint(subkey, 
                                           shape=(8,),
                                           minval=0, maxval=x_train.shape[0])
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            params = update(params, x_batch, y_batch)

            # if step % 20 == 0:
            #     print(f'Epoch [{epoch}] loss: {loss:.4f}')

    print('Evaluating...')

    train_acc = accuracy(batched_predict, params, x_train, y_train)
    test_acc = accuracy(batched_predict, params, x_test, y_test)

main()