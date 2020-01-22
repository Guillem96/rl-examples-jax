import jax


def simple_optimizer(learning_rate: float = 1e-3):

    def update(params, gradients):
        for i in range(len(params)):
            for k in gradients[i]:
                params[i][k] = params[i][k] - learning_rate * gradients[i][k]
        return params

    return jax.jit(update)
