import abc

import jax
import jax.numpy as np

class Layer(abc.ABC):

    def __init__(self):
        self.parameters = dict()

    @abc.abstractmethod
    def init(self, key):
        raise NotImplemented

    @abc.abstractmethod
    def __call__(self, params, x):
        raise NotImplemented

    def add_param(self, **params):
        self.parameters = dict(**self.parameters, **params)


class Linear(Layer):
    
    def __init__(self,
                 in_features: int,
                 out_features: int, 
                 activation = lambda x: x,
                 bias: bool = True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.activation = activation

    def init(self, key):
        W_key, b_key = jax.random.split(key, num=2)
        
        x_init = jax.nn.initializers.xavier_uniform() 
        norm_init = jax.nn.initializers.normal()
        
        W = x_init(W_key, shape=(self.out_features, self.in_features))
        b = None if not self.has_bias else norm_init(b_key, shape=())

        self.add_param(W=W, b=b)

    def __call__(self, params: dict, x: jax.xla.DeviceArray):
        # x [BATCH, IN_FEATURES]
        # W [OUT_FEATURES, IN_FEATURES]
        W = params['W']
        b = params['b']

        out = np.dot(W, x)
        if b is not None:
            out = out + b
        
        return self.activation(out)

