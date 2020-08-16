import tensorflow as tf
from util import assert_shape


"""All transformers interleave two sub layers: attention and a transition
function. This is the collection of transition functions.

TODO: add an LSTM transition function here.
"""


class FFNTransition(tf.keras.layers.Layer):
    def __init__(self, layer_sizes, reg=None):
        """Create a transformer transition function consisting of a stack of
        fully connected layers with relu activations in between, but NO
        activation for the last layer. Typically there are only two layers
        here.

        Args:
            layer_sizes (list of int): Output sizes of the inner layers, but
                                       EXCLUDING the last layer. For example,
                                       if this is an empty list, this
                                       transition function is just an affine
                                       transformation.
            reg: A tf.keras regulariser for kernel weights (optional).
        """
        super(FFNTransition, self).__init__()
        assert (isinstance(layer_sizes, list))
        self.layer_sizes = layer_sizes
        self.layers = None
        self.reg = reg

    def build(self, input_shape):
        assert(len(input_shape) == 3)
        self.input_dim = input_shape[-1]
        assert(isinstance(self.input_dim, int))

        self.layers = [tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(N, activation='relu',
                                  kernel_regularizer=self.reg))
                       for N in self.layer_sizes]

        # the last layer has no activation and automatically has output shape
        # equal to the input dimension
        self.layers.append(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.input_dim, activation=None,
                                  kernel_regularizer=self.reg)))

    def call(self, input):
        assert_shape(input, [None, None, self.input_dim])

        x = input
        for l in self.layers:
            x = l(x)

        assert_shape(x, [None, None, self.input_dim])
        return x
