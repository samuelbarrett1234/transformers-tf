import numpy as np
import tensorflow as tf


"""A collection of internal functions used for the transformer code in the
rest of this package."""


def assert_shape(node, shape):
    """Assert that the shape of `node` fits with the given prescribed
    `shape`. This forms a partial ordering on shapes, where the partial
    order represents "how much we know about the shape". I.e. this
    function asserts that: the shape of `node` satisfies *at least* the
    conditions given by `shape`, but we might know more about the shape
    of `node` than `shape`.

    Args:
        node (tf node): The node whose shape to check.
        shape (tuple or list of (ints or Nones)): The shape list, where int
                                                    values are constraints on
                                                    the dimension, and None
                                                    values are not
                                                    constraints - i.e. this
                                                    shape parameter is
                                                    interpreted exactly like
                                                    TensorFlow does.
    """
    if shape is None:
        return  # None shape is the bottom of the information hierarchy
    assert (len(node.get_shape()) == len(shape))
    assert(all([s1 == s2 or s2 is None
                for s1, s2 in zip(node.get_shape(), shape)]))


def compute_positional_encoding(max_seq_len, attention_dim):
    """Compute the positional encoding tensor. We can't make this much more
    explicit because max_seq_len may be an indeterminate tensor. See the
    UT's corresponding unit test file for a more explicit computation of
    this, or see _compute_time_encoding.

    Args:
        max_seq_len (int or scalar-shaped tf tensor): The maximum sequence
                                                        length for which we
                                                        compute positional
                                                        embeddings.
        attention_dim (int): The dimension of the attention vectors.

    Returns:
        tf.Tensor: A tensor of shape max_seq_len * attention_dim.
    """
    with tf.name_scope('positional_encoding_computation'):
        # Compute an array A such that,
        # A[i, d] = np.sin(i / multiplier)
        # when d is even, and when d is odd:
        # A[i, d] = np.cos(i / multiplier)
        # where i ranges over sequence indices, and d ranges across
        # attention_dim dimension values.

        # compute embedding (warning: sequence length is nondeterministic)
        multiplier_vec = tf.constant([10000.0 ** (2 * (d // 2)
                                                  / attention_dim)
                                      for d in range(attention_dim)],
                                     dtype=tf.float32)
        seq_indices = tf.range(start=0, limit=tf.cast(max_seq_len,
                                                      tf.float32),
                               dtype=tf.float32)

        # use multiplier (broadcasting in effect here!)
        seq_indices = (tf.expand_dims(seq_indices, axis=1)
                       / tf.expand_dims(multiplier_vec, axis=0))
        assert_shape(seq_indices, [None, attention_dim])

        # now apply sin/cos depending on whether d is odd or even; to do
        # this we will perform sin/cos over the whole vector, then
        # separately pick out the odd/even elements, then stitch them
        # together
        seq_sin, seq_cos = (tf.math.sin(seq_indices),
                            tf.math.cos(seq_indices))

        # remove odd/even elements
        seq_sin = seq_sin[:, ::2]
        seq_cos = seq_cos[:, 1::2]

        # if attention_dim is odd, need to pad the cosines, so we can
        # stack:
        if attention_dim % 2 != 0:
            row_of_zero = tf.zeros(shape=[max_seq_len, 1])
            seq_cos = tf.concat([seq_cos, row_of_zero], axis=1)
            newdim = attention_dim + 1
        else:
            newdim = attention_dim

        # stack the even and odd elements together
        seq_indices = tf.stack([seq_sin, seq_cos], axis=1)

        # now use reshape magic to bring them together again:
        seq_indices = tf.transpose(seq_indices, perm=[0, 2, 1])
        seq_indices = tf.reshape(seq_indices, shape=[max_seq_len,
                                                     newdim])

        assert_shape(seq_indices, [None, newdim])

        if attention_dim != newdim:
            # drop extra bit (the +1 attention dim) added eariler
            seq_indices = seq_indices[:, :attention_dim]

        assert_shape(seq_indices, [None, attention_dim])

        return seq_indices


def compute_time_encoding(t, attention_dim):
    """Compute the time (i.e. depth recurrence) encoding.

    Args:
        t (int): An integer >= 0 specifying the depth-recurrent timestep
        attention_dim (int): The dimension of the attention vectors.

    Returns:
        A tensor of shape [attention_dim] which contains the time embedding
        vector.
    """
    assert (t >= 0)

    with tf.name_scope('time_encoding_computation'):
        A = np.zeros((attention_dim,))
        for i in range(attention_dim):
            i2 = i // 2
            exponent = 2 * i2 / attention_dim
            multiplier = 10000.0 ** exponent
            x = t / multiplier
            if i % 2 == 0:
                A[i] = np.sin(x)
            else:
                A[i] = np.cos(x)
        return tf.constant(A, dtype=tf.float32)


class MyLayerNorm(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.gamma = self.add_weight('gamma',
                                     shape=input_shape[-1],
                                     initializer='ones')
        self.beta = self.add_weight('beta',
                                    shape=input_shape[-1],
                                    initializer='ones')

    def call(self, input):
        mean, var = tf.nn.moments(input, axes=[-1],
                                  keepdims=True)
        invstd = tf.math.rsqrt(var)
        return (input - mean) * invstd * self.gamma + self.beta
