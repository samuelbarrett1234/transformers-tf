import tensorflow as tf
from util import assert_shape


"""All transformers interleave two sub layers: attention and a transition
function. This is the collection of attention mechanisms.

All attention layers here return *TWO* outputs: the layer output (as normal)
and the pairwise attention values, as an auxiliary output.
"""


class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim_frac=1.0, eps=1.0e-9):
        """Create a linear attention layer - this is like the normal
        transformer dot product attention, except using a kernel-trick kind
        of optimisation, the complexity of the attention operation is brought
        down to linear, from quadratic, in the sequence length.

        Of course we cannot compute the auxiliary output (pairwise attention
        values) in linear time - this is simply not possible.

        Args:
            num_heads (int): The number of attention heads.
            key_dim_frac (float, optional): The proportion of the input
                                            dimension that each attention head
                                            projects the keys/queries into.
                                            Typical values include 1.0 (the
                                            key/query dim is equal to the value
                                            dim), or 1/num_heads, meaning the
                                            value dim is shared equally over
                                            the attention heads. Defaults to 1.
            eps (float): Numerical stability during normalisation of attention
                         values.
        """
        super(LinearAttention, self).__init__()
        assert(key_dim_frac > 0.0)
        assert(eps >= 0.0)
        self.num_att_heads = num_heads
        self.key_dim_frac = key_dim_frac
        self.eps = eps

    def build(self, input_shape):
        # VALIDATE INPUT
        assert(len(input_shape) == 3)
        self.attention_dim = input_shape[-1]
        assert(isinstance(self.attention_dim, int))
        self.key_dim = int(self.attention_dim * self.key_dim_frac)
        if self.key_dim == 0:
            raise ValueError("Key dimension was 0 - key_dim_frac too small.")

        # CREATE VARIABLES

        # weight initialiser for all of the below:
        att_weight_init = tf.keras.initializers.GlorotNormal()

        # key, query and value weight matrices
        self.Wks = self.add_weight(shape=[self.num_att_heads,
                                          self.attention_dim, self.key_dim],
                                   trainable=True, initializer=att_weight_init,
                                   name='Wks')
        self.Wqs = self.add_weight(shape=[self.num_att_heads,
                                          self.attention_dim, self.key_dim],
                                   trainable=True, initializer=att_weight_init,
                                   name='Wqs')
        self.Wvs = self.add_weight(shape=[self.num_att_heads,
                                          self.attention_dim,
                                          self.attention_dim],
                                   trainable=True, initializer=att_weight_init,
                                   name='Wvs')
        # and the weightings for merging the attention heads back together:
        self.Wo = self.add_weight(shape=[self.num_att_heads,
                                         self.attention_dim,
                                         self.attention_dim],
                                  trainable=True, initializer=att_weight_init,
                                  name='Wo')

    def call(self, input):
        assert_shape(input, [None, None, self.attention_dim])
        node = input

        # compute keys, queries and values for the self attention layer
        keys = tf.einsum('ijk,mkn->mijn', node, self.Wks)
        queries = tf.einsum('ijk,mkn->mijn', node, self.Wqs)
        values = tf.einsum('ijk,mkn->mijn', node, self.Wvs)

        # check shapes
        assert_shape(keys, [self.num_att_heads, None, None,
                            self.key_dim])
        assert_shape(queries, [self.num_att_heads, None, None,
                               self.key_dim])
        assert_shape(values, [self.num_att_heads, None, None,
                              self.attention_dim])

        # apply kernel function (HERE IS THE DIFFERENCE WITH NORMAL DOT
        # PRODUCT ATTENTION!)
        keys = tf.nn.elu(keys) + tf.constant(1.0)
        queries = tf.nn.elu(queries) + tf.constant(1.0)

        att = self._compute_pairwise_att(keys, queries)

        # the cumsum across dimension 2 performs a sum across the sequenece
        # effectively acting as an attention mask. See O(n) transformer
        # paper for details.
        numerator_sum_elements = tf.einsum('ijkm,ijkn->ijknm', keys, values)
        numerator = tf.cumsum(numerator_sum_elements, axis=2)
        assert_shape(numerator, [self.num_att_heads, None, None,
                                 self.attention_dim, self.key_dim])

        # note: a particular bracketing here is crucial to obtain O(n)
        # efficiency rather than O(n^2).
        numerator = tf.einsum('ijknm,ijkm->ijkn', numerator, queries)
        denominator = tf.einsum('ijkm,ijkm->ijk', tf.cumsum(keys, axis=2),
                                queries)

        assert_shape(numerator, [self.num_att_heads, None, None,
                                 self.attention_dim])
        assert_shape(denominator, [self.num_att_heads, None, None])

        # add trailing dim for broadcasting
        denominator = tf.expand_dims(denominator, axis=-1)
        assert_shape(denominator, [self.num_att_heads, None, None, 1])

        node = numerator / (denominator + self.eps)  # normalise

        assert_shape(node, [self.num_att_heads, None, None,
                            self.attention_dim])

        # reduce attention heads by a Wo matrix:
        node = tf.einsum('ijkm,imn->jkn', node, self.Wo)

        assert_shape(node, [None, None, self.attention_dim])

        return node, att

    def _compute_pairwise_att(self, keys, queries):
        """Compute the attention values between pairs of elements in the
        sequence, based on the key and query values.
        Args:
            keys (tf.Tensor): Size: num_att_heads * batch_size * max_seq_len
                              * key_dim
            queries (tf.Tensor): Same size as keys

        Returns: a tensor of shape num_att_heads * batch_sz * max_seq_len
                 * max_seq_len which is appropriately masked and normalised
                 so that position [i,j,k,l] is the amount of attention
                 query k pays to key l in sequence j under attention head i.
        """
        assert_shape(keys, [self.num_att_heads, None, None,
                            self.key_dim])
        assert_shape(queries, [self.num_att_heads, None, None,
                               self.key_dim])

        with tf.name_scope('pairwise_att_calculations'):
            # `att[a,b,c,d]` represents attention head `a`, sequence `b`,
            # and the attention that query `c` pays to key `d`.
            att = tf.einsum('ijkn,ijmn->ijkm', queries, keys)

            # `att` now contains the pairwise dot products
            assert_shape(att, [self.num_att_heads, None, None, None])

            # now apply triangular mask, to ensure attention goes in the right
            # direction (i.e. a node cannot attend to the future).
            # of course, it's important to do this before normalisation.
            att = tf.linalg.band_part(att, -1, 0)

            # normalise by summing across keys
            att = att / (tf.reduce_sum(att, axis=-1, keepdims=True) + self.eps)

            assert_shape(att, [self.num_att_heads, None, None, None])
            return att
