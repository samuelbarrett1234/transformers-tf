import tensorflow as tf
from util import (assert_shape, compute_positional_encoding,
                  compute_time_encoding)


class UniversalTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, attention, transition, num_repeats, layer_org):
        """Create a (universal) transformer which has a fixed number of depth
        recurrences, `num_repeats`. Note that when this == 1, we have the
        traditional transformer layer.

        Args:
            attention (tf.keras.layers.Layer): The attention layer.
            transition (tf.keras.layers.Layer): The transition function layer.
            num_repeats (int): The number of depth recurrences. When this is 1,
                               this layer becomes equivalent to a tranditional
                               transformer layer.
            layer_org (function): Encapsulates the arrangement of the attention
                                  and transition sub-layers within a single
                                  depth recurrence. PostLN is the same as the
                                  traditional transformers, however you can
                                  also set it to PreLN, which is an
                                  alternative layout.
        """
        super(UniversalTransformerLayer, self).__init__()
        self.attention = attention
        self.transition = transition
        self.num_repeats = num_repeats
        self.layer_org = layer_org
        self.value_outputs = []
        self.attention_outputs = []

    def call(self, input):
        # must be two inputs: the values and the mask!
        assert(isinstance(input, list) and len(input) == 2)

        # extract inputs
        x = input[0]
        mask = tf.expand_dims(input[1], axis=-1)  # expand dim for broadcasting

        # deduce input dimension:
        self.attention_dim = x.get_shape()[-1]
        assert(isinstance(self.attention_dim, int))

        # check input shapes
        assert_shape(x, [None, None, self.attention_dim])
        assert_shape(mask, [None, None, 1])

        # compute positional encoding and expand dims to permit broadcasting
        pos_enc = compute_positional_encoding(
            tf.shape(x)[1], self.attention_dim)
        pos_enc = tf.expand_dims(pos_enc, axis=0)

        # initialise with first masked output
        self.value_outputs = [x * mask]
        self.attention_outputs = []

        # now perform depth recurrences:
        for t in range(self.num_repeats):
            with tf.name_scope('depth_step_' + str(t)):
                # positional and time encodings:
                time_enc = compute_time_encoding(t, self.attention_dim)
                time_enc = tf.expand_dims(tf.expand_dims(time_enc, axis=0),
                                          axis=0)
                # timestep and positional embeddings
                x += pos_enc
                x += time_enc

                # attention and transition function
                x, att = self.layer_org([x, self.attention, self.transition])

                # mask the output
                x *= mask

                with tf.name_scope('masking_att_auxiliary_output'):
                    # mask the attention values
                    seq_len = tf.shape(x)[1]
                    num_att_heads = tf.shape(att)[0]

                    # apply sequence length mask after normalising - makes the
                    # output less confusing but not strictly necessary.
                    mask_ = tf.expand_dims(mask, axis=0)  # attention heads

                    # important: tile seq len in dimension 3, not dimension 2,
                    # then transpose and swap dims 2 and 3
                    mask_ = tf.tile(mask_, [num_att_heads, 1, 1, seq_len])
                    mask_ = tf.transpose(mask_, perm=[0, 1, 3, 2])

                    att *= mask_

                # track auxiliary outputs
                self.value_outputs.append(x)
                self.attention_outputs.append(att)

        # stack auxiliary outputs to make them one contiguous tensor

        with tf.name_scope('value_output'):
            # write out a tensor: batch_sz * num_repeats * seq_len
            # * attention_dim
            assert(len(self.value_outputs) == self.num_repeats + 1)
            self.value_outputs = tf.transpose(tf.stack(self.value_outputs),
                                              perm=[1, 0, 2, 3])
            assert_shape(self.value_outputs, [None, self.num_repeats + 1,
                                              None, self.attention_dim])

        with tf.name_scope('attention_output'):
            # write out a tensor: batch_sz * num_att_heads * num_repeats *
            # seq_len * seq_len
            self.attention_outputs = tf.transpose(
                tf.stack(self.attention_outputs), perm=[1, 0, 2, 3, 4])
            assert_shape(self.attention_outputs, [None, self.num_repeats, None,
                                                  None, None])

        # return the last layer of output
        return x

    def get_value_outputs(self):
        """Get the propagation of the value vectors over depth recurrences.
        Since we include the values before passing through the layer, the
        output has num_repeats+1 instead of just num_repeats.
        WARNING: outputs are not masked (i.e. if there are input sequences
        of different lengths, some values returned by this will be bogus.)

        Returns:
            tf.Tensor: A tensor of size batch_size * (num_repeats+1) * seq_len
                       * attention_dim which represents the propagation of the
                       values of the sequence over the depth recurrences.
        """
        assert(not isinstance(self.value_outputs, list))  # ensure stacked
        return self.value_outputs

    def get_attention_outputs(self):
        """Get an array of tensorflow nodes which compute, for each element in
        the batch and for each position in the sequence, which other elements
        it attends to.

        Returns:
            tf.Tensor: A tensor of shape num_att_heads * num_repeats *
            batch_size * max_seq_len * max_seq_len which gives the pairwise
            attention values for each batch, depth-step ("repeat"), and
            attention head. IMPORTANT: the value at index [a,b,c,d,e] is the
            amount of attention from attention head `a`, at timestep `b`,
            sequence `c` in the batch, that element `d` pays to element `e`
            (i.e. `d` is the key, `e` is the query).
        """
        assert(not isinstance(self.attention_outputs, list))  # ensure stacked
        return self.attention_outputs
