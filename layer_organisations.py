import tensorflow as tf
from util import MyLayerNorm


"""Contains a collection of layers which here we call 'layer organisations'.
They abstract the construction of the individual recurrences of the UT's
depth-recurrences. The 'normal' layer organisation is PostLN.

The reason for this basically revolves around this paper:
https://arxiv.org/pdf/2002.04745.pdf

These layers all have the same interface:
- They accept *THREE* inputs: the layer input, the attention layer, and the
  transition function layer,
- They provide *TWO* outputs: the layer output, and the attention auxiliary
  output.
"""


class PostLN(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.2):
        """Create a Post-LN layer organisation, where LN=layer norm.
        This is the standard arrangement of the layer norm in the
        transformers.

        Args:
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2. If
                                            0, then there is no dropout.
        """
        assert(dropout_rate >= 0.0 and dropout_rate <= 1.0)
        super(PostLN, self).__init__()
        self.ln1 = MyLayerNorm()
        self.ln2 = MyLayerNorm()
        if dropout_rate > 0.0:
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
            self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        else:
            self.dropout1, self.dropout2 = None, None

    def call(self, inputs):
        assert(isinstance(inputs, list) and len(inputs) == 3)
        x, attention, transition = inputs[0], inputs[1], inputs[2]

        y, att = attention(x)
        if self.dropout1 is not None:
            y = self.dropout1(y)
        x += y
        x = self.ln1(x)

        y = transition(x)
        if self.dropout2 is not None:
            y = self.dropout2(y)
        x += y
        x = self.ln2(x)

        return x, att


class PreLN(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.2):
        """Create a Pre-LN layer organisation, where LN=layer norm.
        This is an alternative placement of layer normalisation, which is
        purported in https://arxiv.org/pdf/2002.04745.pdf to remove the need
        for the warm startup rate, and makes the gradients better behaved.

        Args:
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2. If
                                            0, then there is no dropout.
        """
        assert(dropout_rate >= 0.0 and dropout_rate <= 1.0)
        super(PreLN, self).__init__()
        self.ln1 = MyLayerNorm()
        self.ln2 = MyLayerNorm()
        self.ln3 = MyLayerNorm()
        if dropout_rate > 0.0:
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
            self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        else:
            self.dropout1, self.dropout2 = None, None

    def call(self, inputs):
        assert(isinstance(inputs, list) and len(inputs) == 3)
        x, attention, transition = inputs[0], inputs[1], inputs[2]

        y, att = attention(self.ln1(x))
        if self.dropout1 is not None:
            y = self.dropout1(y)
        x += y

        y = transition(self.ln2(x))
        if self.dropout1 is not None:
            y = self.dropout1(y)
        x += y

        return self.ln3(x), att
