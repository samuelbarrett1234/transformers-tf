import pytest
from itertools import product
import numpy as np
from util import compute_positional_encoding


model_param_names = "seq_sz,elem_dim"
model_parameters = product([1, 2, 10],
                           [1, 2, 10])


class TestUniversalTransformerUtilities:
    @pytest.mark.parametrize(model_param_names, model_parameters)
    def test_compute_positional_embedding(self, seq_sz, elem_dim):
        # compute the answer using no TensorFlow magic (i.e. the computation is
        # much more explicit here because we don't need to define a computation
        # graph.)
        A = np.zeros(shape=(seq_sz, elem_dim))
        for i in range(seq_sz):
            for d in range(elem_dim):
                d2 = d // 2
                exponent = 2 * d2 / elem_dim
                multiplier = 10000.0 ** exponent

                if d % 2 == 0:  # if even
                    A[i, d] = np.sin(i / multiplier)
                else:  # if odd
                    A[i, d] = np.cos(i / multiplier)

        # now compute B, then check it is close enough to A, the
        # one we computed explicitly
        B = compute_positional_encoding(seq_sz, elem_dim)
        assert(np.all(np.isclose(A, B.numpy())))
