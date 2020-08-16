# transformers-tf

A collection of transformer network implementations in TensorFlow 2.
Specifically, I have implemented ideas from two papers: [Universal Transformers](https://arxiv.org/abs/1807.03819), and [Linear-Complexity Attention](https://arxiv.org/abs/2006.16236), and [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
See `requirements.txt` for dependencies.
All networks implemented as Keras layers, for easy incorporation into other projects.

The directory structure is as follows:
- `transformers.py` contains the 'root' layers, containing the Universal Transformer from the aforementioned paper.
- `attention_mechanisms.py` contains the self-attention sublayer, implemented using linear attention from the aforementioned paper.
- `layer_organisations.py` specifies how the regularisers (layer norm, dropout, etc) are positioned, and includes the option of a `PreLN` layout, rather than the traditional `PostLN` layout, as specified in the third paper mentioned above.
- `util.py` utility functions e.g. positional encodings, and a corresponding `util_test.py` unit test file.

TODO:
- Integrate with TensorFlow's masking system. At the moment, masking is explicit.
