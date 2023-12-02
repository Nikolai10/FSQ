# Finite Scalar Quantization (TensorFlow 2)

[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1FA6AP53oMJq_f641vPesCBuRiXghaXnY?usp=sharing)

This repository provides a TensorFlow implementation of FSQ based on:

- [Finite Scalar Quantization: VQ-VAE Made Simple (ArXiv 2023)](https://arxiv.org/abs/2309.15505).

<p align="center">
    <img src="https://github.com/Nikolai10/FSQ/blob/master/res/doc/figures/fsq.png" width="80%" />
</p>

<p align="center"><em>Image source: https://arxiv.org/abs/2309.15505</em></p>

## Install

```bash
$ git clone https://github.com/Nikolai10/FSQ
```

## Usage

```python
import sys
sys.path.append('/content/FSQ') # adjust path to your needs

```

```python
from finite_scalar_quantization import FSQ
import numpy as np

fsq = FSQ(levels=[3, 5, 4])

z = np.asarray([0.25, 0.6, -7])
zhat = fsq(z) # == fsq.quantize(z)
print(f"Quantized {z} -> {zhat}") # Quantized [ 0.25  0.6  -7.  ] -> [ 0.   0.5 -1. ]

# We can map to an index in the codebook.
idx = fsq.codes_to_indices(zhat)
print(f"Code {zhat} is the {idx}-th index.") # Code [ 0.   0.5 -1. ] is the 10-th index.

# Back to code
code_out = fsq.indices_to_codes(idx)
print(f"Index {idx} mapped back to {zhat}.") # Index 10 mapped back to [ 0.   0.5 -1. ].

```

See [Colab with FSQ code](https://colab.research.google.com/drive/1FA6AP53oMJq_f641vPesCBuRiXghaXnY?usp=sharing) for more details.

## Advanced Usage (FSQ-VAE Demo)

A notebook on how to train a FSQ-VAE is additionally provided here: [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1bA6vvee531jBXoeQSUJfPUQM63_DWYZj?usp=sharing).

This notebook largely follows the Keras tutorial [Vector-Quantized Variational Autoencoders](https://keras.io/examples/generative/vq_vae/); 
the main change is that we replace the ```VectorQuantizer``` with our ```FSQ``` class.

## Comparison

|                  | VQ | FSQ |
|------------------|----|-----|
| Quantization     | argmin_c \|\| z-c \|\| | round(f(z)) |
| Gradients        | Straight Through Estimation (STE) | STE |
| Auxiliary Losses | Commitment, codebook, entropy loss, ... | N/A |
| Tricks           | EMA on codebook, codebook splitting, projections, ...| N/A |
| Parameters       | Codebook | N/A |

## Related work

- official JAX implementation: https://github.com/google-research/google-research/tree/master/fsq,
- external PyTorch port: https://github.com/lucidrains/vector-quantize-pytorch.

## License
[Apache License 2.0](LICENSE)
