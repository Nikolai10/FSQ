# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://colab.research.google.com/github/google-research/google-research/blob/master/fsq/fsq.ipynb,
# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This is a reimplementation of FSQ published in:
F. Mentzer and D. Minnen and E. Agustsson and M. Tschannen:
"Finite Scalar Quantization: VQ-VAE Made Simple"
arXiv, 2023
https://arxiv.org/abs/2309.15505
"""
import numpy as np
import tensorflow as tf

Codeword = tf.Tensor
Indices = tf.Tensor


def round_ste(z):
    """Round with straight through gradients."""
    zhat = tf.round(z)
    return z + tf.stop_gradient(zhat - z)


class FSQ(tf.keras.layers.Layer):
    """Quantizer."""

    '''
    # Example usage
    fsq = FSQ(levels=[3, 5, 4])

    z = np.asarray([0.25, 0.6, -7])
    zhat = fsq(z) # == fsq.quantize(z)
    print(f"Quantized {z} -> {zhat}")
    
    # We can map to an index in the codebook.
    idx = fsq.codes_to_indices(zhat)
    print(f"Code {zhat} is the {idx}-th index.")
    
    # Back to code
    code_out = fsq.indices_to_codes(idx)
    print(f"Index {idx} mapped back to {zhat}.")
    '''

    def __init__(self, levels: list[int], eps: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(
            ([1], np.cumprod(self._levels_np[:-1]))).astype(np.uint32)

        self._implicit_codebook = self.indices_to_codes(
            np.arange(self.codebook_size))

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return np.prod(self._levels)

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: tf.Tensor) -> tf.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = tf.where(tf.math.mod(self._levels_np, 2) == 1, 0.0, 0.5)
        shift = tf.tan(offset / half_l)
        return tf.tanh(z + shift) * half_l - offset

    def quantize(self, z: tf.Tensor) -> Codeword:
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = self._levels_np // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Codeword) -> Indices:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return tf.cast(tf.reduce_sum(zhat * self._basis, axis=-1), tf.uint32)

    def indices_to_codes(self, indices: Indices) -> Codeword:
        """Inverse of `indices_to_codes`."""
        indices = tf.expand_dims(indices, axis=-1)
        codes_non_centered = np.mod(
            np.floor_divide(indices, self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(codes_non_centered)

    def call(self, x):
        return self.quantize(x)
