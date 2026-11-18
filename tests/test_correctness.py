
import numpy as np

from flash_attention import (
    flash_attention,
    standard_attention,
    tiled_attention,
)

rng = np.random.default_rng(42)

N = 128
d = 64

Q = rng.standard_normal((N, d)).astype(np.float32)
K = rng.standard_normal((N, d)).astype(np.float32)
V = rng.standard_normal((N, d)).astype(np.float32)

# Standard attention
reference = standard_attention(
    Q, K, V, causal=True
)

# Tiled FlashAttention
tiled = tiled_attention(
    Q, K, V,
    causal=True,
    block_q=32,
    block_k=64,
)

# Public API
output = flash_attention(
    Q, K, V,
    causal=True,
    backend="tiled",
)

print("Reference shape:", reference.shape)
print("Tiled shape:", tiled.shape)

error = np.max(np.abs(reference - tiled))

print("Maximum absolute error:", error)

np.testing.assert_allclose(
    reference,
    tiled,
    rtol=1e-5,
    atol=1e-5,
)

np.testing.assert_allclose(
    output,
    tiled,
    rtol=1e-5,
    atol=1e-5,
)

print("All checks passed!")
