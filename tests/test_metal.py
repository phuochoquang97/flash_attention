"""Run on Apple Silicon with: python -m pytest -q tests/test_metal.py"""
import numpy as np
import pytest
from flash_attention import flash_attention, standard_attention


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n,d,dv,bk", [(8, 8, 8, 3), (17, 16, 16, 5), (33, 32, 32, 16), (64, 64, 64, 32)])
def test_metal_matches_reference(causal, n, d, dv, bk):
    mx = pytest.importorskip("mlx.core")
    if not mx.is_available(mx.gpu):
        pytest.skip("Requires Metal GPU")
    rng = np.random.default_rng(17)
    q, k, v = [rng.standard_normal(shape).astype(np.float32)
               for shape in ((n, d), (n, d), (n, dv))]
    expected = standard_attention(q, k, v, causal=causal)
    actual = flash_attention(q, k, v, causal=causal, backend="metal", block_k=bk)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_metal_rectangular_noncausal():
    mx = pytest.importorskip("mlx.core")
    if not mx.is_available(mx.gpu):
        pytest.skip("Requires Metal GPU")
    rng = np.random.default_rng(77)
    q = rng.standard_normal((7, 12)).astype(np.float32)
    k = rng.standard_normal((13, 12)).astype(np.float32)
    v = rng.standard_normal((13, 20)).astype(np.float32)
    expected = standard_attention(q, k, v)
    actual = flash_attention(q, k, v, backend="metal", block_k=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
