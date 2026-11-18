
from .reference import standard_attention
from .tiled import tiled_attention
from .metal import metal_attention


def flash_attention(
    Q,
    K,
    V,
    causal=False,
    backend="tiled",
    block_q=32,
    block_k=64,
):
    """
    Unified attention API.

    Parameters
    ----------
    Q : ndarray
        Query matrix (Nq, d).

    K : ndarray
        Key matrix (Nk, d).

    V : ndarray
        Value matrix (Nk, dv).

    causal : bool
        Whether to apply a causal mask.

    backend : str
        "reference" : standard NumPy attention
        "tiled"     : online softmax / tiled NumPy
        "metal"     : Apple GPU (not implemented)

    block_q : int
        Query tile size.

    block_k : int
        Key/value tile size.

    Returns
    -------
    ndarray
        Attention output (Nq, dv).
    """

    if backend == "reference":
        return standard_attention(
            Q, K, V, causal=causal
        )

    if backend == "tiled":
        return tiled_attention(
            Q, K, V,
            causal=causal,
            block_q=block_q,
            block_k=block_k,
        )

    if backend == "metal":
        return metal_attention(
            Q, K, V,
            causal=causal,
            block_q=block_q,
            block_k=block_k,
        )

    raise ValueError(
        f"Unknown backend: {backend!r}. "
        "Choose 'reference', 'tiled', or 'metal'."
    )
