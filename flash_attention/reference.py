
import numpy as np

from .utils import (
    validate_inputs,
    attention_scale,
    causal_mask,
    stable_softmax,
)


def standard_attention(Q, K, V, causal=False):
    """
    Standard scaled dot-product attention.

    Parameters
    ----------
    Q : ndarray, shape (Nq, d)
    K : ndarray, shape (Nk, d)
    V : ndarray, shape (Nk, dv)
    causal : bool

    Returns
    -------
    ndarray, shape (Nq, dv)

    Notes
    -----
    Materializes the complete attention matrix.
    Score memory complexity: O(Nq * Nk).
    """
    validate_inputs(Q, K, V)

    if causal and Q.shape[0] != K.shape[0]:
        raise ValueError("Causal attention requires Nq == Nk")

    dtype = np.result_type(Q.dtype, K.dtype, V.dtype, np.float32)

    q = Q.astype(dtype, copy=False)
    k = K.astype(dtype, copy=False)
    v = V.astype(dtype, copy=False)

    d = q.shape[-1]
    scale = attention_scale(d)

    # Step 1: Attention scores
    scores = (q @ k.T) * scale

    # Step 2: Causal mask
    if causal:
        mask = causal_mask(q.shape[0], k.shape[0])
        scores = np.where(mask, scores, -np.inf)

    # Step 3: Stable softmax
    probabilities = stable_softmax(scores, axis=-1)

    # Step 4: Weighted values
    output = probabilities @ v

    return output
