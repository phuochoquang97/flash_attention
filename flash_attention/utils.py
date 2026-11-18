
import numpy as np


def validate_inputs(Q, K, V):
    """
    Validate single-head attention inputs.

    Q: (Nq, d)
    K: (Nk, d)
    V: (Nk, dv)
    """
    for name, x in [("Q", Q), ("K", K), ("V", V)]:
        if not isinstance(x, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")

        if x.ndim != 2:
            raise ValueError(f"{name} must be 2-dimensional")

        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"{name} must have floating-point dtype")

        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} must contain finite values")

    if Q.shape[1] != K.shape[1]:
        raise ValueError("Q and K must have matching head dimensions")

    if K.shape[0] != V.shape[0]:
        raise ValueError("K and V must have matching sequence lengths")

    if min(Q.shape[0], K.shape[0], Q.shape[1], V.shape[1]) == 0:
        raise ValueError("Attention dimensions must be nonzero")


def attention_scale(d):
    """Scaled dot-product attention factor."""
    return 1.0 / np.sqrt(d)


def causal_mask(n_q, n_k):
    """
    Return a boolean causal mask.

    True means the attention position is allowed.
    This v0.1 implementation uses square self-attention.
    """
    if n_q != n_k:
        raise ValueError("Causal attention requires Nq == Nk in v0.1")

    q_indices = np.arange(n_q)[:, None]
    k_indices = np.arange(n_k)[None, :]

    return k_indices <= q_indices


def stable_softmax(scores, axis=-1):
    """Numerically stable softmax."""
    maximum = np.max(scores, axis=axis, keepdims=True)
    exp_scores = np.exp(scores - maximum)
    denominator = np.sum(exp_scores, axis=axis, keepdims=True)

    return exp_scores / denominator
