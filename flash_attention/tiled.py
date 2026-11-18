
import numpy as np

from .utils import validate_inputs, attention_scale


def tiled_attention(
    Q,
    K,
    V,
    causal=False,
    block_q=32,
    block_k=64,
):
    """
    FlashAttention-style tiled forward pass.

    Uses online softmax and avoids materializing
    the complete Nq x Nk attention matrix.

    This is a NumPy reference algorithm,
    not a fused GPU kernel.

    Parameters
    ----------
    Q : ndarray (Nq, d)
    K : ndarray (Nk, d)
    V : ndarray (Nk, dv)
    causal : bool
    block_q : int
    block_k : int

    Returns
    -------
    ndarray (Nq, dv)
    """
    validate_inputs(Q, K, V)

    if causal and Q.shape[0] != K.shape[0]:
        raise ValueError("Causal attention requires Nq == Nk")

    if (
        not isinstance(block_q, int)
        or isinstance(block_q, bool)
        or block_q <= 0
        or not isinstance(block_k, int)
        or isinstance(block_k, bool)
        or block_k <= 0
    ):
        raise ValueError("Block sizes must be positive integers")

    dtype = np.result_type(Q.dtype, K.dtype, V.dtype, np.float32)

    q = Q.astype(dtype, copy=False)
    k = K.astype(dtype, copy=False)
    v = V.astype(dtype, copy=False)

    Nq, d = q.shape
    Nk = k.shape[0]
    dv = v.shape[1]

    scale = attention_scale(d)
    output = np.empty((Nq, dv), dtype=dtype)

    # Outer loop: query tiles
    for q_start in range(0, Nq, block_q):

        q_end = min(q_start + block_q, Nq)
        q_tile = q[q_start:q_end]

        Bq = q_tile.shape[0]

        # Online softmax state for each query row
        m = np.full(Bq, -np.inf, dtype=dtype)
        l = np.zeros(Bq, dtype=dtype)
        acc = np.zeros((Bq, dv), dtype=dtype)

        # Inner loop: key/value tiles
        for k_start in range(0, Nk, block_k):

            k_end = min(k_start + block_k, Nk)

            # For causal attention, entire future
            # tiles can be skipped.
            if causal and k_start >= q_end:
                break

            k_tile = k[k_start:k_end]
            v_tile = v[k_start:k_end]

            # (Bq, Bk) local score tile
            scores = (q_tile @ k_tile.T) * scale

            # Apply local causal masking
            if causal:
                q_positions = np.arange(
                    q_start, q_end
                )[:, None]

                k_positions = np.arange(
                    k_start, k_end
                )[None, :]

                allowed = k_positions <= q_positions

                scores = np.where(
                    allowed, scores, -np.inf
                )

            # Local per-row maximum
            block_m = np.max(scores, axis=1)

            # Updated running maximum
            new_m = np.maximum(m, block_m)

            # Handle rows that are fully masked.
            # Their new maximum remains -inf.
            valid = np.isfinite(new_m)

            alpha = np.zeros(Bq, dtype=dtype)
            alpha[valid] = np.exp(
                m[valid] - new_m[valid]
            )

            # Probabilities relative to new maximum
            p = np.zeros_like(scores)

            if np.any(valid):
                p[valid] = np.exp(
                    scores[valid] - new_m[valid, None]
                )

            # Online softmax denominator
            l = alpha * l + np.sum(p, axis=1)

            # Online weighted-value accumulator
            acc = (
                alpha[:, None] * acc
                + p @ v_tile
            )

            m = new_m

        # Final normalization
        output[q_start:q_end] = acc / l[:, None]

    return output
