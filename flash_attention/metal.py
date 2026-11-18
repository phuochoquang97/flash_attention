"""Correctness-first Apple Silicon attention forward kernel (MLX + Metal).

Single-head 2D NumPy inputs, float32 compute, no backward pass.
One GPU thread owns one query row and streams through K/V blocks.
This is fused streaming attention, not yet cooperative FlashAttention.
"""

import numpy as np

from .utils import validate_inputs


_SOURCE = r"""
    uint qi = thread_position_in_grid.x;
    if (qi >= NQ) return;

    float m = 0.0f;
    float ell = 0.0f;
    bool started = false;
    float accumulator[DV];
    for (int c = 0; c < DV; ++c) {
        accumulator[c] = 0.0f;
    }

    const float scale = metal::rsqrt(float(D));

    for (int kb = 0; kb < NK; kb += BK) {
        int end = metal::min(kb + BK, NK);
        for (int kj = kb; kj < end; ++kj) {
            if (CAUSAL && kj > int(qi)) continue;

            float score = 0.0f;
            for (int c = 0; c < D; ++c) {
                score += q[qi * D + c] * k[kj * D + c];
            }
            score *= scale;

            // Online stable softmax, merging one score at a time.
            // The first score initializes the running maximum.
            float new_m = started ? metal::max(m, score) : score;
            float alpha = started ? metal::exp(m - new_m) : 0.0f;
            float beta = metal::exp(score - new_m);

            ell = alpha * ell + beta;
            for (int c = 0; c < DV; ++c) {
                accumulator[c] = alpha * accumulator[c]
                               + beta * v[kj * DV + c];
            }
            m = new_m;
            started = true;
        }
    }

    for (int c = 0; c < DV; ++c) {
        out[qi * DV + c] = accumulator[c] / ell;
    }
"""

_KERNEL = None


def metal_attention(Q, K, V, causal=False, block_q=32, block_k=64):
    """Compute attention with a fused streaming Metal kernel on Apple Silicon.

    Accepts NumPy arrays (Nq,d), (Nk,d), (Nk,dv) and returns a float32
    NumPy array (Nq,dv). This prototype uses one Metal thread per query.
    block_q is reserved for the future cooperative version.
    block_k controls the streaming loop's logical K/V tile size.

    This implementation is forward-only, with no gradients and no batch dimension.
    """
    validate_inputs(Q, K, V)
    if causal and Q.shape[0] != K.shape[0]:
        raise ValueError("Causal attention requires Nq == Nk in v0.2")
    for name, size in (("block_q", block_q), ("block_k", block_k)):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"{name} must be a positive integer")
    n_q, d = Q.shape
    n_k, d_v = V.shape[0], V.shape[1]
    if d_v > 128:
        raise ValueError("v0.2 prototype limits value head dimension to 128")

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise ImportError(
            "Metal backend requires MLX. Install in this Python environment: "
            "python -m pip install mlx"
        ) from exc

    if not mx.is_available(mx.gpu):
        raise RuntimeError("MLX GPU unavailable; run on your Apple M5")

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="flash_attention_streaming_v02",
            input_names=["q", "k", "v"],
            output_names=["out"],
            source=_SOURCE,
        )

    q_mx = mx.array(np.ascontiguousarray(Q, dtype=np.float32))
    k_mx = mx.array(np.ascontiguousarray(K, dtype=np.float32))
    v_mx = mx.array(np.ascontiguousarray(V, dtype=np.float32))

    output = _KERNEL(
        inputs=[q_mx, k_mx, v_mx],
        template=[
            ("NQ", int(n_q)),
            ("NK", int(n_k)),
            ("D", int(d)),
            ("DV", int(d_v)),
            ("BK", int(block_k)),
            ("CAUSAL", bool(causal)),
        ],
        grid=(n_q, 1, 1),
        threadgroup=(min(32, n_q), 1, 1),
        output_shapes=[(n_q, d_v)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(output)
    return np.array(output)
