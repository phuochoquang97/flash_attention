# FlashAttention from Scratch — Apple M5

An educational implementation of **exact scaled dot-product attention** and its **online-softmax, streaming** formulation. This repository progresses from NumPy reference code to a custom Apple Silicon GPU kernel using **MLX + Metal**.

> **Status: v0.2 (forward-only, learning prototype).** This is not the production FlashAttention implementation, and the Metal path is not yet cooperatively tiled or tuned for performance.

## Features

| Backend | Implementation | Status |
| --- | --- | --- |
| `reference` | Standard attention, full attention matrix (NumPy) | Implemented |
| `tiled` | Blockwise online softmax without a full attention-score matrix (NumPy) | Implemented |
| `metal` | Fused streaming attention, one Metal thread per query row (MLX) | Implemented; tested on Apple M5 |

All backends currently accept **single-head, 2D NumPy arrays**: `Q` of shape `(Nq, d)`, `K` of shape `(Nk, d)` and `V` of shape `(Nk, dv)`. The Metal prototype computes in `float32` and currently requires `dv <= 128`. Causal attention in this version requires `Nq == Nk`.

## Repository layout

```text
flash_attention/
├── flash_attention/
│   ├── __init__.py
│   ├── reference.py
│   ├── tiled.py
│   ├── metal.py
│   ├── functional.py
│   └── utils.py
├── notebooks/
│   ├── 01_standard_attention.ipynb
│   ├── 02_online_softmax.ipynb
│   ├── 03_tiled_attention.ipynb
│   ├── 04_metal_kernel.ipynb
│   └── 05_cooperative_tiling.ipynb
├── tests/
│   ├── test_correctness.py
│   └── test_metal.py
├── benchmarks/
├── examples/
├── pyproject.toml
└── README.md
```

## Setup (Apple Silicon Mac)

Use an Apple Silicon-native Python environment. From the **project root** (where `pyproject.toml` lives):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[metal,dev]"
```

If you need only the portable NumPy implementation, use:

```bash
python -m pip install -e ".[dev]"
```

Check which interpreter is running:

```bash
python -c "import sys, flash_attention; print(sys.executable); print(flash_attention.__file__)"
python -c "import mlx.core as mx; print('GPU available:', mx.is_available(mx.gpu))"
```

In Jupyter, select the matching `.venv` kernel. If necessary, install packages from a notebook with `%pip`, then restart its kernel.

## Quick start

```python
import numpy as np
from flash_attention import flash_attention

rng = np.random.default_rng(42)
Q = rng.standard_normal((128, 64)).astype(np.float32)
K = rng.standard_normal((128, 64)).astype(np.float32)
V = rng.standard_normal((128, 64)).astype(np.float32)

out = flash_attention(Q, K, V, causal=True, backend="metal", block_k=32)
print(out.shape)  # (128, 64)

# Compare with the NumPy tiled backend
out_cpu = flash_attention(Q, K, V, causal=True, backend="tiled")
np.testing.assert_allclose(out, out_cpu, rtol=1e-4, atol=1e-4)
```

For systems without MLX, change `backend="metal"` to `backend="tiled"`.

## How the algorithm works

Standard scaled dot-product attention is

$$
O = \mathrm{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d}}\right)V
$$

For each query row, the online formulation maintains a running score maximum `m`, an exponential normalizer `ell`, and a weighted value accumulator `acc`. For a new score `s` and value vector `v`, define

$$
m' = \max(m,s), \quad
\alpha = e^{m-m'}, \quad
\beta=e^{s-m'}.
$$

Then update

$$
\ell' = \alpha\ell + \beta, \qquad
\mathrm{acc}'=\alpha\,\mathrm{acc}+\beta v.
$$

After all keys, output `acc / ell`. Special initialization handles the first score. The tiled NumPy implementation performs equivalent updates by blocks instead of individual scores.

**Important:** Not storing the full score matrix does not eliminate quadratic computation. The v0.2 Metal prototype does not yet reuse K/V tiles cooperatively across GPU threads.

## Tests

```bash
python -m pytest -q
python -m pytest -q tests/test_metal.py
```

Run the Metal tests on your Apple M5 with MLX installed. Always compare GPU results against the NumPy reference, using reasonable floating-point tolerances.

## Roadmap

- [x] v0.1 — NumPy reference and tiled online-softmax implementations
- [x] v0.2 — First MLX/Metal streaming forward kernel, verified on M5
- [ ] v0.3 — Correct GPU benchmarks, cooperative K/V tile loading, threadgroup memory
- [ ] v0.4 — Optimize fusion and tile layouts
- [ ] v0.5 — Batched and multi-head attention
- [ ] v0.6 — Backward pass and gradients

## Known limitations

- Forward pass only; no automatic differentiation support.
- Single-head 2D inputs; no batching or grouped-query attention.
- No dropout or custom masks; causal masking only.
- `metal` requires Apple Silicon, MLX, and a working Metal GPU.
- Current `metal` kernel is a correctness-first, single-query-per-thread streaming implementation; not a production performance baseline.
- If your older `functional.py` forwards a `scale=` keyword to `metal_attention`, update that wrapper or make the Metal signature compatible before using that particular API call.

## Research basis

Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022). The code in this project is an independent educational implementation inspired by the paper's online-softmax and memory-aware algorithmic ideas.
