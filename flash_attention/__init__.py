
"""FlashAttention from Scratch — Apple Silicon edition."""

from .functional import flash_attention
from .reference import standard_attention
from .tiled import tiled_attention

__version__ = "0.1.0"

__all__ = [
    "flash_attention",
    "standard_attention",
    "tiled_attention",
]
