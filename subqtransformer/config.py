"""
Configuration dataclass for SubQTransformer models.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SubQConfig:
    """
    Configuration for SubQTransformer (Sub-Quadratic Iterative Transformer).
    
    Args:
        vocab_size: Size of the vocabulary.
        d_model: Dimensionality of the model representations.
        n_heads: Number of attention/routing heads.
        n_layers: Number of physical stacked SubQ blocks.
        default_T: Default number of thought hops (recurrent iterations) per layer.
        max_seq_len: Maximum supported sequence length for positional buffers.
        d_mlp: Hidden dimension of the feedforward MLP (defaults to 4 * d_model).
        jump_offsets: List of multi-scale relative offset distances. If None,
                     uses exponential / Fibonacci jumps covering context up to max_seq_len.
        dropout: Dropout probability.
        adaptive_halting: Whether to enable dynamical per-token early-exit halting by default.
        halt_threshold: Velocity or confidence threshold epsilon for early exiting.
        halt_criterion: Early-exit condition: 'velocity', 'stability', 'confidence', 'entropy'.
        weight_tying: Tie input token embeddings with output linear projection weights.
        layer_norm_eps: Epsilon for LayerNorm.
        bias: Whether to use bias in linear projection layers.
    """
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    default_T: int = 3 
    max_seq_len: int = 2048
    d_mlp: Optional[int] = None
    jump_offsets: Optional[List[int]] = None
    dropout: float = 0.0
    adaptive_halting: bool = False
    halt_threshold: float = 0.08
    halt_criterion: str = "velocity" 
    weight_tying: bool = True
    layer_norm_eps: float = 1e-5 
    bias: bool = False

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
        if self.jump_offsets is None:
            # Default multi-scale relative offset strides
            offsets = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2047] 
            self.jump_offsets = sorted(list(set([o for o in offsets if o < self.max_seq_len])))
            if len(self.jump_offsets) == 0:
                self.jump_offsets = [0]
