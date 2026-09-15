"""
SubQTransformer: Sub-Quadratic Iterative Transformer with Adaptive Dynamical Halting.
"""

from subqtransformer.config import SubQConfig
from subqtransformer.layers import SubQSurfer, SubQBlock
from subqtransformer.model import SubQTransformerLM, SubQTransformerClassifier
from subqtransformer.triton_kernel import subq_attention, SubQTritonFunction, FastSubQAttentionFunction

__version__ = "1.0.0"
__all__ = [
    "SubQConfig",
    "SubQSurfer",
    "SubQBlock",
    "SubQTransformerLM",
    "SubQTransformerClassifier",
    "subq_attention",
    "SubQTritonFunction",
    "FastSubQAttentionFunction",
]
