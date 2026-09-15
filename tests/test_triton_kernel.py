import math
import torch
import pytest
from subqtransformer.triton_kernel import subq_attention, FastSubQAttentionFunction

def test_fast_subq_attention_fallback():
    B, H, L, D = 2, 2, 16, 8
    K_total = 3
    offsets = torch.tensor([0, 2, -2], dtype=torch.int32)
    biases = torch.randn(K_total, requires_grad=True)

    Q = torch.randn(B, H, L, D, requires_grad=True)
    K = torch.randn(B, H, L, D, requires_grad=True)
    V = torch.randn(B, H, L, D, requires_grad=True)

    out = subq_attention(Q, K, V, offsets, biases)
    assert out.shape == (B, H, L, D)

    loss = out.sum()
    loss.backward()

    assert Q.grad is not None
    assert K.grad is not None
    assert V.grad is not None
    assert biases.grad is not None
    assert not torch.isnan(Q.grad).any()
    assert not torch.isnan(K.grad).any()
    assert not torch.isnan(V.grad).any()
    assert not torch.isnan(biases.grad).any()
