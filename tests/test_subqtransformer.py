"""
Comprehensive Unit & Integration Tests for SubQTransformer.
"""

import unittest
import torch
from subqtransformer import (
    SubQConfig,
    SubQSurfer,
    SubQBlock,
    SubQTransformerLM,
    SubQTransformerClassifier
)
import gravimem


class TestSubQTransformer(unittest.TestCase):
    def setUp(self):
        self.config = SubQConfig(
            vocab_size=256,
            d_model=64,
            n_heads=4,
            n_layers=2,
            default_T=3,
            max_seq_len=128,
            d_mlp=128,
            jump_offsets=[0, 1, 2, 4, 8, 16, 32, 64]
        )

    def test_config_initialization(self):
        cfg = SubQConfig(vocab_size=1000, d_model=128)
        self.assertEqual(cfg.d_mlp, 512)
        self.assertTrue(len(cfg.jump_offsets) > 0)

    def test_lm_forward_and_loss(self):
        model = SubQTransformerLM(self.config)
        x = torch.randint(0, self.config.vocab_size, (2, 32))
        y = torch.randint(0, self.config.vocab_size, (2, 32))

        logits, loss = model(x, targets=y)
        self.assertEqual(logits.shape, (2, 32, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))

        # Test backward pass
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Grad is None for {name}")

    def test_adaptive_halting_and_stats(self):
        model = SubQTransformerLM(self.config)
        x = torch.randint(0, self.config.vocab_size, (2, 16))

        logits, stats = model(x, adaptive_halting=True, halt_threshold=0.08, return_stats=True)
        self.assertEqual(logits.shape, (2, 16, self.config.vocab_size))
        self.assertIn("avg_hops_per_layer", stats)
        self.assertIn("mean_total_hops", stats)
        self.assertTrue(stats["mean_total_hops"] > 0)

    def test_generation(self):
        model = SubQTransformerLM(self.config)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=5)
        self.assertEqual(out.shape, (1, 13))

    def test_classifier(self):
        model = SubQTransformerClassifier(num_classes=4, config=self.config)
        x = torch.randint(0, self.config.vocab_size, (3, 20))
        y = torch.tensor([0, 1, 3], dtype=torch.long)

        logits, loss = model(x, targets=y)
        self.assertEqual(logits.shape, (3, 4))
        self.assertIsNotNone(loss)

    def test_mlp_intervals(self):
        # Test mlp_interval = 0 (canonical default: MLP only at end of block)
        cfg0 = SubQConfig(vocab_size=128, d_model=32, n_heads=2, n_layers=1, default_T=4, mlp_interval=0)
        m0 = SubQTransformerLM(cfg0)
        x = torch.randint(0, 128, (2, 16))
        out0 = m0(x)
        self.assertEqual(out0.shape, (2, 16, 128))

        # Test mlp_interval = 1 (MLP at every hop)
        cfg1 = SubQConfig(vocab_size=128, d_model=32, n_heads=2, n_layers=1, default_T=4, mlp_interval=1)
        m1 = SubQTransformerLM(cfg1)
        out1 = m1(x)
        self.assertEqual(out1.shape, (2, 16, 128))

        # Test mlp_interval = 2 (MLP every 2 hops)
        cfg2 = SubQConfig(vocab_size=128, d_model=32, n_heads=2, n_layers=1, default_T=4, mlp_interval=2)
        m2 = SubQTransformerLM(cfg2)
        out2 = m2(x)
        self.assertEqual(out2.shape, (2, 16, 128))

        # Test mlp_interval = 3 with T = 4 (boundary case where steps % interval != 0)
        cfg3 = SubQConfig(vocab_size=128, d_model=32, n_heads=2, n_layers=1, default_T=4, mlp_interval=3)
        m3 = SubQTransformerLM(cfg3)
        out3 = m3(x)
        self.assertEqual(out3.shape, (2, 16, 128))

    def test_harmonic_routing_mode(self):
        cfg = SubQConfig(
            vocab_size=128,
            d_model=32,
            n_heads=2,
            n_layers=1,
            default_T=3,
            routing_mode="harmonic",
            num_waves=8,
            K_peaks=4
        )
        model = SubQTransformerLM(cfg)
        x = torch.randint(0, 128, (2, 16))
        y = torch.randint(0, 128, (2, 16))

        logits, loss = model(x, targets=y)
        self.assertEqual(logits.shape, (2, 16, 128))
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))

        # Verify gradients propagate to wave latent and wave transition parameters
        loss.backward()
        for name, param in model.named_parameters():
            if "wave" in name and param.requires_grad:
                self.assertIsNotNone(param.grad, f"Grad is None for {name}")

    def test_bidirectional_mode(self):
        # Test bidirectional mode (is_causal=False) for vision/encoders
        cfg = SubQConfig(
            vocab_size=128,
            d_model=32,
            n_heads=2,
            n_layers=1,
            default_T=3,
            is_causal=False,
            jump_offsets=[0, 1, 2, 4, 8]
        )
        block = SubQBlock(cfg)
        x = torch.randn(2, 16, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 16, 32))

        # Also test with harmonic routing
        cfg_harm = SubQConfig(
            vocab_size=128,
            d_model=32,
            n_heads=2,
            n_layers=1,
            default_T=3,
            routing_mode="harmonic",
            is_causal=False,
            num_waves=6,
            K_peaks=8
        )
        block_harm = SubQBlock(cfg_harm)
        out_harm = block_harm(x)
        self.assertEqual(out_harm.shape, (2, 16, 32))

    def test_mirrored_bilateral_wave_symmetry(self):
        # Explicit test for the mirror on token i:
        # P peaks on left, P peaks on right, identical biases, no wrap-around
        cfg = SubQConfig(
            vocab_size=64,
            d_model=32,
            n_heads=2,
            n_layers=1,
            default_T=2,
            routing_mode="harmonic",
            is_causal=False,
            num_waves=6,
            K_peaks=8
        )
        surfer = SubQSurfer(cfg)
        p = surfer.p_peaks  # 4
        self.assertEqual(surfer.K, 1 + 2 * p)  # 9 candidates

        offsets, vals, _ = surfer.compute_wave_offsets(surfer.init_wave_latent, B=1, device=torch.device("cpu"))
        # Shape: (1, n_heads, K)
        center_offset = offsets[0, 0, 0].item()
        left_offsets = offsets[0, 0, 1:p+1]
        right_offsets = offsets[0, 0, p+1:]

        self.assertEqual(center_offset, 0)
        # Verify right offsets are exact negation of left offsets (-Δ)
        self.assertTrue(torch.equal(right_offsets, -left_offsets))
        # Verify biases are identical on left and right
        left_vals = vals[0, 0, 1:p+1]
        right_vals = vals[0, 0, p+1:]
        self.assertTrue(torch.equal(left_vals, right_vals))

        # Test single hop attention with edge tokens (L=10)
        x = torch.randn(1, 10, 32)
        out, _ = surfer.single_hop_attention(x)
        self.assertEqual(out.shape, (1, 10, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_custom_routing_mode(self):
        custom_offsets = [0, 1, 3, 7, 15]
        cfg = SubQConfig(
            vocab_size=128,
            d_model=32,
            n_heads=2,
            n_layers=1,
            routing_mode="custom",
            jump_offsets=custom_offsets
        )
        self.assertEqual(cfg.jump_offsets, custom_offsets)
        model = SubQTransformerLM(cfg)
        x = torch.randint(0, 128, (2, 16))
        logits = model(x)
        self.assertEqual(logits.shape, (2, 16, 128))

    def test_gravimem_backward_compatibility(self):
        lm = gravimem.GravimemLM(vocab_size=256, d_model=64, max_seq_len=128)
        x = torch.randint(0, 256, (2, 16))
        out = lm(x)
        self.assertEqual(out.shape, (2, 16, 256))

        cfg = gravimem.SubQConfig(vocab_size=128, d_model=32)
        clf = gravimem.GravimemClassifier(num_classes=5, config=cfg)
        x_clf = torch.randint(0, 128, (2, 10))
        out_clf = clf(x_clf)
        self.assertEqual(out_clf.shape, (2, 5))

        block = gravimem.GravimemBlock(cfg)
        h = torch.randn(2, 10, 32)
        out_block = block(h)
        self.assertEqual(out_block.shape, (2, 10, 32))

        surfer = gravimem.FusedPositionalJumpSurfer(cfg)
        out_surfer = surfer(h)
        self.assertEqual(out_surfer.shape, (2, 10, 32))


if __name__ == "__main__":
    unittest.main()

