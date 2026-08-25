"""
SubQTransformer Model Architectures:
- SubQTransformerLM: Autoregressive Language Model with sub-quadratic attention & dynamic thought depth.
- SubQTransformerClassifier: Sequence / graph reasoning classifier.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from subqtransformer.config import SubQConfig
from subqtransformer.layers import SubQBlock


class SubQTransformerLM(nn.Module):
    """
    SubQTransformer Autoregressive Language Model.
    
    Combines:
    1. Multi-Scale Sub-Quadratic Relative Candidate Routing O(L * K).
    2. Recurrent Message Passing & Contractive Dynamical Relaxation over learned graphs.
    3. Adaptive Per-Token Compute Halting (velocity / stability early exit).
    4. Flexible Layer Stacking (hierarchical semantic abstraction).
    """
    def __init__(self, config: Optional[SubQConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = SubQConfig(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self.config = config
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.default_T = config.default_T

        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len + 64, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            SubQBlock(config) for _ in range(config.n_layers)
        ])

        # Final LayerNorm & Unembedding Head
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.weight_tying:
            self.head.weight = self.tok_emb.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Returns total parameter count (optionally excluding position embeddings)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
        adaptive_halting: Optional[bool] = None,
        halt_threshold: Optional[float] = None,
        return_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Dict]]:
        """
        Forward pass for language modeling.
        
        Args:
            idx: LongTensor of token IDs (B, L)
            targets: Optional ground truth next token IDs (B, L) for loss calculation
            T: Number of thought hops to unroll per layer (overrides default_T)
            adaptive_halting: Whether to use dynamic early-exit halting
            halt_threshold: Velocity convergence threshold epsilon
            return_stats: Whether to return compute savings & hop distribution metrics
        """
        B, L = idx.shape
        device = idx.device

        pos = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        all_stats = [] if return_stats else None

        for block in self.blocks:
            if return_stats:
                x, stats = block(
                    x,
                    T=T,
                    adaptive_halting=adaptive_halting,
                    halt_threshold=halt_threshold,
                    return_stats=True
                )
                all_stats.append(stats)
            else:
                x = block(
                    x,
                    T=T,
                    adaptive_halting=adaptive_halting,
                    halt_threshold=halt_threshold
                )

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        if return_stats:
            combined_stats = {
                "layer_stats": all_stats,
                "avg_hops_per_layer": [s["avg_hops"] for s in all_stats],
                "mean_total_hops": sum(s["avg_hops"] for s in all_stats),
            }
            if targets is not None:
                return logits, loss, combined_stats
            return logits, combined_stats

        if targets is not None:
            return logits, loss
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        T: Optional[int] = None,
        adaptive_halting: Optional[bool] = None,
        halt_threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation with customizable test-time thought depth (T).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits = self(
                idx_cond,
                T=T,
                adaptive_halting=adaptive_halting,
                halt_threshold=halt_threshold
            )
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx


class SubQTransformerClassifier(nn.Module):
    """
    SubQTransformer Sequence / Graph Reasoning Classifier.
    """
    def __init__(self, num_classes: int, config: Optional[SubQConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = SubQConfig(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self.config = config
        self.num_classes = num_classes
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len + 64, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            SubQBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.d_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
        adaptive_halting: Optional[bool] = None,
        halt_threshold: Optional[float] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L = idx.shape
        device = idx.device

        pos = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(
                x,
                T=T,
                adaptive_halting=adaptive_halting,
                halt_threshold=halt_threshold
            )

        # Pool last token representation
        pooled = self.ln_f(x[:, -1, :])
        logits = self.classifier(pooled)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        return logits

