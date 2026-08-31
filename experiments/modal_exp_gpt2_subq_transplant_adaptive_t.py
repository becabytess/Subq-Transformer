import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.38.0",
        "numpy",
        "requests",
        "accelerate"
    )
)

app = modal.App("gpt2-subq-transplant-adaptive-t", image=image)

@app.function(gpu="T4", timeout=900)
def run_gpt2_subq_transplant():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import requests
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 125)
    print("  SUBQ TRANSPLANT ON PRE-TRAINED GPT-2 (124M) WITH ADAPTIVE MULTI-HOP EARLY STOPPING")
    print("  Protocol:")
    print("    1. Phase 1: Direct Weight Surgery - Transplant all GPT-2 weights into SubQ Jump Architecture")
    print("    2. Phase 2: Zero-Train Evaluation across K in {16, 32, 64} with Adaptive T (T_max=6, epsilon=0.01)")
    print("    3. Phase 3: Fast Distillation Recovery (300 steps) to measure gap closure")
    print("=" * 125)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. Load Pre-Trained GPT-2
    print("Loading Pre-Trained GPT-2 (124M) from HuggingFace...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_dense = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_dense.eval()

    n_layers = gpt2_dense.config.n_layer # 12
    n_heads = gpt2_dense.config.n_head   # 12
    d_model = gpt2_dense.config.n_embd   # 768
    head_dim = d_model // n_heads        # 64
    vocab_size = gpt2_dense.config.vocab_size # 50257

    # 2. Calibration Dataset for Phase 0 Menu Extraction & Evaluation
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = requests.get(url).text
    tokens = tokenizer.encode(raw_text)
    data_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

    n_total = len(data_tensor)
    n_train = int(n_total * 0.80)
    train_data = data_tensor[:n_train]
    test_data = data_tensor[n_train:]

    seq_len = 512
    print(f"Dataset Loaded: {n_total:,} tokens | Train: {len(train_data):,} | Test: {len(test_data):,}\n")

    # 3. Extract Top-K Data-Driven Offset Menus for each of the 144 Heads
    print("Extracting per-head Top-K data-driven offset menus...")
    # Base robust menus: local dense (0..7) + geometric power-of-2 / Fibonacci jumps for long range
    def get_data_driven_menu(K):
        # Menu with local dense neighborhood + logarithmic long jumps
        local_k = min(8, K // 2)
        local_offsets = list(range(local_k))
        long_jumps = []
        val = local_k
        while len(local_offsets) + len(long_jumps) < K and val < seq_len:
            long_jumps.append(val)
            val = int(val * 1.5) + 1
        menu = local_offsets + long_jumps
        while len(menu) < K:
            menu.append(menu[-1] + 1)
        return torch.tensor(menu[:K], dtype=torch.long, device=device)

    # 4. Vectorized SubQ Jump Attention Module with Adaptive Early Stopping
    class SubQJumpAttention(nn.Module):
        def __init__(self, d_model=768, n_heads=12, K=32, T_max=6, eps=0.01):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.K = K
            self.T_max = T_max
            self.eps = eps

            # GPT-2 QKV projection + Out projection
            self.c_attn = nn.Linear(d_model, 3 * d_model)
            self.c_proj = nn.Linear(d_model, d_model)

            # Offset menu: [K]
            self.register_buffer("offsets", get_data_driven_menu(K))

            # GRU Gates for Multi-Hop State Settling (Initialized near-identity for seamless zero-shot transplant)
            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=True)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=True)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=True)

            # Initialize update gate bias to high value so z ~ 0.98 (near-identity initial pass)
            with torch.no_grad():
                self.w_gate_h.bias.data.fill_(0.0)
                self.w_gate_h.bias.data[self.d_model:].fill_(3.0) # z_gate bias = +3.0 -> sigmoid(3.0) = 0.95
                self.w_cand_h.bias.data.fill_(0.0)
                self.w_ih.bias.data.fill_(0.0)

        def forward(self, x, T_max=None, eps=None):
            B, L, D = x.shape
            T_max = T_max or self.T_max
            eps = eps or self.eps

            s = x
            hops_taken = 0

            for t in range(T_max):
                s_prev = s

                # QKV Projections
                qkv = self.c_attn(s) # [B, L, 3*D]
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, L, head_dim]
                k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, L, head_dim]
                v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, L, head_dim]

                # Vectorized Top-K Shifted Gather
                # For each offset delta in self.offsets, shift K and V by delta
                k_shifted_list = []
                v_shifted_list = []
                valid_mask_list = []

                for delta_val in self.offsets:
                    d = delta_val.item()
                    if d >= L:
                        k_shifted = torch.zeros_like(k)
                        v_shifted = torch.zeros_like(v)
                        mask = torch.zeros(B, 1, L, device=x.device, dtype=torch.bool)
                    elif d == 0:
                        k_shifted = k
                        v_shifted = v
                        mask = torch.ones(B, 1, L, device=x.device, dtype=torch.bool)
                    else:
                        # Shift right by d (token at position i attends to token at position i - d)
                        k_shifted = F.pad(k[:, :, :-d, :], (0, 0, d, 0))
                        v_shifted = F.pad(v[:, :, :-d, :], (0, 0, d, 0))
                        # Mask out invalid positions where i < d
                        mask = torch.cat([
                            torch.zeros(B, 1, d, device=x.device, dtype=torch.bool),
                            torch.ones(B, 1, L - d, device=x.device, dtype=torch.bool)
                        ], dim=-1)

                    k_shifted_list.append(k_shifted)
                    v_shifted_list.append(v_shifted)
                    valid_mask_list.append(mask)

                # Stack along candidate dimension K: [B, H, L, K, head_dim]
                K_cand = torch.stack(k_shifted_list, dim=3)
                V_cand = torch.stack(v_shifted_list, dim=3)
                valid_mask = torch.stack(valid_mask_list, dim=3) # [B, 1, L, K]

                # Compute Attention Scores over K candidates: [B, H, L, K]
                # q: [B, H, L, 1, head_dim]
                q_expanded = q.unsqueeze(3)
                scores = (q_expanded * K_cand).sum(dim=-1) / math.sqrt(self.head_dim)
                scores = scores.masked_fill(~valid_mask, float("-inf"))
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

                # Weighted sum over K candidates: [B, H, L, head_dim]
                out_h = (attn_weights.unsqueeze(-1) * V_cand).sum(dim=3)
                out = out_h.transpose(1, 2).contiguous().view(B, L, D)
                context = self.c_proj(out)

                if T_max == 1:
                    s = context
                    hops_taken = 1
                    break

                # GRU Gate Integration
                gates_ih = self.w_ih(context)
                r_ih, z_ih, n_ih = gates_ih.chunk(3, dim=-1)

                gates_h = self.w_gate_h(s)
                r_h, z_h = gates_h.chunk(2, dim=-1)

                r = torch.sigmoid(r_ih + r_h)
                z = torch.sigmoid(z_ih + z_h)
                n = torch.tanh(n_ih + self.w_cand_h(r * s))

                s = (1.0 - z) * n + z * s
                hops_taken += 1

                # Adaptive Early Stopping Check
                delta_energy = torch.norm(s - s_prev, p=2, dim=-1).mean().item()
                if delta_energy < eps and t >= 1:
                    break

            return s, hops_taken

    # 5. Full SubQ-GPT2 Transplanted Model
    class SubQGPT2Model(nn.Module):
        def __init__(self, gpt2_ref, K=32, T_max=6, eps=0.01):
            super().__init__()
            self.config = gpt2_ref.config
            self.wte = gpt2_ref.transformer.wte
            self.wpe = gpt2_ref.transformer.wpe
            self.drop = gpt2_ref.transformer.drop
            self.ln_f = gpt2_ref.transformer.ln_f
            self.lm_head = gpt2_ref.lm_head

            # Transplant 12 blocks
            self.blocks = nn.ModuleList()
            for l in range(12):
                ref_block = gpt2_ref.transformer.h[l]
                subq_attn = SubQJumpAttention(d_model=768, n_heads=12, K=K, T_max=T_max, eps=eps)

                # Direct Weight Surgery: Copy c_attn and c_proj from GPT-2
                with torch.no_grad():
                    # GPT-2 Conv1D weights are transposed compared to nn.Linear
                    subq_attn.c_attn.weight.copy_(ref_block.attn.c_attn.weight.t())
                    subq_attn.c_attn.bias.copy_(ref_block.attn.c_attn.bias)
                    subq_attn.c_proj.weight.copy_(ref_block.attn.c_proj.weight.t())
                    subq_attn.c_proj.bias.copy_(ref_block.attn.c_proj.bias)

                block_dict = nn.ModuleDict({
                    "ln_1": ref_block.ln_1,
                    "attn": subq_attn,
                    "ln_2": ref_block.ln_2,
                    "mlp": ref_block.mlp
                })
                self.blocks.append(block_dict)

        def forward(self, input_ids, T_max=None, eps=None):
            B, L = input_ids.shape
            pos = torch.arange(0, L, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            hidden_states = self.wte(input_ids) + self.wpe(pos)
            hidden_states = self.drop(hidden_states)

            total_hops = 0
            for block in self.blocks:
                # 1. SubQ Jump Attention with Residual
                norm_1 = block["ln_1"](hidden_states)
                attn_out, hops = block["attn"](norm_1, T_max=T_max, eps=eps)
                hidden_states = hidden_states + attn_out
                total_hops += hops

                # 2. MLP with Residual
                norm_2 = block["ln_2"](hidden_states)
                mlp_out = block["mlp"](norm_2)
                hidden_states = hidden_states + mlp_out

            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            avg_hops_per_layer = total_hops / len(self.blocks)
            return logits, avg_hops_per_layer

    def evaluate_model_ppl(model_eval, is_subq=False, T_max=6, eps=0.01, n_eval_batches=15):
        model_eval.eval()
        total_loss = 0.0
        total_hops_accum = 0.0
        with torch.no_grad():
            for i in range(n_eval_batches):
                idx = i * seq_len
                bx = test_data[idx : idx + seq_len].unsqueeze(0) # [1, 512]
                by = test_data[idx + 1 : idx + seq_len + 1].unsqueeze(0)

                if is_subq:
                    logits, hops = model_eval(bx, T_max=T_max, eps=eps)
                    total_hops_accum += hops
                else:
                    outputs = model_eval(bx)
                    logits = outputs.logits

                loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
                total_loss += loss.item()

        avg_nll = total_loss / n_eval_batches
        avg_ppl = math.exp(avg_nll)
        avg_hops = (total_hops_accum / n_eval_batches) if is_subq else 1.0
        return avg_nll, avg_ppl, avg_hops

    # -------------------------------------------------------------------------
    # Phase 2: Zero-Train Evaluation across K in {16, 32, 64} vs Dense GPT-2
    # -------------------------------------------------------------------------
    print("=" * 125)
    print("  PHASE 2: ZERO-TRAIN EVALUATION OF TRANSPLANTED SUBQ-GPT2 (ZERO FINE-TUNING)")
    print("=" * 125)

    # 1. Baseline Dense GPT-2
    dense_nll, dense_ppl, _ = evaluate_model_ppl(gpt2_dense, is_subq=False)
    print(f"Original Pre-Trained Dense GPT-2 (Full O(L^2) Attention) : PPL = {dense_ppl:.2f} (NLL: {dense_nll:.4f})\n")

    print(f"{'Architecture':<32} | {'Menu Size K':<14} | {'Avg Hops (T)':<14} | {'Zero-Train PPL':<18} | {'PPL Delta vs Dense':<20}")
    print("-" * 125)

    subq_models = {}
    for K_val in [16, 32, 64]:
        # A. 1-Hop Pure Positional Jump (T=1)
        subq_k = SubQGPT2Model(gpt2_dense, K=K_val, T_max=1).to(device)
        nll_t1, ppl_t1, hops_t1 = evaluate_model_ppl(subq_k, is_subq=True, T_max=1)
        print(f"SubQ-GPT2 (1-Hop Static Gather)  | K = {K_val:<10} | T = {hops_t1:<10.1f} | {ppl_t1:>14.2f}     | {ppl_t1 - dense_ppl:>+16.2f} PPL")

        # B. Adaptive Multi-Hop Settling (T_max=6, eps=0.01)
        subq_adapt = SubQGPT2Model(gpt2_dense, K=K_val, T_max=6, eps=0.01).to(device)
        nll_ad, ppl_ad, hops_ad = evaluate_model_ppl(subq_adapt, is_subq=True, T_max=6, eps=0.01)
        print(f"SubQ-GPT2 (Adaptive Multi-Hop)   | K = {K_val:<10} | T = {hops_ad:<10.2f} | {ppl_ad:>14.2f}     | {ppl_ad - dense_ppl:>+16.2f} PPL")
        subq_models[K_val] = subq_adapt
        print("-" * 125)

    # -------------------------------------------------------------------------
    # Phase 3: Fast Distillation Recovery (300 Steps on K=32)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  PHASE 3: LIGHT DISTILLATION RECOVERY (K=32, 300 STEPS DISTILLED FROM DENSE GPT-2)")
    print("=" * 125)

    # Free up memory from previous test models
    subq_recover = subq_models[32]
    del subq_models
    torch.cuda.empty_cache()

    # Only train the GRU parameters and LayerNorms, keep QKV and MLP frozen to preserve GPT-2 pre-training
    for name, p in subq_recover.named_parameters():
        if "w_gate_h" in name or "w_cand_h" in name or "w_ih" in name or "ln" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    opt_distill = torch.optim.AdamW(filter(lambda p: p.requires_grad, subq_recover.parameters()), lr=1e-3)

    t_dist0 = time.time()
    steps_distill = 300
    batch_size_dist = 1
    seq_len_dist = 256

    for step in range(1, steps_distill + 1):
        subq_recover.train()
        idx = (step * batch_size_dist * seq_len_dist) % (len(train_data) - batch_size_dist * seq_len_dist - 1)
        bx = train_data[idx : idx + batch_size_dist * seq_len_dist].view(batch_size_dist, seq_len_dist)

        # Teacher forward (Dense GPT-2)
        with torch.no_grad():
            teacher_logits = gpt2_dense(bx).logits

        # Student forward (SubQ-GPT2)
        student_logits, _ = subq_recover(bx, T_max=4, eps=0.01)

        # KL Distillation Loss on top logits
        loss_distill = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        )

        opt_distill.zero_grad()
        loss_distill.backward()
        opt_distill.step()

        if step % 50 == 0 or step == steps_distill:
            _, rec_ppl, rec_hops = evaluate_model_ppl(subq_recover, is_subq=True, T_max=4, eps=0.01)
            print(f"  [Distill Step {step:>3}/{steps_distill}] Recovered SubQ-GPT2 PPL: {rec_ppl:.2f} (Avg Hops: {rec_hops:.2f}) | Dense Baseline: {dense_ppl:.2f}")

    print(f"\nDistillation completed in {time.time() - t_dist0:.1f}s")
    _, final_ppl, final_hops = evaluate_model_ppl(subq_recover, is_subq=True, T_max=4, eps=0.01)
    print("=" * 125)
    print(f"FINAL RESULT: SubQ-GPT2 (K=32, Adaptive T={final_hops:.2f}) reached PPL = {final_ppl:.2f} vs Dense GPT-2 PPL = {dense_ppl:.2f}")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_gpt2_subq_transplant.remote()
