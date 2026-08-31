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

app = modal.App("gpt2-subq-scaled-adaptation", image=image)

@app.function(gpu="A10G", timeout=2400)
def run_scaled_transplant_adaptation():
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
    print("  SCALED SUBQ-GPT2 (124M) FULL TRANSPLANT ADAPTATION WITH GRADIENT ACCUMULATION & COSINE ANNEALING")
    print("  Setup:")
    print("    - Model: Full Pre-Trained GPT-2 (124M params, 12 Layers x 12 Heads) Transplanted into SubQ")
    print("    - Attention: O(L * K) SubQ Jump Surfing (K=32, T=4 Multi-Hop Hops)")
    print("    - Optimization: 1,500 Steps | Gradient Accumulation = 4 (2,048 tokens/update) | Cosine Annealing")
    print("    - Hardware: NVIDIA A10G (24GB VRAM) with Mixed Precision (FP16)")
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

    # 2. Text Dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = requests.get(url).text
    tokens = tokenizer.encode(raw_text)
    data_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

    n_total = len(data_tensor)
    n_train = int(n_total * 0.85)
    train_data = data_tensor[:n_train]
    test_data = data_tensor[n_train:]

    seq_len = 512
    print(f"Dataset Loaded: {n_total:,} tokens | Train: {len(train_data):,} | Test: {len(test_data):,} | Sequence Length: {seq_len}\n")

    # 3. Data-Driven Offset Menu (K=32: Local dense 0..7 + Geometric power jumps up to seq_len)
    def get_menu(K=32):
        local_k = 8
        local_offsets = list(range(local_k))
        long_jumps = []
        val = local_k
        while len(local_offsets) + len(long_jumps) < K and val < seq_len:
            long_jumps.append(val)
            val = int(val * 1.4) + 1
        menu = local_offsets + long_jumps
        while len(menu) < K:
            menu.append(menu[-1] + 1)
        return torch.tensor(menu[:K], dtype=torch.long, device=device)

    # 4. Memory-Efficient SubQ Jump Attention Module
    class SubQJumpAttention(nn.Module):
        def __init__(self, d_model=768, n_heads=12, K=32, T_max=4, eps=0.01):
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
            self.register_buffer("offsets", get_menu(K))

            # Learnable Temperature Rescaling per head
            self.temp_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

            # GRU Gates for Multi-Hop State Settling
            self.w_gate_h = nn.Linear(d_model, 2 * d_model, bias=True)
            self.w_cand_h = nn.Linear(d_model, d_model, bias=True)
            self.w_ih = nn.Linear(d_model, 3 * d_model, bias=True)

            with torch.no_grad():
                self.w_gate_h.bias.data.fill_(0.0)
                self.w_gate_h.bias.data[self.d_model:].fill_(3.0)
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
                qkv = self.c_attn(s)
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, L, head_dim]
                k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, L, head_dim]
                v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, L, head_dim]

                # Shifted Gather over K relative offsets
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
                        k_shifted = F.pad(k[:, :, :-d, :], (0, 0, d, 0))
                        v_shifted = F.pad(v[:, :, :-d, :], (0, 0, d, 0))
                        mask = torch.cat([
                            torch.zeros(B, 1, d, device=x.device, dtype=torch.bool),
                            torch.ones(B, 1, L - d, device=x.device, dtype=torch.bool)
                        ], dim=-1)

                    k_shifted_list.append(k_shifted)
                    v_shifted_list.append(v_shifted)
                    valid_mask_list.append(mask)

                # Stack candidates: [B, H, L, K, head_dim]
                K_cand = torch.stack(k_shifted_list, dim=3)
                V_cand = torch.stack(v_shifted_list, dim=3)
                valid_mask = torch.stack(valid_mask_list, dim=3) # [B, 1, L, K]

                # Attention Scores over K candidates
                q_expanded = q.unsqueeze(3)
                scores = (q_expanded * K_cand).sum(dim=-1) / math.sqrt(self.head_dim)
                scores = scores * self.temp_scale
                scores = scores.masked_fill(~valid_mask, float("-inf"))
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

                # Output Context
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

                # Early Stopping
                delta_energy = torch.norm(s - s_prev, p=2, dim=-1).mean().item()
                if delta_energy < eps and t >= 1:
                    break

            return s, hops_taken

    # 5. Full SubQ-GPT2 Model
    class SubQGPT2Model(nn.Module):
        def __init__(self, gpt2_ref, K=32, T_max=4, eps=0.01):
            super().__init__()
            self.config = gpt2_ref.config
            self.wte = nn.Embedding.from_pretrained(gpt2_ref.transformer.wte.weight.clone(), freeze=False)
            self.wpe = nn.Embedding.from_pretrained(gpt2_ref.transformer.wpe.weight.clone(), freeze=False)
            self.drop = nn.Dropout(gpt2_ref.config.embd_pdrop)
            self.ln_f = nn.LayerNorm(768, eps=gpt2_ref.config.layer_norm_epsilon)
            self.ln_f.weight.data.copy_(gpt2_ref.transformer.ln_f.weight.data)
            self.ln_f.bias.data.copy_(gpt2_ref.transformer.ln_f.bias.data)

            self.lm_head = nn.Linear(768, vocab_size, bias=False)
            self.lm_head.weight.data.copy_(gpt2_ref.lm_head.weight.data)

            # Transplant 12 blocks
            self.blocks = nn.ModuleList()
            for l in range(12):
                ref_block = gpt2_ref.transformer.h[l]
                subq_attn = SubQJumpAttention(d_model=768, n_heads=12, K=K, T_max=T_max, eps=eps)

                with torch.no_grad():
                    subq_attn.c_attn.weight.copy_(ref_block.attn.c_attn.weight.t())
                    subq_attn.c_attn.bias.copy_(ref_block.attn.c_attn.bias)
                    subq_attn.c_proj.weight.copy_(ref_block.attn.c_proj.weight.t())
                    subq_attn.c_proj.bias.copy_(ref_block.attn.c_proj.bias)

                ln1 = nn.LayerNorm(768, eps=gpt2_ref.config.layer_norm_epsilon)
                ln1.weight.data.copy_(ref_block.ln_1.weight.data)
                ln1.bias.data.copy_(ref_block.ln_1.bias.data)

                ln2 = nn.LayerNorm(768, eps=gpt2_ref.config.layer_norm_epsilon)
                ln2.weight.data.copy_(ref_block.ln_2.weight.data)
                ln2.bias.data.copy_(ref_block.ln_2.bias.data)

                c_fc = nn.Linear(768, 4 * 768)
                c_fc.weight.data.copy_(ref_block.mlp.c_fc.weight.t())
                c_fc.bias.data.copy_(ref_block.mlp.c_fc.bias)

                c_proj = nn.Linear(4 * 768, 768)
                c_proj.weight.data.copy_(ref_block.mlp.c_proj.weight.t())
                c_proj.bias.data.copy_(ref_block.mlp.c_proj.bias)

                mlp = nn.Sequential(
                    c_fc,
                    nn.GELU(approximate="tanh"),
                    c_proj,
                    nn.Dropout(gpt2_ref.config.resid_pdrop)
                )

                block_dict = nn.ModuleDict({
                    "ln_1": ln1,
                    "attn": subq_attn,
                    "ln_2": ln2,
                    "mlp": mlp
                })
                self.blocks.append(block_dict)

        def forward(self, input_ids, T_max=None, eps=None):
            B, L = input_ids.shape
            pos = torch.arange(0, L, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            hidden_states = self.wte(input_ids) + self.wpe(pos)
            hidden_states = self.drop(hidden_states)

            total_hops = 0
            for block in self.blocks:
                norm_1 = block["ln_1"](hidden_states)
                attn_out, hops = block["attn"](norm_1, T_max=T_max, eps=eps)
                hidden_states = hidden_states + attn_out
                total_hops += hops

                norm_2 = block["ln_2"](hidden_states)
                mlp_out = block["mlp"](norm_2)
                hidden_states = hidden_states + mlp_out

            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            avg_hops = total_hops / len(self.blocks)
            return logits, avg_hops

    def evaluate(model_eval, is_subq=False, T_max=4, eps=0.01, n_eval=20):
        model_eval.eval()
        loss_sum = 0.0
        hops_sum = 0.0
        with torch.no_grad():
            for i in range(n_eval):
                idx = i * seq_len
                bx = test_data[idx : idx + seq_len].unsqueeze(0)
                by = test_data[idx + 1 : idx + seq_len + 1].unsqueeze(0)

                if is_subq:
                    logits, hops = model_eval(bx, T_max=T_max, eps=eps)
                    hops_sum += hops
                else:
                    logits = model_eval(bx).logits

                loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
                loss_sum += loss.item()

        avg_loss = loss_sum / n_eval
        avg_ppl = math.exp(avg_loss)
        avg_hops = (hops_sum / n_eval) if is_subq else 1.0
        return avg_loss, avg_ppl, avg_hops

    # 6. Dense GPT-2 Baseline
    dense_nll, dense_ppl, _ = evaluate(gpt2_dense, is_subq=False)
    print("=" * 125)
    print(f"  REFERENCE TARGET: Standard Pre-Trained Dense GPT-2 (Full Attention): PPL = {dense_ppl:.2f} (NLL: {dense_nll:.4f})")
    print("=" * 125)

    # 7. Initialize SubQ-GPT2 (K=32, T_max=4)
    print("\nInitializing SubQ-GPT2 (124M) with transplanted GPT-2 weights...")
    subq_model = SubQGPT2Model(gpt2_dense, K=32, T_max=4, eps=0.01).to(device)

    # Measure Initial Zero-Shot PPL
    init_nll, init_ppl, init_hops = evaluate(subq_model, is_subq=True, T_max=4, eps=0.01)
    print(f"  --> Initial Zero-Shot SubQ-GPT2: PPL = {init_ppl:.2f} (Avg Hops: {init_hops:.2f})\n")

    # 8. Training Setup: 1,500 Steps, Accumulation=4, Cosine Decay
    for p in subq_model.parameters():
        p.requires_grad = True

    total_steps = 1500
    warmup_steps = 100
    accum_steps = 4
    batch_size = 1
    max_lr = 1.5e-4
    min_lr = 1.0e-5

    optimizer = torch.optim.AdamW(subq_model.parameters(), lr=max_lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    def get_lr(step):
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    print("=" * 125)
    print(f"  STARTING SCALED ADAPTATION: {total_steps} Steps | Accum={accum_steps} (2,048 tokens/step) | Cosine LR ({max_lr:.1e} -> {min_lr:.1e})")
    print("=" * 125)
    print(f"{'Step':<12} | {'SubQ-GPT2 PPL':<18} | {'NLL Loss':<14} | {'Learning Rate':<16} | {'Avg Hops (T)':<14} | {'Gap to Target'}")
    print("-" * 125)

    t0 = time.time()
    optimizer.zero_grad()
    best_ppl = init_ppl

    for step in range(1, total_steps + 1):
        subq_model.train()

        # Update LR
        lr_now = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now

        idx = (step * batch_size * seq_len) % (len(train_data) - batch_size * seq_len - 1)
        bx = train_data[idx : idx + batch_size * seq_len].view(batch_size, seq_len)
        by = train_data[idx + 1 : idx + batch_size * seq_len + 1].view(batch_size, seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            # Forward Teacher
            with torch.no_grad():
                teacher_logits = gpt2_dense(bx).logits

            # Forward Student
            student_logits, _ = subq_model(bx, T_max=4, eps=0.01)

            # Combined Loss: Hard Cross-Entropy + Soft Distillation
            loss_ce = F.cross_entropy(student_logits.view(-1, vocab_size), by.view(-1))
            loss_kl = F.kl_div(
                F.log_softmax(student_logits / 1.5, dim=-1),
                F.softmax(teacher_logits / 1.5, dim=-1),
                reduction="batchmean"
            ) * (1.5 ** 2)

            loss_total = (loss_ce + 0.5 * loss_kl) / accum_steps

        scaler.scale(loss_total).backward()

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(subq_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Evaluate every 100 steps
        if step % 100 == 0 or step == total_steps:
            val_nll, val_ppl, val_hops = evaluate(subq_model, is_subq=True, T_max=4, eps=0.01)
            gap = val_ppl - dense_ppl
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                mark = " ⭐ [New Best]"
            else:
                mark = ""
            print(f"Step {step:>4}/{total_steps} | {val_ppl:>14.2f}     | {val_nll:>10.4f}     | {lr_now:>12.2e}     | T = {val_hops:>8.2f}   | {gap:>+14.2f} PPL{mark}")

    print(f"\nScaled Adaptation completed in {time.time() - t0:.1f}s")
    print("=" * 125)
    final_loss, final_ppl, final_hops = evaluate(subq_model, is_subq=True, T_max=4, eps=0.01)
    print(f"FINAL RESULT: SubQ-GPT2 (124M, K=32, T={final_hops:.2f}) reached Best PPL = {best_ppl:.2f} (Final: {final_ppl:.2f}) vs Dense GPT-2 PPL = {dense_ppl:.2f}")
    print(f"Total Structural Gap Closed: from {init_ppl:.2f} down to {best_ppl:.2f} PPL!")
    print("=" * 125)

@app.local_entrypoint()
def main():
    run_scaled_transplant_adaptation.remote()
