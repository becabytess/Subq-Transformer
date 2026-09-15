import json
import math
import os
import pathlib
import re
import sys
import time
import urllib.request
import modal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "numpy",
        "requests"
    )
)

app = modal.App("exp-s2-041-qwen-harmonic-gsm8k", image=image)
vol = modal.Volume.from_name("subq-qwen-checkpoints", create_if_missing=True)

@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/root/checkpoints": vol})
def run_qwen_harmonic_wave_gsm8k(
    num_eval_samples: int = 150,
    total_steps: int = 600,
    k_peaks: int = 8,
    d_max: int = 256
):
    import math
    import re
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 135, flush=True)
    print("  STUDY S2-041: FOUNDATION LLM (QWEN2.5-0.5B) LAYER-SPECIFIC CONTINUOUS HARMONIC WAVES (T=1) ON GSM8K", flush=True)
    print(f"  Configuration: All 24 Layers | K = {k_peaks} Peaks | d_max = {d_max} Tokens | T = 1 Hop (Depth = 24)", flush=True)
    print("  Grouped Query Attention (14Q / 2KV) | RoPE Rotary Embeddings | SwiGLU MLPs | Continuous Wave Steering", flush=True)
    print(f"  Device: {torch.cuda.get_device_name(0)} (24GB VRAM)", flush=True)
    print("=" * 135, flush=True)

    # -------------------------------------------------------------------------
    # 1. Download & Prepare GSM8K Dataset Directly (JSONL)
    # -------------------------------------------------------------------------
    print("\n[1/5] Downloading GSM8K Dataset (Grade School Math 8K)...", flush=True)
    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

    def fetch_jsonl(url):
        data = []
        req = urllib.request.urlopen(url)
        for line in req.read().decode('utf-8').strip().split('\n'):
            if line.strip():
                data.append(json.loads(line))
        return data

    train_data = fetch_jsonl(train_url)
    test_data = fetch_jsonl(test_url)
    print(f"  GSM8K Loaded: {len(train_data):,} Training Problems | {len(test_data):,} Test Problems", flush=True)

    # -------------------------------------------------------------------------
    # 2. Load Pretrained Qwen2.5-0.5B Tokenizer & Model
    # -------------------------------------------------------------------------
    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"\n[2/5] Loading Pretrained Tokenizer & Model: {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    config = base_model.config
    print(f"  Config: {config.num_hidden_layers} Layers, Hidden Dim={config.hidden_size}, "
          f"Q-Heads={config.num_attention_heads}, KV-Heads={config.num_key_value_heads}, Vocab={config.vocab_size:,}", flush=True)

    num_heads = config.num_attention_heads         # 14
    num_kv_heads = config.num_key_value_heads     # 2
    head_dim = config.hidden_size // num_heads     # 64
    num_key_value_groups = num_heads // num_kv_heads # 7
    num_waves = 12

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, n_kv, slen, h_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv, n_rep, slen, h_dim)
        return hidden_states.reshape(batch, n_kv * n_rep, slen, h_dim)

    # -------------------------------------------------------------------------
    # 3. Define Layer-Specific Continuous Harmonic SubQ Attention
    # -------------------------------------------------------------------------
    class LayerContinuousHarmonicWave(nn.Module):
        """
        Independent continuous harmonic wave generator dedicated to a single layer:
        wave(d) = sum_{m=1}^M A_m * cos(omega_m * d + phi_m) * exp(-gamma_m * d)
        Dynamically extracts top (K-1) peak offsets + offset 0.
        """
        def __init__(self, layer_idx: int, num_waves: int = 12, max_d: int = 256, k_peaks: int = 8):
            super().__init__()
            self.layer_idx = layer_idx
            self.num_waves = num_waves
            self.max_d = max_d
            self.k_peaks = k_peaks

            # Canonical Gaussian initialization (Study 61 / 65 canon)
            self.latents = nn.Parameter(torch.randn(num_waves, 4) * 0.1)

            # Frequency spectrum covering short to long distances
            log_freqs = torch.linspace(0.0, -2.5, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, num_waves))
            self.register_buffer("d_grid", torch.arange(1, max_d).float().view(max_d - 1, 1))

        def forward(self, device=None):
            curr_device = device if device is not None else self.latents.device
            # Compute wave curve over d in [1, max_d - 1]
            amp = torch.tanh(self.latents[:, 0]).view(1, self.num_waves)
            omega = (F.softplus(self.latents[:, 1]).view(1, self.num_waves) * self.base_freqs)
            phi = (self.latents[:, 2] * math.pi).view(1, self.num_waves)
            decay = (F.softplus(self.latents[:, 3]) * 0.05).view(1, self.num_waves)

            # (max_d - 1, num_waves) -> sum -> (max_d - 1,)
            wave_comps = amp * torch.cos(self.d_grid * omega + phi) * torch.exp(-self.d_grid * decay)
            wave_1d = wave_comps.sum(dim=-1)

            topk_vals, past_offsets = torch.topk(wave_1d, k=self.k_peaks - 1, dim=-1)
            past_offsets = past_offsets + 1  # 1-indexed spatial distance

            zero_off = torch.zeros(1, dtype=torch.long, device=curr_device)
            zero_val = torch.zeros(1, dtype=torch.float, device=curr_device)

            offsets = torch.cat([zero_off, past_offsets]) # (K,)
            peak_vals = torch.cat([zero_val, topk_vals])   # (K,)
            return offsets, peak_vals

    class HarmonicSubQQwenAttention(nn.Module):
        """
        Surgical 1-to-1 replacement for Qwen2Attention:
        - T=1 feedforward attention hop
        - Dedicated LayerContinuousHarmonicWave per layer
        - Preserves RoPE rotary positional embeddings
        - Preserves Grouped Query Attention (14 Q heads, 2 KV heads)
        - Fast tensor slicing with differentiable peak logit bias
        """
        def __init__(self, orig_attn, rotary_emb, layer_idx: int, k_peaks: int = 8, max_d: int = 256):
            super().__init__()
            self.layer_idx = layer_idx
            self.q_proj = orig_attn.q_proj
            self.k_proj = orig_attn.k_proj
            self.v_proj = orig_attn.v_proj
            self.o_proj = orig_attn.o_proj
            self.rotary_emb = rotary_emb or getattr(orig_attn, "rotary_emb", None)
            self.head_dim = head_dim
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.num_key_value_groups = num_key_value_groups
            self.scale = 1.0 / math.sqrt(head_dim)
            self.k_peaks = k_peaks
            self.max_d = max_d

            # Dedicated independent harmonic carrier wave for this physical layer
            self.wave_router = LayerContinuousHarmonicWave(
                layer_idx=layer_idx,
                num_waves=num_waves,
                max_d=max_d,
                k_peaks=k_peaks
            )

        def forward(
            self,
            hidden_states,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs
        ):
            B, L, _ = hidden_states.shape
            device = hidden_states.device

            # 1. Linear Projections using Pre-Trained Weights
            query_states = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

            if position_ids is None:
                position_ids = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0)

            # 2. Rotary Position Embeddings (RoPE)
            if position_embeddings is not None:
                cos, sin = position_embeddings
            elif self.rotary_emb is not None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = None, None

            if cos is not None and sin is not None:
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # 3. GQA expansion: Repeat Key/Value heads across query groups (14 heads total)
            key_states = repeat_kv(key_states, self.num_key_value_groups)      # (B, H, L, d_k)
            value_states = repeat_kv(value_states, self.num_key_value_groups)  # (B, H, L, d_k)

            # 4. Generate dynamic harmonic candidate offsets & peak logit biases
            offsets, peak_vals = self.wave_router(device) # (K,), (K,)

            # 5. Fast Direct Tensor Slicing with Causal Masking
            q_pos = torch.arange(L, device=device).unsqueeze(1)    # (L, 1)
            target_indices = q_pos - offsets.view(1, -1)           # (L, K)
            valid_mask = (target_indices >= 0).view(1, 1, L, self.k_peaks) # (1, 1, L, K)
            clamped = torch.clamp(target_indices, min=0)           # (L, K)

            K_g = key_states[:, :, clamped, :]    # (B, H, L, K, d_k)
            V_g = value_states[:, :, clamped, :]  # (B, H, L, K, d_k)

            # 6. Dot-product attention + Harmonic Peak Logit Bias
            scores = (query_states.unsqueeze(3) * K_g).sum(dim=-1) * self.scale # (B, H, L, K)
            scores = scores + peak_vals.view(1, 1, 1, self.k_peaks)            # Differentiable wave steering
            scores = scores.masked_fill(~valid_mask, -1e4)

            attn_weights = F.softmax(scores, dim=-1) * valid_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # 7. Output context gather and projection
            context = (attn_weights.unsqueeze(-1) * V_g).sum(dim=3) # (B, H, L, d_k)
            context = context.transpose(1, 2).contiguous().view(B, L, -1)
            attn_output = self.o_proj(context)

            return attn_output, None

    # -------------------------------------------------------------------------
    # 4. Surgical Replacement Across All 24 Layers
    # -------------------------------------------------------------------------
    print("\n[3/5] Performing 1-to-1 Surgical Replacement of All 24 Layers with Layer-Specific Harmonic Waves...", flush=True)
    base_model.config.use_cache = False
    rotary_emb = getattr(base_model.model, "rotary_emb", None)

    for i, layer in enumerate(base_model.model.layers):
        orig_attn = layer.self_attn
        layer.self_attn = HarmonicSubQQwenAttention(
            orig_attn=orig_attn,
            rotary_emb=rotary_emb,
            layer_idx=i,
            k_peaks=k_peaks,
            max_d=d_max
        ).to(device)
    base_model = base_model.to(device)
    print("  ✅ All 24 layers converted to Harmonic SubQ with independent learned waves!", flush=True)

    # Inspect initial offset distributions for sample layers
    print("\n  Sample Initial Offset Distributions:")
    for l_idx in [0, 6, 12, 18, 23]:
        init_offs, _ = base_model.model.layers[l_idx].self_attn.wave_router(device)
        print(f"    Layer {l_idx:>2} Initial Offsets: {init_offs.cpu().tolist()}")

    # -------------------------------------------------------------------------
    # 5. Format GSM8K for Training
    # -------------------------------------------------------------------------
    def format_prompt(item):
        return f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n{item['answer']}<|im_end|>"

    formatted_texts = [format_prompt(item) for item in train_data]
    tokenized_train = [tokenizer.encode(t, max_length=384, truncation=True) for t in formatted_texts]
    print(f"\n[4/5] Prepared {len(tokenized_train):,} GSM8K training examples (max_len=384).", flush=True)

    # -------------------------------------------------------------------------
    # 6. Fine-Tuning Harmonic SubQ-Qwen2.5-0.5B (600 Steps SFT)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 135, flush=True)
    print("  FINE-TUNING HARMONIC SUBQ-QWEN2.5-0.5B ON GSM8K MATH REASONING (600 STEPS)", flush=True)
    print("=" * 135, flush=True)

    base_model.train()
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4, weight_decay=0.01)
    batch_size = 4
    grad_accum_steps = 4  # Effective batch size = 16
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    step = 0
    t0 = time.time()
    accum_loss = 0.0
    train_history = []

    while step < total_steps:
        indices = torch.randint(0, len(tokenized_train), (batch_size,))
        batch_seqs = [tokenized_train[idx] for idx in indices]
        max_b_len = max(len(s) for s in batch_seqs)

        input_ids = torch.full((batch_size, max_b_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_b_len), -100, dtype=torch.long, device=device)

        for b_idx, seq in enumerate(batch_seqs):
            input_ids[b_idx, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            labels[b_idx, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = base_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps

        loss.backward()
        accum_loss += loss.item() * grad_accum_steps

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 50 == 0:
            avg_l = accum_loss / 50.0
            ppl = math.exp(min(15.0, avg_l))
            elapsed = time.time() - t0
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            current_lr = lr_scheduler.get_last_lr()[0]
            print(f"  Step {step+1:>3}/{total_steps} | Loss: {avg_l:.4f} | Train PPL: {ppl:>6.2f} | "
                  f"LR: {current_lr:.2e} | VRAM: {peak_vram:.1f} MB | Elapsed: {elapsed:.1f}s", flush=True)
            train_history.append({
                "step": step + 1,
                "loss": avg_l,
                "ppl": ppl,
                "elapsed_sec": elapsed
            })
            accum_loss = 0.0

        step += 1

    train_time = time.time() - t0
    ckpt_path = "/root/checkpoints/subq_qwen2_5_05b_s2_041_harmonic.pt"
    print(f"\nSaving fine-tuned Harmonic SubQ checkpoint to {ckpt_path}...", flush=True)
    torch.save(base_model.state_dict(), ckpt_path)
    vol.commit()

    # -------------------------------------------------------------------------
    # 7. Post-Training Learned Wave Analysis Across All 24 Layers
    # -------------------------------------------------------------------------
    print("\n" + "=" * 135, flush=True)
    print("  POST-TRAINING LEARNED WAVE INSPECTION ACROSS ALL 24 LAYERS", flush=True)
    print("=" * 135, flush=True)
    learned_offsets_summary = {}

    for i in range(24):
        layer_attn = base_model.model.layers[i].self_attn
        final_offs, final_peaks = layer_attn.wave_router(device)
        offs_list = final_offs.cpu().tolist()
        peaks_list = [round(p, 3) for p in final_peaks.cpu().tolist()]
        learned_offsets_summary[f"layer_{i}"] = {
            "offsets": offs_list,
            "peak_biases": peaks_list
        }
        if i in [0, 3, 6, 9, 12, 15, 18, 21, 23]:
            print(f"  Layer {i:>2} Final Offsets: {offs_list} | Peaks: {peaks_list}", flush=True)

    # -------------------------------------------------------------------------
    # 8. Rigorous Quantitative Evaluation on GSM8K Test Set (Greedy Decoding)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 135, flush=True)
    print(f"  [5/5] EVALUATING HARMONIC SUBQ-QWEN2.5-0.5B ON {num_eval_samples} UNSEEN GSM8K TEST PROBLEMS", flush=True)
    print("=" * 135, flush=True)

    def extract_answer_number(text: str) -> str:
        if "####" in text:
            ans_part = text.split("####")[-1].strip()
            match = re.search(r"(-?[\d,]+(?:\.\d+)?)", ans_part)
            if match:
                return match.group(1).replace(",", "")
        numbers = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
        if numbers:
            return numbers[-1].replace(",", "")
        return ""

    base_model.eval()
    correct = 0
    total = 0
    has_delimiter = 0
    total_gen_tokens = 0
    eval_t0 = time.time()
    sample_solutions = []

    for idx, item in enumerate(test_data[:num_eval_samples]):
        question = item["question"]
        gold_answer_text = item["answer"]
        gold_num = extract_answer_number(gold_answer_text)

        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        curr_ids = input_ids.clone()
        generated_tokens = []

        with torch.no_grad():
            for _ in range(160):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = base_model(curr_ids)
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                if next_token.item() == tokenizer.eos_token_id or "<|im_end|>" in tokenizer.decode([next_token.item()]):
                    break

                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())

        gen_text = tokenizer.decode(generated_tokens)
        pred_num = extract_answer_number(gen_text)

        is_correct = (pred_num != "" and pred_num == gold_num)
        if is_correct:
            correct += 1
        if "####" in gen_text:
            has_delimiter += 1

        total += 1
        total_gen_tokens += len(generated_tokens)

        if idx < 3:
            sample_solutions.append({
                "problem_idx": idx + 1,
                "question": question[:120] + "...",
                "gold_answer": gold_num,
                "predicted_answer": pred_num,
                "is_correct": is_correct,
                "generated_text": gen_text.strip()
            })

        if (idx + 1) % 25 == 0 or (idx + 1) == num_eval_samples:
            cur_acc = (correct / total) * 100.0
            cur_fmt = (has_delimiter / total) * 100.0
            print(f"  Progress {idx+1:>3}/{num_eval_samples} | "
                  f"Exact Accuracy: {cur_acc:>5.2f}% ({correct}/{total}) | "
                  f"Delimiter Adherence: {cur_fmt:>5.1f}%", flush=True)

    eval_time = time.time() - eval_t0
    exact_acc = (correct / total) * 100.0
    delimiter_rate = (has_delimiter / total) * 100.0
    eval_tok_per_sec = total_gen_tokens / max(1e-5, eval_time)

    print("\n" + "=" * 135, flush=True)
    print("  STUDY S2-041 COMPLETE BENCHMARK SUMMARY", flush=True)
    print("=" * 135, flush=True)
    print(f"  Exact Math Accuracy:       {exact_acc:.2f}% ({correct}/{total})", flush=True)
    print(f"  Delimiter Adherence (####): {delimiter_rate:.1f}%", flush=True)
    print(f"  Training Time (600 steps): {train_time:.1f}s", flush=True)
    print(f"  Evaluation Time (150 prob): {eval_time:.1f}s ({eval_tok_per_sec:.1f} tok/s)", flush=True)

    results = {
        "study": "S2-041",
        "description": "Qwen2.5-0.5B Foundation LLM with Layer-Specific Continuous Harmonic Waves (T=1) on GSM8K",
        "base_model": model_id,
        "config": {
            "num_layers": 24,
            "T_hops_per_layer": 1,
            "total_hops": 24,
            "k_peaks": k_peaks,
            "d_max": d_max,
            "training_steps": total_steps,
            "eval_samples": num_eval_samples
        },
        "metrics": {
            "exact_accuracy_pct": exact_acc,
            "correct_count": correct,
            "total_count": total,
            "delimiter_adherence_pct": delimiter_rate,
            "training_time_sec": train_time,
            "eval_time_sec": eval_time,
            "eval_tok_per_sec": eval_tok_per_sec
        },
        "train_history": train_history,
        "learned_offsets": learned_offsets_summary,
        "sample_solutions": sample_solutions,
        "historical_baselines": {
            "dense_qwen2_5_05b_study44": {
                "exact_accuracy_pct": 11.33,
                "correct_count": 17,
                "total_count": 150,
                "delimiter_adherence_pct": 71.3
            },
            "hardcoded_subq_study44": {
                "exact_accuracy_pct": 1.33,
                "correct_count": 2,
                "total_count": 150,
                "delimiter_adherence_pct": 72.7
            },
            "recurrent_t6_subq_study45": {
                "exact_accuracy_pct": 0.00,
                "correct_count": 0,
                "total_count": 150,
                "delimiter_adherence_pct": 40.7
            }
        }
    }

    return results

@app.local_entrypoint()
def main():
    print("Launching Study S2-041 on Modal NVIDIA A10G...", flush=True)
    res = run_qwen_harmonic_wave_gsm8k.remote()

    output_dir = pathlib.Path("season2/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "s2_041_qwen_harmonic_wave_gsm8k.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"\n✅ Results successfully saved to {out_file}", flush=True)
    print(f"Final Accuracy: {res['metrics']['exact_accuracy_pct']:.2f}% | "
          f"Delimiter Adherence: {res['metrics']['delimiter_adherence_pct']:.1f}%")

if __name__ == "__main__":
    main()
