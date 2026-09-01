import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "numpy"
    )
)

app = modal.App("exp-qwen-gsm8k-eval", image=image)
vol = modal.Volume.from_name("subq-qwen-checkpoints", create_if_missing=True)

@app.function(gpu="A10G", timeout=2400, volumes={"/root/checkpoints": vol})
def evaluate_qwen_gsm8k():
    import json
    import math
    import os
    import re
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 135)
    print("  STUDY 44: RIGOROUS QUANTITATIVE GSM8K MATH REASONING BENCHMARK: DENSE QWEN2.5-0.5B VS. SUBQ-QWEN2.5-0.5B")
    print("  Full Exact-Match Extraction (#### <number>), Accuracy Scorecard, and Generation Speed on Unseen Test Set")
    print("=" * 135)
    print(f"Device: {torch.cuda.get_device_name(0)} (24GB VRAM)")

    # -------------------------------------------------------------------------
    # 1. Download GSM8K Test Dataset
    # -------------------------------------------------------------------------
    print("\n[1/5] Downloading GSM8K Dataset...")
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
    print(f"Dataset: {len(train_data):,} Training Problems | {len(test_data):,} Test Problems")

    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_prompt(item):
        return f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n{item['answer']}<|im_end|>"

    formatted_texts = [format_prompt(item) for item in train_data]
    tokenized_train = [tokenizer.encode(t, max_length=384, truncation=True) for t in formatted_texts]

    # -------------------------------------------------------------------------
    # 2. Train Dense Qwen2.5-0.5B Baseline (Controlled 600 Steps) if not cached
    # -------------------------------------------------------------------------
    dense_ckpt_path = "/root/checkpoints/dense_qwen2_5_05b_gsm8k.pt"
    subq_ckpt_path = "/root/checkpoints/subq_qwen2_5_05b_gsm8k.pt"

    if not os.path.exists(dense_ckpt_path):
        print("\n[2/5] Training Dense Qwen2.5-0.5B Baseline (Identical 600 Steps SFT on GSM8K)...")
        dense_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        dense_model.config.use_cache = False
        dense_model.train()
        optimizer = torch.optim.AdamW(dense_model.parameters(), lr=1e-4, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600, eta_min=1e-5)

        torch.manual_seed(42)
        step = 0
        total_steps = 600
        batch_size = 4
        grad_accum_steps = 4

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
                outputs = dense_model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / grad_accum_steps

            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(dense_model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                print(f"  [Dense Qwen Train] Step {step+1:>3}/600 | Loss: {loss.item()*grad_accum_steps:.4f}")
            step += 1

        print(f"Saving fine-tuned Dense Qwen checkpoint to {dense_ckpt_path}...")
        torch.save(dense_model.state_dict(), dense_ckpt_path)
        vol.commit()
        del dense_model
        torch.cuda.empty_cache()
    else:
        print(f"\n[2/5] Found existing trained Dense checkpoint at {dense_ckpt_path}.")

    # -------------------------------------------------------------------------
    # 3. Define SubQ Layer Architecture
    # -------------------------------------------------------------------------
    OFFSETS = [0, 1, 2, 4, 8, 16, 32, 64]
    K = len(OFFSETS)
    config = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    num_key_value_groups = num_heads // num_kv_heads

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, n_kv_heads, slen, h_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, h_dim)
        return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, h_dim)

    class SubQQwenAttention(nn.Module):
        def __init__(self, orig_attn, rotary_emb=None):
            super().__init__()
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
            self.offsets = OFFSETS

        def forward(self, hidden_states, position_ids=None, attention_mask=None, past_key_values=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
            B, L, _ = hidden_states.shape

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

            if position_ids is None:
                position_ids = torch.arange(0, L, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

            # RoPE
            if position_embeddings is not None:
                cos, sin = position_embeddings
            elif self.rotary_emb is not None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = None, None

            if cos is not None and sin is not None:
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # Causal SubQ Logarithmic Offset Routing
            scores_list, valid_masks = [], []
            for d in self.offsets:
                if d == 0:
                    s = (query_states * key_states).sum(dim=-1) * self.scale
                    m = torch.ones(B, self.num_heads, L, device=hidden_states.device, dtype=torch.bool)
                elif d < L:
                    q_slice = query_states[:, :, d:, :]
                    k_slice = key_states[:, :, :-d, :]
                    s_valid = (q_slice * k_slice).sum(dim=-1) * self.scale
                    s = F.pad(s_valid, (d, 0), value=-1e4)
                    m = torch.cat([
                        torch.zeros(B, self.num_heads, d, device=hidden_states.device, dtype=torch.bool),
                        torch.ones(B, self.num_heads, L - d, device=hidden_states.device, dtype=torch.bool)
                    ], dim=-1)
                else:
                    s = torch.full((B, self.num_heads, L), -1e4, device=hidden_states.device)
                    m = torch.zeros(B, self.num_heads, L, device=hidden_states.device, dtype=torch.bool)
                scores_list.append(s)
                valid_masks.append(m)

            scores = torch.stack(scores_list, dim=-1)
            valid_mask = torch.stack(valid_masks, dim=-1)
            scores = scores.masked_fill(~valid_mask, float("-inf"))
            attn_weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)

            out = torch.zeros_like(query_states)
            for k_idx, d in enumerate(self.offsets):
                w_k = attn_weights[:, :, :, k_idx : k_idx + 1]
                if d == 0:
                    out = out + w_k * value_states
                elif d < L:
                    v_shifted = F.pad(value_states[:, :, :-d, :], (0, 0, d, 0))
                    out = out + w_k * v_shifted

            out = out.transpose(1, 2).contiguous().view(B, L, -1)
            attn_output = self.o_proj(out)
            return attn_output, None

    # -------------------------------------------------------------------------
    # 4. Helper Functions for GSM8K Extraction & Accuracy
    # -------------------------------------------------------------------------
    def extract_answer_number(text: str) -> str:
        """Extract the exact numerical answer following '####' or from generated text."""
        # 1. Look for delimiter '#### <number>'
        if "####" in text:
            ans_part = text.split("####")[-1].strip()
            # extract first number in ans_part
            match = re.search(r"(-?[\d,]+(?:\.\d+)?)", ans_part)
            if match:
                return match.group(1).replace(",", "")
        
        # 2. Fallback: look for last number in text
        numbers = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
        if numbers:
            return numbers[-1].replace(",", "")
        return ""

    def evaluate_model_on_gsm8k(model, model_name: str, num_eval_samples: int = 150):
        print(f"\nEvaluating {model_name} on {num_eval_samples} GSM8K Test Problems (Greedy Decoding)...")
        model.eval()
        correct = 0
        total = 0
        has_delimiter = 0
        total_gen_tokens = 0
        t0 = time.time()

        for idx, item in enumerate(test_data[:num_eval_samples]):
            question = item["question"]
            gold_answer_text = item["answer"]
            gold_num = extract_answer_number(gold_answer_text)

            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            curr_ids = input_ids.clone()
            generated_tokens = []

            with torch.no_grad():
                for _ in range(160): # Max 160 tokens
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(curr_ids)
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

            if (idx + 1) % 25 == 0 or (idx + 1) == num_eval_samples:
                cur_acc = (correct / total) * 100.0
                print(f"  [{model_name}] Progress {idx+1:>3}/{num_eval_samples} | Exact Accuracy: {cur_acc:>5.2f}% ({correct}/{total}) | Delimiter Adherence: {(has_delimiter/total)*100:.1f}%")

        elapsed = time.time() - t0
        tok_per_sec = total_gen_tokens / max(1e-5, elapsed)
        acc = (correct / total) * 100.0
        fmt_rate = (has_delimiter / total) * 100.0
        
        return {
            "name": model_name,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "format_adherence": fmt_rate,
            "tok_per_sec": tok_per_sec,
            "time_sec": elapsed
        }

    # -------------------------------------------------------------------------
    # 5. Run Quantitative Shootout
    # -------------------------------------------------------------------------
    EVAL_SAMPLES = 150 # 150 test problems for thorough quantitative benchmarking

    # A. Evaluate Trained Dense Qwen2.5-0.5B
    print("\n" + "=" * 135)
    print("  LOADING AND EVALUATING TRAINED DENSE QWEN2.5-0.5B")
    print("=" * 135)
    dense_eval_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    dense_eval_model.load_state_dict(torch.load(dense_ckpt_path, map_location=device))
    dense_eval_model.config.use_cache = False
    
    dense_results = evaluate_model_on_gsm8k(dense_eval_model, "Dense Qwen2.5-0.5B", num_eval_samples=EVAL_SAMPLES)
    del dense_eval_model
    torch.cuda.empty_cache()

    # B. Evaluate Trained SubQ-Qwen2.5-0.5B
    print("\n" + "=" * 135)
    print("  LOADING AND EVALUATING TRAINED SUBQ-QWEN2.5-0.5B")
    print("=" * 135)
    subq_eval_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    rotary_emb = getattr(subq_eval_model.model, "rotary_emb", None)
    for layer in subq_eval_model.model.layers:
        orig_attn = layer.self_attn
        layer.self_attn = SubQQwenAttention(orig_attn, rotary_emb=rotary_emb)
    
    subq_eval_model.load_state_dict(torch.load(subq_ckpt_path, map_location=device))
    subq_eval_model.config.use_cache = False
    
    subq_results = evaluate_model_on_gsm8k(subq_eval_model, "SubQ-Qwen2.5-0.5B (K=8)", num_eval_samples=EVAL_SAMPLES)

    # -------------------------------------------------------------------------
    # 6. Master Scorecard
    # -------------------------------------------------------------------------
    print("\n" + "=" * 135)
    print("  MASTER GSM8K EXACT-MATCH REASONING BENCHMARK SCORECARD")
    print("=" * 135)
    print(f"{'Model Architecture':<30} | {'Test Accuracy (%)':<20} | {'Correct / Total':<18} | {'Delimiter Adherence':<22} | {'Throughput':<15}")
    print("-" * 135)
    print(f"{dense_results['name']:<30} | {dense_results['accuracy']:>6.2f}%{'':<13} | {dense_results['correct']:>3}/{dense_results['total']:<14} | {dense_results['format_adherence']:>6.1f}%{'':<15} | {dense_results['tok_per_sec']:>6.1f} tok/s")
    print(f"{subq_results['name']:<30} | {subq_results['accuracy']:>6.2f}%{'':<13} | {subq_results['correct']:>3}/{subq_results['total']:<14} | {subq_results['format_adherence']:>6.1f}%{'':<15} | {subq_results['tok_per_sec']:>6.1f} tok/s")
    print("=" * 135)

    retention = (subq_results['accuracy'] / max(1e-5, dense_results['accuracy'])) * 100.0
    print(f"\n⚡ SubQ-Qwen Reasoning Retention vs. Full Dense Qwen: {retention:.1f}%")

@app.local_entrypoint()
def main():
    evaluate_qwen_gsm8k.remote()
