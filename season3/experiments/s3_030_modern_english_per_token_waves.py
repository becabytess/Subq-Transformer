"""S3-030: State-Conditioned Per-Token Wave Router on Modern English (Sherlock Holmes)
with Deep Token-by-Token Search Radar Inspection Across Thought Hops (T=4).

Scientific Objective:
1. Train the State-Conditioned Per-Token Wave Router (T=4 hops, K=8 peaks, d=128, L=256)
   on clean modern English prose (The Adventures of Sherlock Holmes, Arthur Conan Doyle).
2. Measure validation loss and perplexity without any confirmation bias.
3. Conduct a token-by-token behavioral inspection on clean probe sentences:
   - For specific syntactic/semantic tokens (pronouns, verbs, quotes, dialogue), inspect:
     * Emitted wave parameters: amplitudes, dominant frequencies, spatial damping.
     * Top-K attended past offsets.
     * Exact attended characters and word tokens in prefix.
     * Attention softmax weights.
     * Antenna shift across thought iterations t = 1 -> 2 -> 3 -> 4 (hypothesis updating).
"""

import json
import math
import os
import time
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0", "numpy", "matplotlib"
)
app = modal.App("season3-s3-030-modern-english-radar")
volume = modal.Volume.from_name("subq-models-vol", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/models": volume})
def run_modern_english_experiment(
    corpus_text: str,
    seed: int = 42,
    total_steps: int = 2000,
    eval_interval: int = 250,
    val_batches: int = 30,
    seq_len: int = 256,
    batch_size: int = 32,
    d_model: int = 128,
    n_hops: int = 4,
    k_peaks: int = 8,
    num_waves: int = 12,
    mlp_ratio: int = 2,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Prepare Corpus
    chars = sorted(list(set(corpus_text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in corpus_text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    print("=" * 105)
    print("  S3-030: STATE-CONDITIONED PER-TOKEN WAVE ROUTER (MODERN ENGLISH - SHERLOCK HOLMES)")
    print(f"  Corpus: {len(data):,} chars | Vocab: {vocab_size} | Hops T = {n_hops} | K = {k_peaks} Peaks | SeqLen = {seq_len}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 105)

    def get_batch(split="train", gen=None):
        src = train_data if split == "train" else val_data
        max_idx = len(src) - seq_len - 1
        if gen is not None:
            ix = torch.randint(0, max_idx, (batch_size,), generator=gen)
        else:
            ix = torch.randint(0, max_idx, (batch_size,))
        x = torch.stack([src[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([src[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

    # 2. State-Conditioned Per-Token Wave Architecture
    class StateConditionedPerTokenWaveLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_hops = n_hops
            self.k_peaks = k_peaks
            self.num_waves = num_waves
            self.scale = 1.0 / math.sqrt(n_hops)
            self.max_d = seq_len

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)

            # Phase 1: Causal cuDNN GRU Scanner + FEN Query Extractor
            self.scanner = nn.GRU(d_model, d_model, batch_first=True)
            self.gate = nn.Linear(d_model, d_model)
            self.q_proj = nn.Linear(d_model, d_model)

            # Phase 2: Discrete Key and Value Projections
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

            # State-Conditioned Per-Token Wave Head
            self.ln_wave = nn.LayerNorm(d_model)
            self.wave_head = nn.Linear(d_model, num_waves * 4)

            log_freqs = torch.linspace(0.0, -2.5, num_waves)
            self.register_buffer("base_freqs", (10.0 ** log_freqs).view(1, 1, num_waves, 1))
            self.register_buffer("d_grid", torch.arange(1, self.max_d).float().view(1, 1, 1, self.max_d - 1))

            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)

            mlp_dim = d_model * mlp_ratio
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, d_model),
            )

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, return_dynamics: bool = False, return_radar: bool = False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            # Phase 1: Causal Scan -> Query Escrow Vault
            H, _ = self.scanner(x)                     # [B, L, D]
            G = torch.sigmoid(self.gate(H))            # [B, L, D]
            Q_raw = self.q_proj(G * H)                 # [B, L, D]
            E_q_all = torch.cumsum(Q_raw, dim=1)       # [B, L, D]

            # Phase 2: Multi-Hop Retrieval over Discrete Tokens with State-Conditioned Waves
            state = H
            pos_grid = torch.arange(L, device=idx.device).view(1, L, 1)

            diagnostics = {
                "state_velocities": [],
                "state_cosine_alignments": [],
                "wave_decays": [],
                "wave_reaches": [],
                "mean_attended_distances": [],
                "hop_entropies": [],
                "offset_std_per_token": [],
            }
            radar_log = []
            prev_state = state

            for hop in range(self.n_hops):
                # 1. State emits per-token wave parameters: [B, L, num_waves, 4]
                wave_params = self.wave_head(self.ln_wave(state)).view(B, L, self.num_waves, 4)
                amp = torch.tanh(wave_params[..., 0]).unsqueeze(-1)                          # [B, L, num_waves, 1]
                omega = (F.softplus(wave_params[..., 1]).unsqueeze(-1) * self.base_freqs)    # [B, L, num_waves, 1]
                phi = (wave_params[..., 2] * math.pi).unsqueeze(-1)                          # [B, L, num_waves, 1]
                decay = (F.softplus(wave_params[..., 3]) * 0.05).unsqueeze(-1)               # [B, L, num_waves, 1]

                # 2. Continuous personal 1D wave evaluated across distance grid: [B, L, max_d - 1]
                wave_comps = amp * torch.cos(omega * self.d_grid + phi) * torch.exp(-decay * self.d_grid)
                wave_1d = wave_comps.sum(dim=2)  # [B, L, max_d - 1]

                # 3. Personal top-K peaks per token
                topk_vals, past_offsets = torch.topk(wave_1d, k=self.k_peaks - 1, dim=-1)
                past_offsets = past_offsets + 1
                zero_off = torch.zeros((B, L, 1), dtype=torch.long, device=idx.device)
                zero_val = torch.zeros((B, L, 1), dtype=torch.float, device=idx.device)

                active_offsets = torch.cat([zero_off, past_offsets], dim=-1)  # [B, L, K]
                peak_vals = torch.cat([zero_val, topk_vals], dim=-1)          # [B, L, K]

                # 4. Vectorized Gathering of Keys and Values per token
                targets = pos_grid - active_offsets  # [B, L, K]
                valid_mask = targets >= 0
                targets_clamped = torch.clamp(targets, min=0)

                K_discrete = self.k_proj(state)  # [B, L, D]
                V_discrete = self.v_proj(state)  # [B, L, D]

                batch_idx = torch.arange(B, device=idx.device).view(B, 1, 1).expand(B, L, self.k_peaks)
                K_cand = K_discrete[batch_idx, targets_clamped]  # [B, L, K, D]
                V_cand = V_discrete[batch_idx, targets_clamped]  # [B, L, K, D]

                # 5. Query matching with per-token wave logit bias
                q = self.ln_q(E_q_all + state)  # [B, L, D]
                k = self.ln_k(K_cand)           # [B, L, K, D]
                v = V_cand                      # [B, L, K, D]

                scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d_model) + peak_vals
                scores = scores.masked_fill(~valid_mask, -1e4)
                weights = F.softmax(scores, dim=-1) * valid_mask.float()
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

                context = (weights.unsqueeze(-1) * v).sum(dim=2)  # [B, L, D]

                next_state = state + self.scale * context
                next_state = next_state + self.scale * self.mlp(self.ln_mlp(next_state))

                if return_radar:
                    radar_log.append({
                        "hop": hop + 1,
                        "active_offsets": active_offsets[0].cpu(),
                        "weights": weights[0].cpu(),
                        "peak_vals": peak_vals[0].cpu(),
                        "mean_decay": decay[0].mean(dim=-1).mean(dim=-1).cpu(),
                        "wave_amplitudes": amp[0].abs().mean(dim=-1).mean(dim=-1).cpu(),
                    })

                if return_dynamics:
                    diff = torch.norm(next_state - prev_state, dim=-1)
                    denom = torch.norm(next_state, dim=-1) + 1e-6
                    v_t = (diff / denom).mean().item()
                    diagnostics["state_velocities"].append(v_t)

                    cos_s = F.cosine_similarity(next_state.flatten(0, 1), prev_state.flatten(0, 1), dim=-1).mean().item()
                    diagnostics["state_cosine_alignments"].append(cos_s)

                    mean_decay = decay.mean().item()
                    diagnostics["wave_decays"].append(mean_decay)
                    diagnostics["wave_reaches"].append(1.0 / max(mean_decay, 1e-5))

                    mean_dist = (weights * active_offsets.float()).sum(dim=-1).mean().item()
                    diagnostics["mean_attended_distances"].append(mean_dist)

                    ent = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()
                    diagnostics["hop_entropies"].append(ent)

                    std_offsets = active_offsets[:, :, 1:].float().std(dim=1).mean().item()
                    diagnostics["offset_std_per_token"].append(std_offsets)

                    prev_state = next_state

                state = next_state

            logits = self.head(self.ln_f(state))
            if return_radar:
                return logits, radar_log
            if return_dynamics:
                return logits, diagnostics
            return logits

    # 3. Train Model
    train_gen = torch.Generator().manual_seed(seed)
    val_gen = torch.Generator().manual_seed(seed + 1000)

    model = StateConditionedPerTokenWaveLM().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def get_lr(step):
        if step < 100:
            return 1e-3 * (step + 1) / 100
        progress = (step - 100) / max(1, total_steps - 100)
        return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + math.cos(math.pi * progress))

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch("train", train_gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == total_steps:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(val_batches):
                    vx, vy = get_batch("val", val_gen)
                    v_logits = model(vx)
                    val_loss += F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1)).item()
            val_loss /= val_batches
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - t0
            print(f"  Step {step:4d}/{total_steps} | Train: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Elapsed: {elapsed:.1f}s", flush=True)

    total_training_time = time.time() - t0

    # 4. Extract Validation Dynamics
    print("\nExtracting dynamical trajectory across validation batches...")
    model.eval()
    val_gen_eval = torch.Generator().manual_seed(seed + 9999)

    accumulated = {
        "velocities": [0.0] * n_hops,
        "state_cosines": [0.0] * n_hops,
        "decays": [0.0] * n_hops,
        "reaches": [0.0] * n_hops,
        "mean_distances": [0.0] * n_hops,
        "entropies": [0.0] * n_hops,
        "offset_std_per_token": [0.0] * n_hops,
    }

    with torch.no_grad():
        for _ in range(val_batches):
            vx, vy = get_batch("val", val_gen_eval)
            _, d = model(vx, return_dynamics=True)
            for h in range(n_hops):
                accumulated["velocities"][h] += d["state_velocities"][h]
                accumulated["state_cosines"][h] += d["state_cosine_alignments"][h]
                accumulated["decays"][h] += d["wave_decays"][h]
                accumulated["reaches"][h] += d["wave_reaches"][h]
                accumulated["mean_distances"][h] += d["mean_attended_distances"][h]
                accumulated["entropies"][h] += d["hop_entropies"][h]
                accumulated["offset_std_per_token"][h] += d["offset_std_per_token"][h]

    for k in accumulated:
        accumulated[k] = [v / val_batches for v in accumulated[k]]

    print("=" * 105)
    print(f"  S3-030: STATE-CONDITIONED PER-TOKEN WAVE SCORECARD (T={n_hops})")
    print("=" * 105)
    print(f"{'Hop':<5} | {'Velocity v(t)':<15} | {'State Cosine':<14} | {'Wave Reach R':<14} | {'Attn Entropy H':<16} | {'Mean Dist d':<12} | {'Offset Std (Diversity)'}")
    print("-" * 105)
    for h in range(n_hops):
        print(f"{h+1:<5} | {accumulated['velocities'][h]:<15.4f} | {accumulated['state_cosines'][h]:<14.4f} | {accumulated['reaches'][h]:<14.2f} | {accumulated['entropies'][h]:<16.4f} | {accumulated['mean_distances'][h]:<12.2f} | {accumulated['offset_std_per_token'][h]:.2f}")
    print("-" * 105)
    print(f"Final Val Loss: {val_loss:.4f} | Final Val PPL: {val_ppl:.2f} | Training Time: {total_training_time:.1f}s")
    print("=" * 105)

    # 5. Behavioral Search Radar Inspection on Probe Sentences
    print("\n" + "=" * 105)
    print("  DEEP BEHAVIORAL SEARCH RADAR INSPECTION: TOKEN-BY-TOKEN WAVE DYNAMICS ACROSS HOPS")
    print("=" * 105)

    probe_sentences = [
        {
            "name": "Probe 1: Pronoun / Entity Resolution",
            "text": "Sherlock Holmes sat in his armchair, smoking his pipe. The visitor knocked on the door and asked for help.",
            "probe_words": ["his", "his", "visitor", "asked"]
        },
        {
            "name": "Probe 2: Quotation & Dialogue Discourse",
            "text": "He said: \u201cIt is an extraordinary problem, my dear Watson, but we shall solve it.\u201d",
            "probe_words": ["\u201c", "Watson", "solve", "\u201d"]
        },
        {
            "name": "Probe 3: Subordinate Clause & Causality",
            "text": "The doctor hurried away because the patient had suddenly collapsed in the garden.",
            "probe_words": ["because", "patient", "collapsed"]
        }
    ]

    def fmt_char(c):
        if c == " ":
            return "␣"
        if c == "\n":
            return "\\n"
        return c

    inspection_results = []

    for p_idx, probe in enumerate(probe_sentences):
        p_text = probe["text"]
        clean_p_text = "".join([c if c in char_to_ix else " " for c in p_text])
        p_indices = [char_to_ix[c] for c in clean_p_text]
        if len(p_indices) < seq_len:
            p_padded = p_indices + [char_to_ix.get(" ", 0)] * (seq_len - len(p_indices))
        else:
            p_padded = p_indices[:seq_len]

        p_tensor = torch.tensor([p_padded], dtype=torch.long, device=device)

        with torch.no_grad():
            _, radar = model(p_tensor, return_radar=True)

        print(f"\n--- {probe['name']} ---")
        print(f"Full Text: \"{p_text}\" (Length: {len(p_text)} chars)")

        found_probes = []
        search_start = 0
        for word in probe["probe_words"]:
            pos = p_text.find(word, search_start)
            if pos != -1:
                end_pos = pos + len(word) - 1
                found_probes.append((word, pos, end_pos))
                search_start = pos + 1
            else:
                pos_any = p_text.find(word)
                if pos_any != -1:
                    found_probes.append((word, pos_any, pos_any + len(word) - 1))

        probe_entry = {
            "name": probe["name"],
            "text": p_text,
            "tokens": []
        }

        for word, start_pos, target_pos in found_probes:
            token_char = p_text[target_pos]
            context_snippet = p_text[max(0, target_pos - 15):target_pos + 1]
            print(f"\n  >> Probe Token: '{word}' (Position {target_pos}, Char: '{fmt_char(token_char)}', Context: '...{context_snippet}')")

            token_hop_history = []
            prev_offsets = None

            for h in range(n_hops):
                hop_data = radar[h]
                offsets = hop_data["active_offsets"][target_pos].tolist()
                weights = hop_data["weights"][target_pos].tolist()
                peak_vals = hop_data["peak_vals"][target_pos].tolist()
                decay = hop_data["mean_decay"][target_pos].item()
                amp = hop_data["wave_amplitudes"][target_pos].item()
                reach = 1.0 / max(decay, 1e-5)

                attended_targets = []
                for off, w, pv in zip(offsets, weights, peak_vals):
                    target_idx = target_pos - off
                    if target_idx >= 0:
                        char_at = p_text[target_idx]
                        surrounding = p_text[max(0, target_idx - 3):min(len(p_text), target_idx + 4)]
                        attended_targets.append({
                            "offset": off,
                            "weight": round(w, 4),
                            "peak_val": round(pv, 3),
                            "target_char": char_at,
                            "snippet": surrounding
                        })
                    else:
                        attended_targets.append({
                            "offset": off,
                            "weight": 0.0,
                            "peak_val": 0.0,
                            "target_char": "[MASKED]",
                            "snippet": "[MASKED]"
                        })

                sorted_targets = sorted(attended_targets, key=lambda x: x["weight"], reverse=True)

                if prev_offsets is not None:
                    shift = sum(abs(o1 - o2) for o1, o2 in zip(offsets[1:], prev_offsets[1:])) / (k_peaks - 1)
                else:
                    shift = 0.0
                prev_offsets = offsets

                top3_str = ", ".join([
                    f"Δ={t['offset']} ('{fmt_char(t['target_char'])}' in '{t['snippet']}', w={t['weight']:.2f})"
                    for t in sorted_targets[:3]
                ])

                print(f"     Hop {h+1}: Reach R={reach:4.1f} | Shift={shift:4.1f} | Top Attended: {top3_str}")

                token_hop_history.append({
                    "hop": h + 1,
                    "reach": round(reach, 2),
                    "antenna_shift": round(shift, 2),
                    "mean_amplitude": round(amp, 4),
                    "targets": sorted_targets
                })

            probe_entry["tokens"].append({
                "word": word,
                "start_pos": start_pos,
                "target_pos": target_pos,
                "hop_history": token_hop_history
            })

        inspection_results.append(probe_entry)

    print("=" * 105)
    print("  INSPECTION COMPLETE")
    print("=" * 105)

    return {
        "final_val_loss": val_loss,
        "final_val_ppl": val_ppl,
        "n_hops": n_hops,
        "hops": list(range(1, n_hops + 1)),
        "velocities": accumulated["velocities"],
        "state_cosines": accumulated["state_cosines"],
        "decays": accumulated["decays"],
        "reaches": accumulated["reaches"],
        "mean_distances": accumulated["mean_distances"],
        "entropies": accumulated["entropies"],
        "offset_std_per_token": accumulated["offset_std_per_token"],
        "training_time": total_training_time,
        "parameters": param_count,
        "inspection": inspection_results,
    }


@app.local_entrypoint()
def main():
    print("Launching S3-030 Modern English Per-Token Wave Experiment on Modal A10G...")
    data_path = "season3/data/sherlock.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing corpus file at {data_path}. Run data preparation first.")

    with open(data_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    print(f"Loaded clean corpus ({len(corpus):,} characters). Submitting to Modal...")
    result = run_modern_english_experiment.remote(corpus_text=corpus)

    os.makedirs("season3/results", exist_ok=True)
    out_path = "season3/results/s3_030_modern_english_per_token_waves.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVED] Experiment scorecard and deep inspection saved to {out_path}")
