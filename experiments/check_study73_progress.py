import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")
app = modal.App("check-study73-progress-detail", image=image)
volume = modal.Volume.from_name("subq-models-vol")

@app.function(volumes={"/models": volume})
def check_progress():
    import os, json, torch

    print("=" * 80)
    print("  MODAL CLOUD VOLUME: STUDY 73 REAL-TIME PROGRESS")
    print("=" * 80)

    for name, ckpt_file, res_file in [
        ("Variant 1 (Sequential 2-Layer SubQ)", "/models/study73_1_sequential_checkpoint.pt", "/models/study73_1_results.json"),
        ("Variant 2 (Interleaved 2-Layer SubQ)", "/models/study73_2_interleaved_checkpoint.pt", "/models/study73_2_results.json")
    ]:
        print(f"\n--- {name} ---")
        if os.path.exists(res_file):
            with open(res_file) as f:
                res = json.load(f)
            print(f"  [COMPLETED] Top-1: {res['top1']:.2f}% | Top-5: {res['top5']:.2f}% | Loss: {res['loss']:.4f} | Time: {res['time']:.1f}s")
        elif os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location="cpu")
            print(f"  [IN PROGRESS] Epoch: {ckpt['epoch']}/20 | Train Acc: {ckpt.get('train_acc', 0):.2f}% | Time Elapsed: {ckpt.get('total_time', 0):.1f}s")
        else:
            print("  [STARTING] Initializing...")
    print("=" * 80)

@app.local_entrypoint()
def main():
    check_progress.remote()
