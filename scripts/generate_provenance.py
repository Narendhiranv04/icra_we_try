import json
import hashlib
import os
import torch
import sys
from pathlib import Path

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

out = {
    "TRAINED_CODE_COMMIT": "cfc436e315f72fca536794f81efc8d2ce60be8b2",
    "POSTPROCESS_CODE_COMMIT": "8a54aa84f1db14ebceca0e2e384f562b47608e66",
    "model_seeds": [11, 23, 42, 67, 101],
    "effective_split_seed": 42,
    "effective_sampler_seed": 42,
    "bootstrap_draws": 2000,
    "encoders": {
        "visual": "DINOv2 ViT-B/14 (model id: dinov2_vitb14)",
        "text": "SentenceTransformer all-MiniLM-L6-v2"
    },
    "environment": {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
}

bench_a = Path("learning_data/index.jsonl")
if bench_a.exists():
    out["benchmark_a_hash"] = hash_file(bench_a)
    with open(bench_a) as f:
        out["benchmark_a_counts"] = sum(1 for line in f if line.strip())
else:
    out["benchmark_a_hash"] = "Not found"

bench_b = Path("data/manifests/context_challenge_manifest.jsonl")
if bench_b.exists():
    out["benchmark_b_hash"] = hash_file(bench_b)
    with open(bench_b) as f:
        records = [json.loads(line) for line in f if line.strip()]
        out["benchmark_b_records"] = len(records)
        scenes = set(r["pair_id"] for r in records)
        out["benchmark_b_physical_scenes"] = len(scenes)

        state_counts_records = {"A": 0, "B": 0, "C": 0, "D": 0}
        state_counts_scenes = {"A": 0, "B": 0, "C": 0, "D": 0}
        seen_scenes = set()

        for r in records:
            state = r.get("state", "UNKNOWN")
            state_counts_records[state] += 1
            if r["pair_id"] not in seen_scenes:
                seen_scenes.add(r["pair_id"])
                state_counts_scenes[state] += 1

        out["benchmark_b_scene_counts_by_state"] = state_counts_scenes
        out["benchmark_b_task_record_counts_by_state"] = state_counts_records
else:
    out["benchmark_b_hash"] = "Not found"

Path("artifacts/learning_stage1").mkdir(parents=True, exist_ok=True)
with open("artifacts/learning_stage1/dataset_provenance.json", "w") as f:
    json.dump(out, f, indent=2)

print("Saved provenance")
