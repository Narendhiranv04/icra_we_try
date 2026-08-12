import json
import hashlib
from pathlib import Path

manifest_path = Path("data/manifests/context_challenge_manifest.jsonl")
freeze_out = Path("artifacts/context_forcing_v1/benchmark_b_freeze.json")
freeze_out.parent.mkdir(parents=True, exist_ok=True)

with open(manifest_path, "r") as f:
    records = [json.loads(line) for line in f if line.strip()]

rgb_hashes = set()
pairs = set()
scenes = {}

for r in records:
    pid = r["pair_id"]
    state = r["state"]
    pairs.add(pid)
    rgb_hashes.add(r["query_rgb_sha256"])
    if pid not in scenes:
        scenes[pid] = state

counts_by_state = {"A": 0, "B": 0, "C": 0, "D": 0}
for pid, state in scenes.items():
    counts_by_state[state] += 1

record_counts_by_state = {"A": 0, "B": 0, "C": 0, "D": 0}
for r in records:
    record_counts_by_state[r["state"]] += 1

with open(manifest_path, "rb") as f:
    manifest_hash = hashlib.sha256(f.read()).hexdigest()

freeze_data = {
    "role": "held_out_test_only",
    "frozen_release_commit": "00e05b040dc0c5fcadf348d3aa603d276ad2cc6e",
    "manifest_path": str(manifest_path),
    "manifest_sha256": manifest_hash,
    "physical_scenes": len(pairs),
    "task_records": len(records),
    "scenes_by_state": counts_by_state,
    "records_by_state": record_counts_by_state,
    "rgb_sha256_hashes": sorted(list(rgb_hashes))
}

with open(freeze_out, "w") as f:
    json.dump(freeze_data, f, indent=2)

print("Benchmark B freeze created at", freeze_out)
