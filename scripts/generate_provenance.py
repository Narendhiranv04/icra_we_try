import json
import subprocess
import hashlib
from pathlib import Path
from collections import Counter

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def main():
    manifest_path = Path("data/manifests/pilot_manifest.jsonl")
    out_path = Path("artifacts/learning_stage1/dataset_provenance.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
        
    records = []
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    controls = [r for r in records if r.get("sample_type") == "positive_control"]
    pairs = [r for r in records if r.get("sample_type") != "positive_control"]
    
    splits = Counter()
    tasks = Counter()
    demos = set()
    
    for p in pairs:
        spec = p.get("stop", {}).get("spec", {})
        splits[spec.get("split", "unknown")] += 1
        tasks[spec.get("task_family", "unknown")] += 1
    
    provenance = {
        "tested_benchmark_commit": get_git_commit(),
        "pilot_config": "configs/pilot.yaml",
        "pilot_manifest_path": str(manifest_path),
        "pilot_manifest_sha256": sha256,
        "matched_pair_count": len(pairs),
        "query_count": len(pairs) * 2,
        "control_count": len(controls),
        "split_counts": dict(splits),
        "task_counts": dict(tasks),
        "query_root_directory": "data/pilot_queries"
    }
    
    with open(out_path, 'w') as f:
        json.dump(provenance, f, indent=2)
        
    print(f"Dataset provenance saved to {out_path}")

if __name__ == "__main__":
    main()
