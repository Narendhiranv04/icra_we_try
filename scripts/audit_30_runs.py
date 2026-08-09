import json
import os
from pathlib import Path

models = [
    "query_only", "language_query", "demo_query", 
    "full_no_ranking", "pooled_multimodal", "relational_heatmap"
]
seeds = [11, 23, 42, 67, 101]

audit_results = {}
all_passed = True

def check_file(path):
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    if path.suffix == '.json':
        try:
            with open(path) as f:
                json.load(f)
        except:
            return False
    return True

for m in models:
    for s in seeds:
        dir_name = f"learning_outputs/{m}_seed{s}"
        p = Path(dir_name)
        
        expected_files = [
            "resolved_config.yaml",
            "best.ckpt",
            "last.ckpt",
            "metrics.json",
            "metrics_by_split.json",
            "context_challenge_results.json",
            "context_challenge_results_raw.json",
            "bootstrap_results.json",
            "train_pairs.json",
            "val_pairs.json"
        ]
        
        if m in ["language_query", "demo_query", "full_no_ranking", "pooled_multimodal", "relational_heatmap"]:
            expected_files.append("conditioning_sensitivity.json")
            
        run_status = {"exists": p.exists(), "files": {}}
        if not run_status["exists"]:
            all_passed = False
            audit_results[dir_name] = run_status
            continue
            
        for f in expected_files:
            file_path = p / f
            exists = check_file(file_path)
            run_status["files"][f] = exists
            if not exists:
                all_passed = False
                
        audit_results[dir_name] = run_status

out_path = Path("artifacts/learning_stage1/final_run_audit.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(audit_results, f, indent=2)

if all_passed:
    print("ALL 30 RUNS PASSED")
    raise SystemExit(0)
else:
    print("SOME RUNS FAILED AUDIT")
    raise SystemExit(1)
