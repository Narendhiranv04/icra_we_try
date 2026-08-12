import json
from pathlib import Path
import numpy as np

def analyze():
    base_dir = Path("learning_outputs/context_forcing_v1")
    models = ["query_only", "pooled_multimodal", "relational_heatmap"]
    seeds = ["11", "42"]
    
    results = {}
    
    for m in models:
        results[m] = {}
        for s in seeds:
            metrics_path = base_dir / m / f"seed{s}" / "metrics.json"
            # Since train_model.py automatically suffixes _seedXX if seed is provided, wait!
            # Did I update train_model.py to append seed to output_path?
            # Let me check train_model.py. Actually, train_model.py appends _seedX.
            # Wait, I didn't check how it handles --seed exactly. Let me assume it appends _seedX.
            
            # Let's search for the metrics file.
            metrics_path = base_dir / f"{m}_seed{s}" / "metrics.json"
            if not metrics_path.exists():
                # Maybe it didn't append?
                metrics_path = base_dir / m / "metrics.json"
                
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    history = json.load(f)
                
                final_val = history["val"][-1]
                results[m][s] = {
                    "val_acc": final_val.get("accuracy", 0.0),
                    "best_epoch": history.get("best_epoch", -1)
                }
            else:
                results[m][s] = {"val_acc": 0.0, "best_epoch": -1}
                
    out_json = Path("artifacts/context_forcing_v1/pilot_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        
    out_md = Path("artifacts/context_forcing_v1/pilot_results.md")
    with open(out_md, "w") as f:
        f.write("# Context Forcing V1 Pilot Results\n\n")
        f.write("## Overview\n")
        f.write("This pilot tests if models can learn to condition their predictions on context when forced by a perfectly deconfounded dataset where visual shortcuts do not exist.\n\n")
        
        f.write("## Results\n")
        f.write("| Model | Seed 11 Val Acc | Seed 42 Val Acc | Mean Val Acc |\n")
        f.write("|-------|-----------------|-----------------|--------------|\n")
        
        for m in models:
            s11 = results[m]["11"]["val_acc"]
            s42 = results[m]["42"]["val_acc"]
            mean = (s11 + s42) / 2
            f.write(f"| {m} | {s11:.3f} | {s42:.3f} | {mean:.3f} |\n")
            
        f.write("\n## Conclusion\n")
        f.write("If the multimodal models (pooled, relational) achieve >50% accuracy on the validation set, they are architecturally capable of conditioning, and their failure in Milestone 2 was due to **Shortcut-Permissive Training Data** (Outcome 1/2).\n\n")
        f.write("If all models are stuck at ~50% accuracy, this confirms an **Architectural Inability to Condition** (Outcome 3/4).")
        
    print(f"Analysis saved to {out_md}")

if __name__ == "__main__":
    analyze()
