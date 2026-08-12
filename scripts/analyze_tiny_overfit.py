import json
from pathlib import Path
import numpy as np

def analyze():
    base_dir = Path("learning_outputs/context_forcing_v1_tiny")
    
    models = ["query_only", "pooled_multimodal", "relational_heatmap"]
    
    # 1. Gradients
    combined_grads = {}
    for m in models:
        grad_path = base_dir / m / "gradient_norms_epoch1.json"
        if grad_path.exists():
            with open(grad_path, "r") as f:
                combined_grads[m] = json.load(f)
        else:
            combined_grads[m] = None
            print(f"Warning: Missing gradients for {m}")
            
    out_grads = Path("artifacts/context_forcing_v1/context_parameter_gradient_norms.json")
    out_grads.parent.mkdir(parents=True, exist_ok=True)
    with open(out_grads, "w") as f:
        json.dump(combined_grads, f, indent=2)
        
    # 2. Performance metrics
    results = {}
    
    for m in models:
        metrics_path = base_dir / m / "metrics.json"
        if not metrics_path.exists():
            results[m] = {"status": "missing_metrics"}
            continue
            
        with open(metrics_path, "r") as f:
            history = json.load(f)
            
        best_epoch = history.get("best_epoch", len(history["train"]))
        
        # We care about the last epoch of training data (overfit)
        final_train = history["train"][-1]
        final_val = history["val"][-1]
        
        results[m] = {
            "best_epoch": best_epoch,
            "final_train_loss": final_train.get("loss"),
            "final_train_acc": final_train.get("accuracy"),
            "final_val_loss": final_val.get("loss"),
            "final_val_acc": final_val.get("accuracy")
        }
        
    out_results = Path("artifacts/context_forcing_v1/tiny_overfit_results.json")
    with open(out_results, "w") as f:
        json.dump(results, f, indent=2)
        
    # 3. Write markdown summary
    out_md = Path("artifacts/context_forcing_v1/tiny_overfit_results.md")
    
    with open(out_md, "w") as f:
        f.write("# Tiny Overfit Diagnostic Results\n\n")
        f.write("## Overview\n")
        f.write("This diagnostic tested if models could overfit a tiny 8-pair dataset. ")
        f.write("A model capable of using context (demo/text) to differentiate identical RGB inputs should achieve 100% accuracy on the train set. ")
        f.write("A model that ignores context (e.g. `query_only`) will fail to overfit because identical images map to different labels.\n\n")
        
        f.write("## Accuracy (Final Epoch)\n")
        f.write("| Model | Train Acc | Val Acc | Notes |\n")
        f.write("|-------|-----------|---------|-------|\n")
        for m in models:
            res = results.get(m, {})
            tr_acc = res.get('final_train_acc', 0)
            val_acc = res.get('final_val_acc', 0)
            note = "Pass" if tr_acc > 0.95 else ("Fail (Expected for Query Only)" if m == "query_only" else "FAIL")
            f.write(f"| {m} | {tr_acc:.2f} | {val_acc:.2f} | {note} |\n")
            
        f.write("\n## Gradient Norms (Epoch 1)\n")
        f.write("Checking if context-specific parameters receive gradients during the first backward pass:\n\n")
        
        for m in models:
            f.write(f"### {m}\n")
            grads = combined_grads.get(m)
            if not grads:
                f.write("No gradient data found.\n\n")
                continue
            
            f.write("```json\n")
            f.write(json.dumps(grads, indent=2))
            f.write("\n```\n\n")
            
    print(f"Analysis saved to {out_md}")

if __name__ == "__main__":
    analyze()
