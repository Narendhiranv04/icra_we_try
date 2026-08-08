import argparse
import yaml
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning.dataset import LearningDataset, PairBatchSampler
from src.learning.trainer import validate_epoch
from src.learning.losses import LearningLoss
from src.learning.utils import resolve_checkpoint
from scripts.train_model import get_model

def run_evaluation(model, dataset, criterion, device, batch_size=8, desc=""):
    sampler = PairBatchSampler(dataset, batch_size, seed=42)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
    metrics = validate_epoch(model, loader, criterion, device)
    print(f"[{desc}] Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f} | Latent PLA: {metrics.get('latent_pla', 0):.4f}")
    return metrics

def compare_metrics(baseline, perturbed):
    b_preds = np.array(baseline.get("_raw_preds", []))
    p_preds = np.array(perturbed.get("_raw_preds", []))

    b_scores = np.array(baseline.get("_raw_scores", []))
    p_scores = np.array(perturbed.get("_raw_scores", []))

    res = {
        "accuracy": perturbed["accuracy"],
        "f1": perturbed.get("f1", 0),
        "latent_pla": perturbed.get("latent_pla", 0),
    }

    b_logits = np.array(baseline.get("_raw_logits", []))
    p_logits = np.array(perturbed.get("_raw_logits", []))

    # Assert finite
    if len(b_logits) > 0:
        assert np.all(np.isfinite(b_logits)), "Baseline logits contain NaN or Inf"
    if len(p_logits) > 0:
        assert np.all(np.isfinite(p_logits)), "Perturbed logits contain NaN or Inf"
    assert np.all(np.isfinite(b_preds)), "Baseline probs contain NaN or Inf"
    assert np.all(np.isfinite(p_preds)), "Perturbed probs contain NaN or Inf"
    if len(b_scores) > 0:
        assert np.all(np.isfinite(b_scores)), "Baseline scores contain NaN or Inf"
    if len(p_scores) > 0:
        assert np.all(np.isfinite(p_scores)), "Perturbed scores contain NaN or Inf"

    b_class = (b_preds > 0.5).astype(int)
    p_class = (p_preds > 0.5).astype(int)
    flips = (b_class != p_class).mean()

    if len(b_logits) > 0 and len(b_logits) == len(p_logits):
        delta_logit = np.abs(p_logits - b_logits).mean()
        res["mean_delta_logit"] = float(delta_logit)

    delta_prob = np.abs(p_preds - b_preds).mean()
    res["flip_rate"] = float(flips)
    res["mean_delta_prob"] = float(delta_prob)
    res["mean_confidence"] = float(np.abs(p_preds - 0.5).mean() * 2)

    if len(b_scores) > 0 and len(b_scores) == len(p_scores):
        delta_compat = np.abs(p_scores - b_scores).mean()
        res["mean_compatibility"] = float(p_scores.mean())
        res["mean_delta_compatibility"] = float(delta_compat)

    b_lat = baseline.get("_raw_latents", [])
    p_lat = perturbed.get("_raw_latents", [])
    if len(b_lat) > 0 and len(b_lat) == len(p_lat):
        b_lat = np.array(b_lat)
        p_lat = np.array(p_lat)
        assert np.all(np.isfinite(b_lat)), "Baseline latents contain NaN or Inf"
        assert np.all(np.isfinite(p_lat)), "Perturbed latents contain NaN or Inf"

        # Calculate cosine similarity
        norm_b = np.linalg.norm(b_lat, axis=1)
        norm_p = np.linalg.norm(p_lat, axis=1)
        # Avoid division by zero
        norm_b[norm_b == 0] = 1e-8
        norm_p[norm_p == 0] = 1e-8

        cos_sim = np.sum(b_lat * p_lat, axis=1) / (norm_b * norm_p)
        res["mean_latent_cosine_sim"] = float(cos_sim.mean())
        res["mean_latent_cosine_change"] = float(1.0 - cos_sim.mean())

    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to learning_outputs/<experiment_name>")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    with open(out_dir / "resolved_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config).to(device)

    # Use resolve_checkpoint
    ckpt_path = resolve_checkpoint(out_dir)

    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    criterion = LearningLoss(
        margin=config.get("margin", 0.2),
        lambda_cls=config.get("lambda_cls", 1.0),
        lambda_rank=config.get("lambda_rank", 0.0),
        lambda_heat=config.get("lambda_heat", 0.0)
    )

    splits = ["id_train", "id_val", "unseen_object", "unseen_background", "compositional"]
    all_metrics = {}

    for split in splits:
        dataset = LearningDataset(
            index_path=config.get("index_path", "learning_data/index.jsonl"),
            features_dir=config.get("feature_cache_path", "learning_data/features"),
            split=split,
            return_masks=config.get("heatmap", False),
            seed=config.get("seed", 42),
            split_seed=config.get("split_seed", 42)
        )
        if len(dataset) > 0:
            desc = f"[TRAIN] Split {split}" if split == "id_train" else f"Split {split}"
            m = run_evaluation(model, dataset, criterion, device, config.get("batch_size", 8), desc=desc)
            # Remove raw arrays before saving
            m_clean = {k: v for k, v in m.items() if not k.startswith("_raw")}
            if split != "id_train":
                all_metrics[split] = m_clean

            if split == "id_val":
                baseline_id_val = m

    with open(out_dir / "metrics_by_split.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Saved evaluation metrics to {out_dir / 'metrics_by_split.json'}")

    # Ablation logic
    if "relational" in str(out_dir) or "pooled_multimodal" in str(out_dir) or "demo_query" in str(out_dir) or "language_query" in str(out_dir):
        print("Running Conditioning Diagnostics...")
        ablation_results = {}

        def run_ablation(ablation_mode, name):
            res_all = {}
            for split in splits:
                ds = LearningDataset(
                    index_path=config.get("index_path", "learning_data/index.jsonl"),
                    features_dir=config.get("feature_cache_path", "learning_data/features"),
                    split=split,
                    return_masks=config.get("heatmap", False),
                    seed=config.get("seed", 42),
                    split_seed=config.get("split_seed", 42),
                    ablation_mode=ablation_mode
                )
                if len(ds) == 0: continue
                m = run_evaluation(model, ds, criterion, device, batch_size=8, desc=f"{name} {split}")

                # We need baseline for this split
                baseline_split = all_metrics.get(split) if split != "id_val" else baseline_id_val
                if baseline_split:
                    # all_metrics might not have raw arrays saved. We should only do compare_metrics if raw arrays exist.
                    # Since we stripped raw arrays for save, let's just get them!
                    # Actually, we need to run baseline for all splits with raw arrays.
                    pass

            # Since compare_metrics needs raw arrays, let's just evaluate id_val for ablation comparison.
            # But the prompt said "RUN CONDITIONING DIAGNOSTICS ON ALL EVAL SPLITS... on id_val, unseen_object, unseen_background, compositional".
            # So we must collect the full metrics. We can store raw metrics in `baselines_raw = {}`.
            pass

        baselines_raw = {}
        for split in splits:
            ds = LearningDataset(
                index_path=config.get("index_path", "learning_data/index.jsonl"),
                features_dir=config.get("feature_cache_path", "learning_data/features"),
                split=split,
                return_masks=config.get("heatmap", False),
                seed=config.get("seed", 42),
                split_seed=config.get("split_seed", 42)
            )
            if len(ds) > 0:
                m_raw = run_evaluation(model, ds, criterion, device, batch_size=8, desc=f"Baseline {split} (raw)")
                baselines_raw[split] = m_raw
                # Save raw arrays for bootstrapping
                m_raw_save = {k: v for k, v in m_raw.items() if k.startswith("_raw") and isinstance(v, list)}

                # Assert finiteness for baselines
                for arr_key in ["_raw_logits", "_raw_preds", "_raw_scores", "_raw_latents"]:
                    if arr_key in m_raw_save and len(m_raw_save[arr_key]) > 0:
                        assert np.all(np.isfinite(m_raw_save[arr_key])), f"Baseline {arr_key} in {split} contains NaN or Inf"

                with open(out_dir / f"metrics_raw_{split}.json", "w") as f:
                    # Convert float32 to float for JSON
                    # Actually json.dump handles list of floats, but numpy float32 might need conversion if directly dumping numpy array
                    # The trainer returns lists, so json.dump should be fine unless it has numpy floats.
                    # We can use a custom encoder or convert to float list.
                    def to_float(obj):
                        if isinstance(obj, np.ndarray): return obj.tolist()
                        if isinstance(obj, np.generic): return float(obj)
                        return obj

                    # Convert lists of numpy arrays to lists of lists for JSON
                    for k in m_raw_save:
                        m_raw_save[k] = [to_float(x) for x in m_raw_save[k]]

                    json.dump(m_raw_save, f)

        def run_ablation_splits(ablation_mode, name):
            ablation_results[name] = {}
            for split in splits:
                if split not in baselines_raw: continue
                ds = LearningDataset(
                    index_path=config.get("index_path", "learning_data/index.jsonl"),
                    features_dir=config.get("feature_cache_path", "learning_data/features"),
                    split=split,
                    return_masks=config.get("heatmap", False),
                    seed=config.get("seed", 42),
                    split_seed=config.get("split_seed", 42),
                    ablation_mode=ablation_mode
                )
                m = run_evaluation(model, ds, criterion, device, batch_size=8, desc=f"{name} {split}")
                ablation_results[name][split] = compare_metrics(baselines_raw[split], m)

        run_ablation_splits("wrong_instruction", "Wrong Instruction")
        run_ablation_splits("heldout_paraphrase", "Held-out Paraphrase")
        run_ablation_splits("zero_text", "Zero Text")
        run_ablation_splits("wrong_demo", "Wrong Demo")
        run_ablation_splits("zero_demo", "Zero Demo")

        with open(out_dir / "conditioning_sensitivity.json", "w") as f:
            json.dump(ablation_results, f, indent=2)

if __name__ == "__main__":
    main()
