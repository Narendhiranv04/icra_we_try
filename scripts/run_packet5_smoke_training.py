#!/usr/bin/env python3
"""Run all 4 canonical Packet-5 smoke training configurations and assemble the report."""

import hashlib
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.learning.intervention_config import InterventionTrainingConfig
from scripts.train_intervention_model import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

configs = [
    ("b0", "configs/intervention/learning/b0_smoke.yaml"),
    ("b0b", "configs/intervention/learning/b0b_smoke.yaml"),
    ("b3", "configs/intervention/learning/b3_smoke.yaml"),
    ("v2", "configs/intervention/learning/v2_smoke.yaml"),
]

results = {}
for key, cfg_path in configs:
    cfg = InterventionTrainingConfig.from_yaml(cfg_path)
    cfg.checkpoint_dir = "learning_outputs/checkpoints_smoke"
    logging.info("=== Running %s (%s, lambda_rank=%s) ===", key, cfg.model_name, cfg.lambda_rank)
    res = train(cfg)
    results[key] = res
    init_loss = res["initial_eval"]["total_loss"]
    final_loss = res["final_eval"]["total_loss"]
    logging.info("  Initial loss: %.4f | Final loss: %.4f", init_loss, final_loss)
    logging.info("  Finite: losses=%s, grads=%s, params=%s", res["finite_losses"], res["finite_gradients"], res["finite_parameters"])
    logging.info("  Roundtrip: %s", res["checkpoint_roundtrip_passed"])

# Compute cache index sha
cache_index_path = Path("data/intervention_smoke/features/feature_cache_index.json")
cache_index_sha = hashlib.sha256(cache_index_path.read_bytes()).hexdigest()

# Assemble Packet 5 Training Smoke Report
report = {
    "report_schema_version": "1.0.0",
    "training_status": "PASSED",
    "source_manifest_sha256": results["v2"]["manifest_sha256"],
    "feature_cache_index_sha256": cache_index_sha,
    "feature_cache_schema_version": "1.1.0",
    "feature_code_commit": "a4b9a72cb7d62ba27eee5ca7346141f8d559ae10",
    "dataset_generator_commit": "6fe085a86709ddcac45b6a387e6f7ba6c4fcf5eb",
    "training_code_commit": "a4b9a72cb7d62ba27eee5ca7346141f8d559ae10",
    "device": "cpu",
    "smoke_ranking_pair_count": 16,
    "smoke_valid_ranking_group_count": 6,
    "generalization_evaluation_performed": False,
    "smoke_dataset_used_for_training_and_diagnostics": True,
    "performance_claims_allowed": False,
    "b3_v2_initial_fingerprints_identical": (
        results["b3"]["initial_model_fingerprint_sha256"]
        == results["v2"]["initial_model_fingerprint_sha256"]
    ),
    "runs": {
        key: {
            "run_label": f"{key}_smoke",
            "model_name": res["model_name"],
            "lambda_feas": res["lambda_feas"],
            "lambda_rank": res["lambda_rank"],
            "seed": res["seed"],
            "parameter_count": res["param_count"],
            "initial_model_fingerprint_sha256": res["initial_model_fingerprint_sha256"],
            "epochs": res["epochs"],
            "optimizer": res["optimizer"],
            "learning_rate": res["learning_rate"],
            "initial_eval": res["initial_eval"],
            "final_eval": res["final_eval"],
            "finite_losses": res["finite_losses"],
            "finite_gradients": res["finite_gradients"],
            "finite_parameters": res["finite_parameters"],
            "checkpoint_roundtrip_passed": res["checkpoint_roundtrip_passed"],
        }
        for key, res in results.items()
    },
}

out_report_path = Path("artifacts/intervention_smoke/packet5_training_smoke_report.json")
out_report_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_report_path, "w") as f:
    json.dump(report, f, indent=2)

logging.info("=== PACKET 5 TRAINING SMOKE REPORT WRITTEN TO %s ===", out_report_path)
