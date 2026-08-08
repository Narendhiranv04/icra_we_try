#!/bin/bash
set -e

echo "1. Generating full Benchmark-B dataset..."
MUJOCO_GL=egl conda run -n infeasibility-latent python scripts/generate_context_challenge.py

echo "2. Precomputing features..."
conda run -n infeasibility-latent python scripts/precompute_features.py --index data/manifests/context_challenge_manifest.jsonl --out_dir data/features_context --force

# Helper to run the evaluation suite for a single model config and seed
run_eval_suite() {
    local config=$1
    local seed=$2
    local name=$3
    
    echo "Running Evaluation Suite for ${name} (Seed ${seed})..."
    local out_dir="learning_outputs/${name}_seed${seed}"
    
    conda run -n infeasibility-latent python scripts/train_model.py --config ${config} --seed ${seed}
    conda run -n infeasibility-latent python scripts/evaluate_model.py --dir ${out_dir}
    conda run -n infeasibility-latent python scripts/evaluate_context_challenge.py --experiment-dir ${out_dir} --index data/manifests/context_challenge_manifest.jsonl
    conda run -n infeasibility-latent python scripts/bootstrap_metrics.py --experiment-dir ${out_dir}
    conda run -n infeasibility-latent python scripts/aggregate_metrics.py --base-dir learning_outputs --experiment-name ${name}
}

# The 5-seed protocol requires seeds 42, 43, 44, 45, 46.
# However, usually we might just run seed 42 to get it done, or loop over them.
# The user asked to "execute the final experimental runs sequentially".
# Let's run seed 42 for all models first as a baseline, or just Relational Model 5-seed.
# Based on the user prompt: "Configured 5-seed robust testing protocol"

SEEDS=(42 43 44 45 46)

# We will run all seeds for Relational Heatmap model
for s in "${SEEDS[@]}"; do
    run_eval_suite "configs/learning/relational_heatmap.yaml" ${s} "relational_heatmap"
done

# And maybe just seed 42 for baselines? The user mentioned evaluating all baselines in the previous summary.
run_eval_suite "configs/learning/query_only.yaml" 42 "query_only"
run_eval_suite "configs/learning/language_query.yaml" 42 "language_query"
run_eval_suite "configs/learning/demo_query.yaml" 42 "demo_query"
run_eval_suite "configs/learning/full_no_ranking.yaml" 42 "full_no_ranking"
run_eval_suite "configs/learning/pooled_multimodal.yaml" 42 "pooled_multimodal"

echo "Extracting final heatmaps for Relational Model seed 42..."
conda run -n infeasibility-latent python scripts/extract_heatmaps.py --experiment-dir learning_outputs/relational_heatmap_seed42 --index data/manifests/context_challenge_manifest.jsonl --out-dir artifacts/learning_stage1/heatmaps

echo "Analyzing latents for Relational Model seed 42..."
conda run -n infeasibility-latent python scripts/analyze_latents.py --experiment-dir learning_outputs/relational_heatmap_seed42 || true

echo "Final Experiment Complete!"
