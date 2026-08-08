#!/bin/bash
set -euo pipefail

export PYTHONPATH=.

echo "Phase 1: Generating full Benchmark-B dataset..."
MUJOCO_GL=egl conda run -n infeasibility-latent python scripts/generate_context_challenge.py

echo "Phase 2: Validating Benchmark-B dataset..."
conda run -n infeasibility-latent python scripts/validate_benchmark_b_dataset.py

echo "Phase 3: Precomputing features for Benchmark A (Training Data)..."
conda run -n infeasibility-latent python scripts/precompute_features.py --index learning_data/index.jsonl --out_dir learning_data/features --force

echo "Phase 4: Precomputing features for Benchmark B (Context Challenge)..."
conda run -n infeasibility-latent python scripts/precompute_features.py --index data/manifests/context_challenge_manifest.jsonl --out_dir data/features_context --force

# The 6 models
MODELS=(
    "configs/learning/query_only.yaml"
    "configs/learning/language_query.yaml"
    "configs/learning/demo_query.yaml"
    "configs/learning/full_no_ranking.yaml"
    "configs/learning/pooled_multimodal.yaml"
    "configs/learning/relational_heatmap.yaml"
)

NAMES=(
    "query_only"
    "language_query"
    "demo_query"
    "full_no_ranking"
    "pooled_multimodal"
    "relational_heatmap"
)

# The exactly specified 5 seeds
SEEDS=(11 23 42 67 101)

echo "Phases 5-10: Training and Evaluating 6 Models over 5 Seeds..."

for i in "${!MODELS[@]}"; do
    config="${MODELS[$i]}"
    name="${NAMES[$i]}"
    
    for seed in "${SEEDS[@]}"; do
        echo "Running ${name} (Seed ${seed})..."
        out_dir="learning_outputs/${name}_seed${seed}"
        
        conda run -n infeasibility-latent python scripts/train_model.py --config "${config}" --seed "${seed}"
        conda run -n infeasibility-latent python scripts/evaluate_model.py --dir "${out_dir}"
        conda run -n infeasibility-latent python scripts/evaluate_context_challenge.py --experiment-dir "${out_dir}" --index data/manifests/context_challenge_manifest.jsonl
        conda run -n infeasibility-latent python scripts/bootstrap_metrics.py --experiment-dir "${out_dir}"
    done
done

echo "Phase 11: Aggregating Metrics across all models..."
conda run -n infeasibility-latent python scripts/aggregate_metrics.py

echo "Phase 12: Generating final visualizations (Heatmaps & Latents) for Relational Model (Seed 42)..."
conda run -n infeasibility-latent python scripts/extract_heatmaps.py --experiment-dir learning_outputs/relational_heatmap_seed42 --index data/manifests/context_challenge_manifest.jsonl --out_dir artifacts/learning_stage1/heatmaps
conda run -n infeasibility-latent python scripts/analyze_latents.py --experiment-dir learning_outputs/relational_heatmap_seed42

echo "Final Experiment Complete!"
