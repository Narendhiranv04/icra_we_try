#!/bin/bash
set -e

# Build the learning index first using the correct pilot data
python scripts/build_learning_index.py

# Precompute frozen DINOv2 and SentenceTransformer features for fast loading
python scripts/precompute_features.py

# Precompute for context challenge
python scripts/precompute_features.py --index data/manifests/context_challenge_manifest.jsonl --out_dir data/features_context --force

seeds=(11 23 42 67 101)
configs=("query_only" "language_query" "demo_query" "full_no_ranking" "pooled_multimodal" "relational_heatmap")

for seed in "${seeds[@]}"; do
    for config in "${configs[@]}"; do
        echo "Running $config with seed $seed..."
        python scripts/train_model.py --config configs/learning/${config}.yaml --seed $seed
        python scripts/evaluate_model.py --dir learning_outputs/${config}_seed${seed}
        python scripts/evaluate_context_challenge.py --experiment-dir learning_outputs/${config}_seed${seed}
    done
done

# Aggregate metrics
python scripts/aggregate_metrics.py

echo "All multi-seed experiments complete!"
