#!/bin/bash
set -e

echo "Starting full diagnostic pilot..."

# query_only
conda run --no-capture-output -n infeasibility-latent python scripts/train_model.py \
    --config configs/context_forcing_v1/query_only.yaml --seed 11
conda run --no-capture-output -n infeasibility-latent python scripts/train_model.py \
    --config configs/context_forcing_v1/query_only.yaml --seed 42

# pooled_multimodal
conda run --no-capture-output -n infeasibility-latent python scripts/train_model.py \
    --config configs/context_forcing_v1/pooled_multimodal.yaml --seed 11
conda run --no-capture-output -n infeasibility-latent python scripts/train_model.py \
    --config configs/context_forcing_v1/pooled_multimodal.yaml --seed 42

# relational_heatmap
conda run --no-capture-output -n infeasibility-latent python scripts/train_model.py \
    --config configs/context_forcing_v1/relational_heatmap.yaml --seed 11
conda run --no-capture-output -n infeasibility-latent python scripts/train_model.py \
    --config configs/context_forcing_v1/relational_heatmap.yaml --seed 42

echo "All training completed!"
