#!/bin/bash
set -e

echo "Running query_only..."
conda run --no-capture-output -n infeasibility-latent python scripts/train_model.py \
    --config configs/context_forcing_v1_tiny/query_only.yaml --log_gradients

echo "Running pooled_multimodal..."
conda run --no-capture-output -n infeasibility-latent python scripts/train_model.py \
    --config configs/context_forcing_v1_tiny/pooled_multimodal.yaml --log_gradients

echo "Running relational_heatmap..."
conda run --no-capture-output -n infeasibility-latent python scripts/train_model.py \
    --config configs/context_forcing_v1_tiny/relational_heatmap.yaml --log_gradients

echo "Tiny Overfit Training completed!"
