#!/bin/bash
set -e

echo "Running Query Only..."
python scripts/train_model.py --config configs/learning/query_only.yaml
python scripts/evaluate_model.py --dir learning_outputs/query_only

echo "Running Language + Query..."
python scripts/train_model.py --config configs/learning/language_query.yaml
python scripts/evaluate_model.py --dir learning_outputs/language_query

echo "Running Demo + Query..."
python scripts/train_model.py --config configs/learning/demo_query.yaml
python scripts/evaluate_model.py --dir learning_outputs/demo_query

echo "Running Full (No Ranking)..."
python scripts/train_model.py --config configs/learning/full_no_ranking.yaml
python scripts/evaluate_model.py --dir learning_outputs/full_no_ranking

echo "Running Pooled Multimodal (with Ranking)..."
python scripts/train_model.py --config configs/learning/pooled_multimodal.yaml
python scripts/evaluate_model.py --dir learning_outputs/pooled_multimodal
python scripts/analyze_latents.py --dir learning_outputs/pooled_multimodal || true

echo "Running Relational Model..."
python scripts/train_model.py --config configs/learning/relational_heatmap.yaml
python scripts/evaluate_model.py --dir learning_outputs/relational_heatmap
python scripts/analyze_latents.py --dir learning_outputs/relational_heatmap || true

echo "All experiments complete."
