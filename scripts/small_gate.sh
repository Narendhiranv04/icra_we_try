#!/bin/bash
set -e

echo "Generating small Benchmark B..."
python3 -c "
from src.generation.context_challenge_generator import ContextChallengeGenerator
import json
from pathlib import Path
generator = ContextChallengeGenerator(output_dir='data/context_challenge_gate')
manifest_path = Path('data/manifests/context_challenge_manifest_gate.jsonl')
manifest_path.parent.mkdir(parents=True, exist_ok=True)
all_records = []
idx = 0
for state in ['A', 'B', 'C', 'D']:
    seed = 1000 + idx
    scene_id = f'scene_{idx:03d}_{state}'
    records = generator.generate_scene(scene_id, state, seed=seed)
    all_records.extend(records)
    idx += 1
with open(manifest_path, 'w') as f:
    for r in all_records:
        f.write(json.dumps(r) + '\n')
print('Small Benchmark B generated.')
"

echo "Precomputing context challenge features..."
python scripts/precompute_features.py --index data/manifests/context_challenge_manifest_gate.jsonl --out_dir data/features_context --force

echo "Running Query-Only gate..."
python scripts/train_model.py --config configs/learning/query_only_gate.yaml --seed 42
python scripts/evaluate_model.py --dir learning_outputs/query_only_gate_seed42
python scripts/evaluate_context_challenge.py --experiment-dir learning_outputs/query_only_gate_seed42 --index data/manifests/context_challenge_manifest_gate.jsonl
python scripts/bootstrap_metrics.py --experiment-dir learning_outputs/query_only_gate_seed42

echo "Running Relational v1 gate..."
python scripts/train_model.py --config configs/learning/relational_heatmap_gate.yaml --seed 42
python scripts/evaluate_model.py --dir learning_outputs/relational_heatmap_gate_seed42
python scripts/evaluate_context_challenge.py --experiment-dir learning_outputs/relational_heatmap_gate_seed42 --index data/manifests/context_challenge_manifest_gate.jsonl
python scripts/bootstrap_metrics.py --experiment-dir learning_outputs/relational_heatmap_gate_seed42

echo "Validating Gate Assertions..."
python scripts/validate_gate.py

echo "Gate passed!"
