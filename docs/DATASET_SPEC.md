# Dataset Specification: Relational Precondition MuJoCo Benchmark v0.1

## Overview
The benchmark dataset consists of paired demonstration-conditioned relational query scenes designed to evaluate whether a robot should **PROCEED** with an intended action or **STOP** because a relational precondition is violated.

## Dataset Profiles

### 1. Corrected Smoke Profile (`configs/smoke.yaml`)
- **Demonstration Videos**: 2 genuine Fetch robot videos (1 open-box, 1 pick-and-place)
- **Query Pairs**: 8 matched counterfactual pairs (4 for Task 1, 4 for Task 2)
- **Total Query Images**: 16 paired RGB query images
- **Splits**: `id`, `unseen_object`, `unseen_background`, `compositional`

### 2. Pilot Profile (`configs/pilot.yaml`)
- **Demonstration Videos**: 6 genuine Fetch robot videos (3 per task family)
- **Query Pairs**: 120 matched counterfactual pairs (60 per task family)
- **Total Query Images**: 240 paired RGB query images
- **Splits**: 30 pairs per split (`id`, `unseen_object`, `unseen_background`, `compositional`)

## Metadata Schema (`EpisodeSpec`)
Each query pair contains a `metadata.json` record formatted as:
```json
{
  "pair_id": "pair_task_1_001",
  "task_id": "task_1",
  "instruction": "Open the box.",
  "split": "id",
  "stop": {
    "label": "STOP",
    "is_occupied": true,
    "culprits": ["blocker1"],
    "rgb_path": "data/queries/pair_task_1_001/stop_rgb.png",
    "instance_segmentation_path": "data/queries/pair_task_1_001/stop_instance_segmentation.png",
    "candidate_object_mask_path": "data/queries/pair_task_1_001/stop_candidate_object_mask.png",
    "relation_target_mask_path": "data/queries/pair_task_1_001/stop_relation_target_mask.png",
    "causal_violation_mask_path": "data/queries/pair_task_1_001/stop_causal_violation_mask.png",
    "combined_visualization_path": "data/queries/pair_task_1_001/stop_combined_relation_visualization.png",
    "spec": { ... }
  },
  "proceed": {
    "label": "PROCEED",
    "is_occupied": false,
    "culprits": [],
    "rgb_path": "data/queries/pair_task_1_001/proceed_rgb.png",
    "instance_segmentation_path": "data/queries/pair_task_1_001/proceed_instance_segmentation.png",
    "candidate_object_mask_path": "data/queries/pair_task_1_001/proceed_candidate_object_mask.png",
    "relation_target_mask_path": "data/queries/pair_task_1_001/proceed_relation_target_mask.png",
    "causal_violation_mask_path": "data/queries/pair_task_1_001/proceed_causal_violation_mask.png",
    "combined_visualization_path": "data/queries/pair_task_1_001/proceed_combined_relation_visualization.png",
    "spec": { ... }
  }
}
```
