# Gemini 3.1 Pro Implementation Phases

> Self-contained implementation prompts for sequential execution by a coding agent.
> Each phase has explicit prerequisites, objectives, files, tests, and stopping conditions.
> Based on the master plan at `docs/CAUSAL_INTERVENTION_RESEARCH_PLAN.md`.

---

## Phase P1: Intervention Types and Generator (No Rendering)

### Prerequisite
- Repository at commit `f7439e2` or descendant on branch `feature/intervention-v1`
- MuJoCo environment functional (verified by existing tests)

### Objective
Create the core dataclasses and the candidate intervention generator that takes a STOP scene specification and produces a set of candidate interventions (correct, irrelevant, hard-negative) without rendering any images.

### Files to Create

#### `src/interventions/__init__.py`
Empty package init.

#### `src/interventions/intervention_types.py`
Dataclasses:
- `Action(action_type: str, target: str, arguments: Dict[str, str])`
- `CandidateObject(name: str, object_type: str, role: str, position: List[float], orientation: List[float], is_culprit: bool)`
- `Intervention(intervention_id: str, operator: str, object_name: str, destination: str, destination_pose: List[float], intervention_type: str)`
- `InterventionOutcome(intervention: Intervention, post_feasible: bool, causal_effect: int, validation_details: Dict)`

All should be frozen dataclasses or simple classes with `to_dict()` and `from_dict()` serialization.

#### `src/interventions/intervention_generator.py`
Class `InterventionGenerator`:
- `__init__(self)` — no heavy state
- `generate_candidates(self, model, data, task_id, culprit_names, distractor_names, rng) -> List[Intervention]`
  - For each culprit: generate RELOCATE(culprit, safe_region) — correct
  - For each culprit: generate RELOCATE(culprit, still_obstructing_pose) — hard negative
  - For each distractor: generate RELOCATE(distractor, safe_region) — irrelevant
  - Generate NONE — identity control
  - Use scene_utils functions for sampling positions
  - For Task 1 "still_obstructing" = sample another position ON the lid
  - For Task 2 "still_obstructing" = sample another position IN the target
  - For "safe_region" = sample position beside box (T1) or outside target (T2)
  - Return shuffled list with randomized intervention_idx

### Tests to Create

#### `tests/test_interventions.py`
1. `test_generate_candidates_correct_count` — For 1 culprit + 1 distractor, get exactly 4 interventions (correct, hard_neg, irrelevant, none)
2. `test_correct_intervention_safe_region` — Correct intervention destination is physically off the lid / outside target
3. `test_hard_negative_still_obstructs` — Hard negative destination is still on lid / in target
4. `test_none_intervention_is_identity` — NONE has zero destination change
5. `test_intervention_ids_unique` — All intervention_ids are unique within a scene
6. `test_intervention_idx_randomized` — Over 100 calls, intervention_idx order is not fixed (check that correct intervention doesn't always appear first)

### Expected Output
- 3 new source files in `src/interventions/`
- 1 new test file with 6 passing tests
- No rendering, no images, no GPU needed

### Stopping Condition
All 6 tests pass. `intervention_generator.generate_candidates()` produces well-formed `Intervention` objects with correct typing. Verify with `pytest tests/test_interventions.py -v`.

---

## Phase P2: Intervention Validator (Simulator Checks)

### Prerequisite
- Phase P1 complete (intervention types and generator available)
- MuJoCo environment functional

### Objective
Create the intervention validator that takes a scene (MuJoCo model + data), an intervention, and verifies its effect on action feasibility using the privileged simulator predicates. Also extend `occupancy_checks.py` with a unified dispatcher.

### Files to Create

#### `src/interventions/intervention_validator.py`
Class `InterventionValidator`:
- `validate_intervention_effect(self, model, data, intervention, task_id) -> InterventionOutcome`
  - Clone the scene state
  - Apply the intervention: move the specified object to `destination_pose`
  - Settle physics
  - Evaluate `check_action_feasibility(model, data, task_id)` 
  - Compute `causal_effect = post_feasible - pre_feasible`
  - Return `InterventionOutcome`
- `validate_all_interventions(self, model, data, interventions, task_id) -> List[InterventionOutcome]`
  - For each intervention: clone state → apply → check → restore
  - Verify that non-intervened objects didn't move

### Files to Modify

#### `src/validation/occupancy_checks.py`
Add at the bottom (do NOT modify existing functions):
```python
def check_action_feasibility(model, data, task_id, **kwargs):
    """Unified dispatcher for action feasibility checking."""
    if task_id == "task_1":
        is_occupied, culprits, measurements = check_lid_occupancy(model, data, **kwargs)
        return not is_occupied  # feasible = lid is NOT occupied
    elif task_id == "task_2":
        is_occupied, culprits, measurements = check_target_occupancy(model, data, **kwargs)
        return not is_occupied  # feasible = target is NOT occupied
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
```

### Tests to Add to `tests/test_interventions.py`

7. `test_correct_intervention_flips_feasibility` — Build a STOP scene for Task 1 with a blocker on lid. Apply RELOCATE(blocker, safe_region). Verify feasibility changes from False to True.
8. `test_irrelevant_preserves_infeasibility` — Same STOP scene. Apply RELOCATE(distractor, safe_region). Verify feasibility remains False.
9. `test_hard_negative_preserves_infeasibility` — Apply RELOCATE(blocker, still_on_lid). Verify feasibility remains False.
10. `test_none_preserves_state` — Apply NONE intervention. Verify scene state unchanged.
11. `test_only_declared_object_changes` — After applying an intervention, verify all other objects are within tolerance of original positions.

### Expected Output
- 1 new source file: `src/interventions/intervention_validator.py`
- 1 modified file: `src/validation/occupancy_checks.py` (addition only)
- 5 new tests (7-11) in `tests/test_interventions.py`

### Stopping Condition
All 11 tests pass. `validate_intervention_effect()` correctly produces `InterventionOutcome` with accurate `causal_effect` labels verified against simulator ground truth. Run: `pytest tests/test_interventions.py -v`.

---

## Phase P3: Intervention Scene Generator (Rendering + Records)

### Prerequisite
- Phases P1 and P2 complete
- MuJoCo rendering functional (EGL)

### Objective
Create the full pipeline that takes a scene specification, generates all candidate interventions, renders pre/post images, and produces intervention record JSONL.

### Files to Create

#### `src/interventions/intervention_scene_generator.py`
Class `InterventionSceneGenerator`:
- `__init__(self, output_dir, resolution=(640, 480), camera_name="robot0:ego_camera")`
- `generate_scene_with_interventions(self, scene_spec, interventions, task_id, rng) -> List[InterventionRecord]`
  - Build scene using `SceneBuilder`
  - Apply observation rig
  - Render pre-intervention RGB, segmentation, masks
  - For each intervention:
    - Clone state
    - Apply intervention (move object)
    - Settle physics
    - Render post-intervention RGB, segmentation
    - Validate effect via `InterventionValidator`
    - Save images to disk
    - Create `InterventionRecord` with all paths and labels
  - Return list of records
- `generate_manifest(self, records, output_path)` — Write JSONL manifest
- Uses existing `OffscreenRenderer`, `SceneBuilder`, observation rig infrastructure

The generator should reuse patterns from `counterfactual_generator.py` for:
- Scene construction
- Camera/lighting setup
- Mask generation
- Deterministic seeding

But should NOT modify `counterfactual_generator.py` or be part of it.

### Files to Create

#### `scripts/generate_intervention_dataset.py`
CLI entry point:
```python
parser.add_argument("--config", required=True)
parser.add_argument("--profile", choices=["smoke", "pilot", "full"])
```
Reads config YAML, iterates over scene specifications, calls `InterventionSceneGenerator`.

#### `configs/intervention/smoke.yaml`
```yaml
profile_name: "intervention_smoke"
seed: 200
output_dir: "data/intervention_v1"
num_scenes_per_task: 4  # 4 STOP scenes per task = 8 total
interventions_per_scene: 4  # correct, hard_neg, irrelevant, none
resolution: [640, 480]
camera_name: "robot0:ego_camera"
```

### Tests to Add

12. `test_intervention_scene_deterministic` — Same seed → identical outputs (RGB max diff ≤ 5)
13. `test_post_rgb_matches_state` — Post-intervention RGB shows object in new position
14. `test_manifest_records_complete` — All required fields present in every JSONL record

### Expected Output
- 1 new source file: `src/interventions/intervention_scene_generator.py`
- 1 new script: `scripts/generate_intervention_dataset.py`
- 1 new config: `configs/intervention/smoke.yaml`
- 3 new tests (12-14)

### Stopping Condition
Smoke dataset generates without error. `data/intervention_v1/` contains scene directories with pre/post RGB, segmentation, and masks. JSONL manifest has 32 records (8 scenes × 4 interventions). All tests pass.

---

## Phase P4: Smoke Dataset Validation

### Prerequisite
- Phase P3 complete, smoke dataset generated

### Objective
Create validation script for intervention datasets. Run full validation on smoke data. Create end-to-end smoke test script.

### Files to Create

#### `scripts/validate_intervention_dataset.py`
Validates:
- All file paths exist
- All required fields present
- `causal_effect` matches independently re-evaluated feasibility
- No pair_id leakage across splits
- Intervention ordering randomization check
- Label balance check
- Object mask validity

#### `scripts/run_intervention_smoke.sh`
End-to-end:
1. Generate intervention smoke dataset
2. Validate dataset
3. Run unit tests
4. Report status

### Tests

15. `test_smoke_validation_passes` — Full smoke dataset passes validation

### Expected Output
- 2 new scripts
- 1 new test
- Clean smoke pipeline exit code 0

### Stopping Condition
`bash scripts/run_intervention_smoke.sh` exits 0. All validation checks pass. Manual inspection of 4+ intervention pairs confirms physical correctness.

---

## Phase P5: Intervention Dataset and Feature Extraction

### Prerequisite
- Phase P4 complete (validated smoke dataset)

### Objective
Create the PyTorch dataset class for intervention records and extend feature extraction for intervention images.

### Files to Create

#### `src/learning/intervention_dataset.py`
Class `InterventionDataset(Dataset)`:
- `__init__(self, index_path, features_dir, split, ...)`
- Loads intervention JSONL records
- For each record, loads:
  - Pre-intervention query features (global + patch) — same as existing
  - Post-intervention query features (global + patch) — new
  - Intervention object crop features — new
  - Text features — same as existing
  - Demo features — same as existing
- Returns `InterventionBatch`-style dict

Class `InterventionBatchSampler(Sampler)`:
- Groups interventions by scene_id
- Ensures all interventions from same scene appear in same batch (for ranking loss)

#### Modify `scripts/precompute_features.py`
Add `--intervention-manifest` argument:
- Process post-intervention images the same way as query images
- Extract object crop features: crop image using GT bounding box, resize to 224×224, extract DINOv2 global feature
- Save as `intervention_{scene_id}_{intervention_id}_features.pt`

### Tests

#### `tests/test_intervention_dataset.py`
16. `test_dataset_loads_without_error` — Smoke dataset loads
17. `test_batch_shapes_correct` — Batch tensors have expected shapes
18. `test_all_feature_files_exist` — Every referenced feature path exists
19. `test_no_pair_leakage` — All interventions from same scene in same split

### Expected Output
- 1 new source file
- 1 modified script
- 4 new tests

### Stopping Condition
`InterventionDataset` loads smoke data. Batches have correct shapes. All tests pass.

---

## Phase P6: V2 Model Architecture

### Prerequisite
- Phase P5 complete (dataset loads)

### Objective
Create the intervention-conditioned relational model (V2) and clean model registry.

### Files to Create

#### `src/learning/models/model_registry.py`
```python
def get_model(config: dict) -> nn.Module:
    """Clean model dispatch replacing inspect.signature hack."""
```
Supports: query_only, pooled_multimodal, relational, intervention_relational

#### `src/learning/models/intervention_relational.py`
Class `InterventionConditionedRelationalModel(nn.Module)`:
- Inherits structure from `DemoLanguageConditionedRelationalModel`
- Adds:
  - `proj_interv = nn.Linear(896, latent_dim)` — intervention projection
  - `interv_type_embed = nn.Embedding(2, 64)` — NONE/RELOCATE
  - `interv_dest_embed = nn.Embedding(4, 64)` — destination types
  - Temporal position embedding extended to `1 + K + 1`
- `forward(self, text_feat, demo_global, query_patch, interv_object_crop, interv_type_idx, interv_dest_idx)`
  - Builds intervention descriptor: concat(crop, type_embed, dest_embed)
  - Projects to latent: z_int
  - Appends z_int to temporal sequence
  - Self-attention → cross-attention → classifier
  - Returns (logit_post, ranking_score, Z_R)

### Tests

#### `tests/test_intervention_models.py`
20. `test_v2_forward_shapes` — Output shapes match spec
21. `test_v2_none_intervention` — NONE intervention produces valid output
22. `test_v2_gradient_through_intervention` — Intervention parameters receive gradients
23. `test_model_registry_dispatch` — All model types correctly dispatched

### Expected Output
- 2 new source files
- 4 new tests

### Stopping Condition
V2 model forward pass produces tensors of correct shapes. Gradients flow through intervention tokens. `pytest tests/test_intervention_models.py -v` passes.

---

## Phase P7: Intervention Losses and Metrics

### Prerequisite
- Phase P6 complete (model produces outputs)

### Objective
Create intervention-specific loss functions and evaluation metrics.

### Files to Create

#### `src/learning/intervention_losses.py`
Class `InterventionLoss(nn.Module)`:
- `__init__(self, lambda_feas, lambda_rank, lambda_effect, lambda_heat, margin)`
- `forward(self, predictions, batch) -> (loss, loss_dict)`
  - L_feas: BCE on post-intervention feasibility
  - L_rank_interv: margin ranking loss between correct and incorrect interventions (grouped by scene_id)
  - L_effect: MSE between predicted Δ̂ and GT Δ
  - L_heat: heatmap loss (optional)

#### `src/learning/intervention_metrics.py`
Function `compute_intervention_metrics(predictions, batch) -> dict`:
- Standard feasibility metrics (accuracy, F1, AUROC)
- Intervention top-1 accuracy (per scene)
- Intervention MRR
- Culprit top-1 accuracy
- False relevance rate
- Mean |Δ̂| for irrelevant
- Effect separation AUC

### Tests
Inline assertions in the metric functions for edge cases.

### Expected Output
- 2 new source files

### Stopping Condition
Loss computes without error on synthetic data. Metrics produce expected values on known inputs.

---

## Phase P8: Training Script and Smoke Training

### Prerequisite
- Phases P5-P7 complete

### Objective
Create the training script for intervention models and verify convergence on smoke data.

### Files to Create

#### `scripts/train_intervention_model.py`
- Loads config YAML
- Builds dataset, model, loss, optimizer
- Training loop (similar to existing `train_model.py` but using intervention interfaces)
- Saves checkpoints, metrics, resolved config
- Uses model_registry for dispatch

#### `configs/intervention/learning/intervention_relational.yaml`
```yaml
model: "intervention_relational"
latent_dim: 256
demo_frames: 4
batch_size: 4
learning_rate: 1e-4
epochs: 50
lambda_feas: 1.0
lambda_rank: 0.3
lambda_effect: 0.5
lambda_heat: 0.3
heatmap: true
mixed_precision: true
index_path: "data/intervention_v1/manifests/intervention_manifest.jsonl"
feature_cache_path: "data/intervention_v1/features"
output_path: "learning_outputs/intervention_v1/intervention_relational"
```

#### Baseline configs
- `intervention_query_only.yaml`
- `intervention_feasibility_only.yaml`

### Tests

24. `test_training_converges_smoke` — Training loss decreases over 10 epochs on smoke data
25. `test_checkpoint_saves` — Best checkpoint file exists after training

### Expected Output
- 1 new script
- 3+ new configs
- 2 new tests

### Stopping Condition
Smoke training runs to completion. Loss decreases. Checkpoint saved. `pytest tests/test_intervention_models.py -v` passes.

---

## Phase P9: Evaluation and Oracle Verification

### Prerequisite
- Phase P8 complete (trained model exists)

### Objective
Create evaluation script and verify oracle intervention selection achieves expected CFR.

### Files to Create

#### `scripts/evaluate_intervention_model.py`
- Loads trained model
- Runs on val split
- Computes all intervention metrics
- Optionally: runs oracle CFR in simulator (loads scene, applies predicted intervention, checks feasibility)
- Saves results JSON

### Tests

26. `test_oracle_cfr_100` — Oracle (argmax GT Δ) achieves ≥ 95% CFR
27. `test_query_only_shortcut_bound` — Query-only model ≤ 1/N + margin

### Expected Output
- 1 new script
- 2 new tests
- `artifacts/intervention_v1/oracle_verification.json`

### Stopping Condition
Oracle CFR verified ≥ 95%. Query-only model at or near chance. Evaluation script produces complete JSON results.

---

## Phase P10: Pilot Dataset Generation

### Prerequisite
- Phase P4 complete (smoke pipeline works)

### Objective
Generate the full pilot intervention dataset (~60 scenes × 4-6 interventions).

### Files to Create

#### `configs/intervention/pilot.yaml`
```yaml
profile_name: "intervention_pilot"
seed: 300
output_dir: "data/intervention_v1_pilot"
num_scenes_per_task: 30
interventions_per_scene: 4-6
resolution: [640, 480]
```

### Expected Output
- 240-360 intervention records
- Pre/post images for each
- Validated manifest
- Features extracted

### Stopping Condition
Dataset validation passes. Feature extraction completes. Manual spot-check of 10 records.

---

## Phase P11: Multi-Seed Experiments and Baselines

### Prerequisite
- Phases P8-P10 complete

### Objective
Run Experiments 1-6 from the research plan with 5-seed protocol.

### Expected Output
- Results for: query_only, feasibility_only, direct_culprit, intervention_relational
- 5 seeds each: 11, 23, 42, 67, 101
- Aggregated metrics with bootstrap CIs
- Gate G3 evaluation

### Stopping Condition
All experiments complete. Gate G3 quantitative thresholds evaluated. Results saved to `artifacts/intervention_v1/`.

---

## Phase P12: Ablation Studies

### Prerequisite
- Phase P11 complete

### Objective
Run the 10-ablation matrix from Section 11.

### Expected Output
- 10 ablations × 5 seeds = 50 runs
- Ablation table
- Statistical significance tests

### Stopping Condition
All ablations complete. Table shows clear contribution of each component.

---

## Phase P13: Representation Probes and Analysis

### Prerequisite
- Phase P11 complete

### Objective
Implement linear probes and representation analysis tools.

### Files to Create

#### `src/evaluation/representation_probes.py`
- `LinearProbe`: Train linear classifier on frozen latent features
- Probes for: feasibility, culprit identity, relation type, intervention effect
- Nearest-neighbor analysis
- Representation sensitivity analysis

### Expected Output
- Probe accuracy tables
- Latent structure analysis
- Sensitivity analysis results

### Stopping Condition
Probes run without error. Results saved. Key finding: intervention-supervised model has more linearly separable culprit/relation information than feasibility-only model.

---

*Each phase should be executed sequentially. Do not proceed to phase N+1 until phase N's stopping condition is met.*
