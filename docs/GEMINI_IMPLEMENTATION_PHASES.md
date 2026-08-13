# Gemini 3.1 Pro Implementation Phases

> Self-contained implementation prompts for sequential execution by a coding agent.
> Each phase has explicit prerequisites, objectives, files, tests, and stopping conditions.
> Based on the master plan at `docs/CAUSAL_INTERVENTION_RESEARCH_PLAN.md`.

---

## Phase P1: Intervention Types and Generator (No Rendering)

### Prerequisite
- Repository at commit `35758b5` or descendant on branch `feature/causal-intervention-v2`
- MuJoCo environment functional (verified by existing tests)

### Objective
Create the core dataclasses and the candidate intervention generator that takes a scene specification and produces a set of candidate interventions (repair, irrelevant, hard-negative, harmful, identity) without rendering any images.

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
- `generate_candidates(self, model, data, task_id, culprit_names, distractor_names, rng, is_stop_scene=True) -> List[Intervention]`
  - For STOP scenes:
    - For culprit: generate RELOCATE(culprit, safe_region) — repair ($\Delta = +1$)
    - For culprit: generate RELOCATE(culprit, still_obstructing_pose) — hard negative ($\Delta = 0$)
    - For each distractor: generate RELOCATE(distractor, safe_region) — irrelevant ($\Delta = 0$)
    - Generate NONE — identity control ($\Delta = 0$)
  - For PROCEED scenes:
    - For distractor: generate RELOCATE(distractor, obstructing_pose) — harmful ($\Delta = -1$)
    - For each distractor: generate RELOCATE(distractor, safe_region) — irrelevant ($\Delta = 0$)
    - Generate NONE — identity control ($\Delta = 0$)
  - Use scene_utils functions for sampling positions
  - For Task 1 "still_obstructing" = sample another position ON the lid
  - For Task 2 "still_obstructing" = sample another position IN the target
  - For "safe_region" = sample position beside box (T1) or outside target (T2)
  - Return shuffled list with randomized intervention_idx

### Tests to Create

#### `tests/test_interventions.py`
1. `test_generate_candidates_stop_count` — For 1 culprit + 1 distractor STOP scene, get exactly 4 interventions (repair, hard_neg, irrelevant, identity)
2. `test_generate_candidates_proceed_count` — For 1 distractor PROCEED scene, get exactly 3 interventions (harmful, irrelevant, identity)
3. `test_correct_intervention_safe_region` — Repair intervention destination is physically off the lid / outside target
4. `test_hard_negative_still_obstructs` — Hard negative destination is still on lid / in target
5. `test_none_intervention_is_identity` — NONE has zero destination change
6. `test_intervention_ids_unique` — All intervention_ids are unique within a scene
7. `test_intervention_idx_randomized` — Over 100 calls, intervention_idx order is not fixed (repair doesn't always appear first)

### Expected Output
- 3 new source files in `src/interventions/`
- 1 new test file with 7 passing tests
- No rendering, no images, no GPU needed

### Stopping Condition
All 7 tests pass. `intervention_generator.generate_candidates()` produces well-formed `Intervention` objects with correct typing. Verify with `pytest tests/test_interventions.py -v`.

---

## Phase P2: Intervention Validator (Simulator Checks)

### Prerequisite
- Phase P1 complete (intervention types and generator available)
- MuJoCo environment functional

### Objective
Create the intervention validator that takes a scene (MuJoCo model + data), an intervention, and verifies its effect on action feasibility using the privileged simulator predicates ($F_R$). Also extend `occupancy_checks.py` with a unified dispatcher.

### Files to Create

#### `src/interventions/intervention_validator.py`
Class `InterventionValidator`:
- `validate_intervention_effect(self, model, data, intervention, task_id) -> InterventionOutcome`
  - Clone the scene state
  - Apply the intervention: move the specified object to `destination_pose`
  - Settle physics
  - Evaluate `check_action_feasibility(model, data, task_id)` 
  - Compute `causal_effect = post_feasible - pre_feasible` ($\in \{-1, 0, +1\}$)
  - Return `InterventionOutcome`
- `validate_all_interventions(self, model, data, interventions, task_id) -> List[InterventionOutcome]`
  - For each intervention: clone state → apply → check → restore
  - Verify that non-intervened objects didn't move

### Files to Modify

#### `src/validation/occupancy_checks.py`
Add at the bottom (do NOT modify existing functions):
```python
def check_action_feasibility(model, data, task_id, **kwargs):
    """Unified dispatcher for relational precondition feasibility (F_R)."""
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

8. `test_correct_intervention_flips_feasibility` — Build a STOP scene for Task 1 with a blocker on lid. Apply RELOCATE(blocker, safe_region). Verify feasibility changes from False to True (causal_effect = +1).
9. `test_irrelevant_preserves_infeasibility` — Same STOP scene. Apply RELOCATE(distractor, safe_region). Verify feasibility remains False (causal_effect = 0).
10. `test_harmful_intervention_breaks_feasibility` — Build a PROCEED scene. Apply RELOCATE(distractor, onto_lid). Verify feasibility changes from True to False (causal_effect = -1).
11. `test_hard_negative_preserves_infeasibility` — Apply RELOCATE(blocker, still_on_lid). Verify feasibility remains False (causal_effect = 0).
12. `test_none_preserves_state` — Apply NONE intervention. Verify scene state unchanged.
13. `test_only_declared_object_changes` — After applying an intervention, verify all other objects are within tolerance of original positions.

### Expected Output
- 1 new source file: `src/interventions/intervention_validator.py`
- 1 modified file: `src/validation/occupancy_checks.py` (addition only)
- 6 new tests (8-13) in `tests/test_interventions.py`

### Stopping Condition
All 13 tests pass. `validate_intervention_effect()` correctly produces `InterventionOutcome` with accurate `causal_effect` labels verified against simulator ground truth. Run: `pytest tests/test_interventions.py -v`.

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
- `generate_scene_interventions(self, scene_spec) -> List[InterventionRecord]`
  - Sets up base scene (STOP or PROCEED)
  - Renders pre-intervention observation (RGB, segmentation, candidate crops)
  - Generates candidate interventions via `InterventionGenerator`
  - For each candidate:
    - Validates effect via `InterventionValidator`
    - Renders post-intervention image (for GT validation and visualization only)
    - Constructs JSONL record
  - Saves all images and writes JSONL

---

## Phase P4: Smoke Dataset Generation & Validation

### Prerequisite
- Phase P3 complete

### Objective
Generate the definitive smoke intervention dataset (6 base scenes, 22 candidate records) exercising all causal transitions $\Delta \in \{-1, 0, +1\}$ across Task 1 and Task 2. Validate with automated dataset integrity checks.

### Smoke Dataset Composition (6 Scenes, 22 Records):
- **Task 1 (`OPEN`)**:
  - 2 STOP scenes (1 culprit + 1 distractor): $2 \times 4 = 8$ records ($\Delta \in \{+1, 0, 0, 0\}$)
  - 1 PROCEED scene (1 distractor): $1 \times 3 = 3$ records ($\Delta \in \{-1, 0, 0\}$)
- **Task 2 (`PLACE`)**:
  - 2 STOP scenes (1 culprit + 1 distractor): $2 \times 4 = 8$ records ($\Delta \in \{+1, 0, 0, 0\}$)
  - 1 PROCEED scene (1 distractor): $1 \times 3 = 3$ records ($\Delta \in \{-1, 0, 0\}$)

### Files to Create

#### `scripts/validate_intervention_dataset.py`
Validates:
- All referenced image and feature paths exist
- All required schema fields present under strict partitioning
- `causal_effect` matches independently re-evaluated simulator state
- No scene_id / pair_id leakage across splits
- Intervention index randomization verified
- All 3 effect values $\{-1, 0, +1\}$ present and verified

#### `scripts/run_intervention_smoke.sh`
End-to-end master runner:
1. Generate intervention smoke dataset
2. Validate dataset
3. Run unit tests
4. Report exit code 0

### Tests
14. `test_smoke_validation_passes` — Full smoke dataset passes validation
15. `test_smoke_exercises_all_effects` — Dataset contains non-zero counts for $\Delta=+1$, $\Delta=0$, and $\Delta=-1$.

### Stopping Condition
`bash scripts/run_intervention_smoke.sh` exits 0. All 22 records pass validation.

---

## Phase P5: Intervention Dataset and Feature Extraction

### Prerequisite
- Phase P4 complete (validated smoke dataset)

### Objective
Create the PyTorch dataset class for candidate intervention records and extend feature extraction. Enforce strict isolation between model forward inputs and privileged metadata.

### Files to Create

#### `src/learning/intervention_dataset.py`
Class `InterventionDataset(Dataset)`:
- `__init__(self, index_path, features_dir, split, ...)`
- Loads intervention JSONL records.
- Explicitly partitions output into:
  - `model_inputs`: `text_feat`, `demo_global`, `query_patch`, `interv_object_crop`, `interv_current_geom`, `interv_dest_geom`, `interv_operator_idx`.
  - `supervision_targets`: `pre_feasible`, `post_feasible`, `causal_effect`.
  - `privileged_metadata`: `scene_id`, `pair_id`, `task_id`, `intervention_id`, `intervention_type`, `is_culprit`.
- Enforces that NO post-intervention visual features (`post_rgb`, `post_features`) are returned in `model_inputs`.

Class `InterventionBatchSampler(Sampler)`:
- Groups interventions by `scene_id` so all candidates for a scene appear in the same batch (required for grouped ranking loss).

#### Modify `scripts/precompute_features.py`
Add `--intervention-manifest` argument:
- Precomputes DINOv2 global feature for candidate object crops.
- Precomputes pre-scene query patch and global features.

### Tests

#### `tests/test_intervention_dataset.py`
16. `test_dataset_loads_without_error` — Smoke dataset loads
17. `test_batch_shapes_correct` — Batch tensors have expected parameterized shapes
18. `test_no_post_state_leakage` — Asserts that `model_inputs` dictionary contains zero post-intervention tensors
19. `test_no_pair_leakage` — All interventions from same base scene reside in same split

### Expected Output
- 1 new source file: `src/learning/intervention_dataset.py`
- 1 modified script: `scripts/precompute_features.py`
- 4 new tests

### Stopping Condition
`InterventionDataset` loads smoke data. Zero post-state leakage verified. All tests pass.

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
Supports: `query_only` (B0), `simple_intervention` (B0b), `relational_feasibility_only` (B3), `intervention_relational` (V2).

#### `src/learning/models/intervention_relational.py`
Class `InterventionConditionedRelationalModel(nn.Module)`:
- Parameterized input dimension:
  `interv_input_dim = vision_dim + current_geom_dim + dest_geom_dim + operator_embed_dim`
  `proj_interv = nn.Linear(interv_input_dim, latent_dim)`
- `operator_embed = nn.Embedding(2, 64)` — index 0: `NONE`, index 1: `RELOCATE`
- Temporal position embedding extended to `1 + K + 1`
- `forward(self, text_feat, demo_global, query_patch, interv_object_crop, interv_current_geom, interv_dest_geom, interv_operator_idx)`
  - Constructs neutral descriptor: `concat(crop, current_geom, dest_geom, operator_embed(operator_idx))`
  - Projects to latent: `z_int = proj_interv(...)`
  - Appends `z_int` to temporal sequence `[z_t, z_d_1..K, z_int]`
  - Temporal self-attention $\to$ cross-attention (query patches attend context) $\to$ mean pool $\to$ classifier MLP
  - Returns `(logit_post, ranking_score, Z_R)`

### Tests

#### `tests/test_intervention_models.py`
20. `test_v2_forward_shapes` — Output tensor shapes match spec
21. `test_v2_none_intervention` — NONE intervention (operator=0) produces valid output
22. `test_v2_gradient_through_intervention` — Intervention parameters and geometry features receive gradients
23. `test_model_registry_dispatch` — All model types correctly dispatched

### Stopping Condition
V2 model forward pass produces tensors of correct shapes. Gradients flow cleanly. `pytest tests/test_intervention_models.py -v` passes.

---

## Phase P7: Intervention Losses and Metrics

### Prerequisite
- Phase P6 complete

### Objective
Create minimal canonical intervention loss function and evaluation metrics.

### Files to Create

#### `src/learning/intervention_losses.py`
Class `InterventionLoss(nn.Module)`:
- `__init__(self, lambda_feas=1.0, lambda_rank=0.3, margin=0.3)`
- `forward(self, predictions, batch) -> (loss, loss_dict)`
  - $\mathcal{L}_{post\_feas}$: `BCEWithLogitsLoss(logit_post, batch["post_feasible"])`
  - $\mathcal{L}_{rank\_interv}$: `MarginRankingLoss` between pairs of candidate interventions from the same scene
  - Canonical loss: $\mathcal{L}_{V2} = \mathcal{L}_{post\_feas} + \lambda_{rank} \cdot \mathcal{L}_{rank\_interv}$

#### `src/learning/intervention_metrics.py`
Function `compute_intervention_metrics(predictions, batch) -> dict`:
- Post-feasibility metrics (Accuracy, F1, AUROC)
- Intervention Top-1 accuracy (per scene candidate set)
- Counterfactual Repair Rate (CFR)
- False Relevance Rate (FRR) on irrelevant / identity interventions
- Hard Negative separation AUC

### Stopping Condition
Loss and metrics compute without error on synthetic batches.

---

## Phase P8: Training Script and Smoke Training

### Prerequisite
- Phases P5-P7 complete

### Objective
Create the training script for intervention models and verify convergence on smoke data.

### Files to Create

#### `scripts/train_intervention_model.py`
- Training loop using intervention dataset interfaces
- Dispatches model via `model_registry`
- Handles grouped batch sampling and evaluation logging

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
mixed_precision: true
index_path: "data/intervention_v1/manifests/intervention_manifest.jsonl"
feature_cache_path: "data/intervention_v1/features"
output_path: "learning_outputs/intervention_v1/intervention_relational"
```

#### Baseline configs:
- `configs/intervention/learning/baseline_b0_query_only.yaml`
- `configs/intervention/learning/baseline_b0b_simple_intervention.yaml`
- `configs/intervention/learning/baseline_b3_feasibility_only.yaml` ($\lambda_{rank} = 0$)

### Tests
24. `test_training_converges_smoke` — Training loss decreases over 10 epochs on smoke data
25. `test_checkpoint_saves` — Checkpoint saved properly

---

## Phase P9: Evaluation, Baselines & Oracle Verification

### Prerequisite
- Phase P8 complete

### Objective
Create evaluation script and verify oracle intervention selection achieves expected CFR, while evaluating against baselines B0, B0b, and B3.

### Files to Create

#### `scripts/evaluate_intervention_model.py`
- Evaluates model on validation split
- Computes Top-1, CFR, FRR, AUC
- Runs simulator oracle CFR verification (executes predicted top-1 intervention in MuJoCo and checks $F_R$)

### Tests
26. `test_oracle_cfr_upper_bound` — Oracle intervention selection ($\arg\max \Delta_i^{GT}$) achieves $\ge 95\%$ CFR
27. `test_b0b_candidate_ranking_baseline` — Simple candidate MLP baseline (B0b) evaluated on candidate ranking
28. `test_b3_feasibility_only_ablation` — Feasibility-only relational model (B3) evaluated without ranking loss

### Stopping Condition
Oracle CFR verified $\ge 95\%$. Evaluation script outputs complete JSON reports.

---

## Phase P10: Pilot Dataset Generation

### Prerequisite
- Phase P4 complete

### Objective
Generate the full pilot intervention dataset (~60 scenes $\times$ 4-6 candidates $\approx$ 240-360 records) and extract features.

---

## Phase P11: Multi-Seed Experiments and Baselines

### Prerequisite
- Phases P8-P10 complete

### Objective
Execute the 5-seed protocol (seeds: 11, 23, 42, 67, 101) comparing B0, B0b, B3, B4, and V2 with 10,000 bootstrap resamples.

---

## Phase P12: Ablation Studies

### Prerequisite
- Phase P11 complete

### Objective
Execute the leave-one-out ablation matrix (A1: No Intervention, A2: No Text, A3: No Demo, A4: No Ranking Loss, A8: No Crop, A9: No Geometry).

---

## Phase P13: Representation Probes and Analysis

### Prerequisite
- Phase P11 complete

### Objective
Train linear probes on frozen latent representations (`z_S`, `z_R`) for feasibility, culprit identity, and violated relations.

---

*Each phase should be executed sequentially. Do not proceed to phase N+1 until phase N's stopping condition is met.*

