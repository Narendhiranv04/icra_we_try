# Causal Intervention Research Plan — Master Blueprint

> **Repository**: `/home/projects/long-horizon/infeasibiilty-latent`
> **Branch**: `feature/context-forcing-v1-diagnostic`
> **HEAD commit**: `f7439e2` ("Complete context_forcing_v1 diagnostic pilot")
> **Audit date**: 2026-08-13
> **Auditor**: Claude Opus 4.6 (planning-only turn)

---

## Table of Contents

1. [Current-State Audit](#1-current-state-audit)
2. [Freeze / Reuse Decisions](#2-freeze--reuse-decisions)
3. [Scientific Claim Ladder](#3-scientific-claim-ladder)
4. [Dataset V2 Design](#4-dataset-v2-design)
5. [Architecture V2 Design](#5-architecture-v2-design)
6. [Architecture V3 Design](#6-architecture-v3-design)
7. [Loss Design](#7-loss-design)
8. [Data / Model Interfaces](#8-data--model-interfaces)
9. [Repository Change Plan](#9-repository-change-plan)
10. [Experiment Plan](#10-experiment-plan)
11. [Ablation Matrix](#11-ablation-matrix)
12. [Generalization Matrix](#12-generalization-matrix)
13. [Metric Definitions](#13-metric-definitions)
14. [Test Plan](#14-test-plan)
15. [Compute Plan](#15-compute-plan)
16. [Checkpoint / Go-No-Go Gates](#16-checkpoint--go-no-go-gates)
17. [Paper-Facing Experiment Story](#17-paper-facing-experiment-story)
18. [Failure Modes](#18-failure-modes)
19. [Implementation Order for Gemini 3.1 Pro](#19-implementation-order-for-gemini-31-pro)
20. [Master TODO](#20-master-todo)

---

## 1. Current-State Audit

### 1.1 Branch and Commit

| Field | Value |
|-------|-------|
| Branch | `feature/context-forcing-v1-diagnostic` |
| HEAD | `f7439e2` |
| Commit message | "Complete context_forcing_v1 diagnostic pilot (Outcome 1/2 Confirmed)" |
| Working tree | Clean (modified: `artifacts/learning_stage1/pytest.xml`; untracked: scratch image scripts) |
| Origin/main | `feature/demo-conditioned-relational-latent` at `00e05b0` |

### 1.2 Important Modules (Verified from Source)

#### Environment Layer (`src/environment/`)
| File | Purpose | Status |
|------|---------|--------|
| `scene_builder.py` (12.7KB) | Dynamic XML scene construction, object spawning, background/lighting application | ✅ Production-validated |
| `scene_utils.py` (9.0KB) | Canonical geometry queries: lid/target frames, local↔world transforms, sample positions | ✅ Production-validated |
| `renderer.py` (13.3KB) | Offscreen MuJoCo rendering: RGB, segmentation, culprit masks, region masks | ✅ Production-validated |
| `observation_rig.py` (5.4KB) | Camera/robot-base-pose rigs (egocentric, context-challenge) | ✅ Production-validated |
| `robot_integration.py` (14.8KB) | Fetch robot qpos initialization, joint configuration | ✅ Production-validated |
| `model_loading.py` (3.2KB) | MuJoCo XML loading utilities | ✅ Production-validated |

#### Generation Layer (`src/generation/`)
| File | Purpose | Status |
|------|---------|--------|
| `counterfactual_generator.py` (57.7KB) | Master matched-pair STOP/PROCEED scene generation with masks, invariants, privileged checks | ✅ Production-validated (1207 lines) |
| `context_forcing_generator.py` (15.7KB) | Deconfounded dataset: same RGB → different task labels, Benchmark B blocklist | ✅ Validated for M0 diagnostic |
| `context_challenge_generator.py` (13.0KB) | Benchmark B: same-RGB context-reversal challenge set | ✅ Validated |
| `demonstration_generator.py` (10.8KB) | Genuine Fetch robot demo generation (open-box, pick-place) | ✅ Production-validated |
| `split_planner.py` (7.3KB) | ID/unseen-object/unseen-background/compositional split construction | ✅ Production-validated |
| `background_randomization.py` (4.9KB) | Background/lighting randomization | ✅ Production-validated |
| `scene_config.py` (4.2KB) | EpisodeSpec dataclass, seed management | ✅ Production-validated |
| `object_randomization.py` (2.3KB) | Object identity/pose randomization | ✅ Validated |

#### Validation Layer (`src/validation/`)
| File | Purpose | Status |
|------|---------|--------|
| `occupancy_checks.py` (14.6KB) | Privileged simulator occupancy predicates: `check_lid_occupancy`, `check_target_occupancy` with footprint overlap, contact, vertical gap, settling | ✅ Production-validated |
| `dataset_validator.py` (39.3KB) | Comprehensive manifest validation: uniqueness, file integrity, mask checks, split leakage | ✅ Production-validated |
| `demonstration_validator.py` (23.8KB) | Demo video/metadata validation with physical invariant checks | ✅ Production-validated |
| `demonstration_distinctness.py` (6.3KB) | Inter-demo content overlap detection | ✅ Validated |

#### Learning Layer (`src/learning/`)
| File | Purpose | Status |
|------|---------|--------|
| `dataset.py` (8.5KB) | `LearningDataset` (precomputed features), `PairBatchSampler`, ablation modes | ✅ Production-validated |
| `trainer.py` (8.3KB) | `train_epoch`, `validate_epoch` with mixed precision, gradient logging | ✅ Production-validated |
| `losses.py` (2.9KB) | `LearningLoss`: BCE + MarginRankingLoss + DiceLoss for heatmap | ✅ Production-validated |
| `metrics.py` (4.8KB) | Accuracy, balanced accuracy, F1, pair consistency, latent PLA, IoU/Dice for heatmaps, per-task breakdowns | ✅ Production-validated |
| `vision_encoder.py` (2.0KB) | DINOv2 ViT-B/14, frozen, global+patch features | ✅ Validated |
| `text_encoder.py` (1.8KB) | MiniLM-L6-v2, frozen, mean-pooled embeddings | ✅ Validated |

#### Model Architectures (`src/learning/models/`)
| File | Purpose | Status |
|------|---------|--------|
| `query_only.py` (580B) | MLP baseline: query_global → logit | ✅ Validated |
| `pooled_multimodal.py` (2.1KB) | Mean-pooled context + query with ablation switches for text/demo | ✅ Validated |
| `relational_model.py` (4.3KB) | Cross-attention: [text, demo_frames] self-attn → query_patch cross-attn → classifier + ranking score + Z_R | ✅ Validated |
| `heatmap_decoder.py` (1.8KB) | CNN decoder: 16×16 patch tokens → 224×224 causal heatmap | ✅ Validated |

### 1.3 Current Datasets

| Dataset | Records | Purpose | Location |
|---------|---------|---------|----------|
| Smoke | 16 | CI/CD regression | `data/` (generated) |
| Pilot | 152 (120 pairs + 32 controls) | Primary learning dataset | `data/` (generated) |
| Context Challenge (Benchmark B) | 80 (40 physical scenes × 2 tasks) | Same-RGB context-reversal test | `data/context_challenge/` |
| Context Forcing V1 | ~128 train + ~64 val (native splits) | Deconfounded forcing dataset | `data/context_forcing_v1/` |

### 1.4 Current Experimental Findings (Verified)

#### Milestone 1 — Representation Learning (COMPLETE)
- 30 runs: 6 model families × 5 seeds (11, 23, 42, 67, 101)
- **Key finding**: All models achieve 0.0 reversal accuracy on Benchmark B
- Query-only achieves high accuracy on Benchmark A → visual shortcuts
- Relational model: 100% logical test accuracy, but shortcut-dependent

#### Milestone 2 — Context Challenge (COMPLETE)
- Benchmark B demonstrates shortcut exploitation definitively
- Models cannot flip predictions for identical RGB with swapped task context
- Strong evidence of shortcut-permissive data, not architectural inability

#### Context Forcing V1 (COMPLETE — current branch)
- **query_only**: 0.703 val accuracy (at chance boundary for forced-context task)
- **pooled_multimodal**: 0.977 val accuracy
- **relational_heatmap**: 1.000 val accuracy
- **Conclusion**: Architectures *can* condition on context when shortcuts are removed. The earlier failure was data-driven.

### 1.5 What Is Validated

- ✅ Deterministic MuJoCo scene generation and reconstruction
- ✅ Matched counterfactual pairs with minimal interventions
- ✅ Privileged simulator occupancy predicates
- ✅ Causal/candidate/target masks
- ✅ DINOv2 + MiniLM feature extraction with provenance tracking
- ✅ Pair-based batch sampling preserving counterfactual structure
- ✅ Cross-attention relational architecture conditions on text+demo
- ✅ Heatmap decoder produces causal region maps
- ✅ ID/unseen-object/unseen-background/compositional splits
- ✅ 5-seed multi-run protocol with bootstrap metrics
- ✅ Context challenge / Benchmark B protocol
- ✅ Context forcing deconfounded dataset generation

### 1.6 What Is Incomplete

- ❌ No candidate-intervention generation per scene
- ❌ No intervention-effect labels (Δ_i)
- ❌ No hard-negative interventions
- ❌ No intervention-conditioned model architecture
- ❌ No object-centric / factored representation
- ❌ No explicit action/plan representation beyond natural language
- ❌ No demonstration trajectory (EE pose, gripper state) features
- ❌ No residual-plan conditioning
- ❌ No discovery-triggered relevance assessment
- ❌ No TAMP integration
- ❌ No within-episode memory component

### 1.7 Technical Debt / Scientific Risk

1. **`counterfactual_generator.py` is 1207 lines** — monolithic. Intervention generation should NOT further bloat this file. New module needed.
2. **`trainer.py` uses `inspect.signature` dispatch** — fragile for adding new model families. Should introduce a model interface protocol.
3. **No structured action representation** — instructions are plain text strings. Need explicit `Action` dataclass.
4. **No variable object count support** — current scenes have 1-2 candidate objects, hardcoded in many places.
5. **No AUROC/AUPRC** in current metrics — needed for calibration analysis.
6. **Label convention**: current code uses `STOP=1, PROCEED=0`. The intervention framework uses `feasible=1, infeasible=0`. These are complementary (`feasible = 1 - STOP_label`). The mapping must be handled explicitly and consistently.
7. **Pilot dataset is small** (120 pairs = 240 images). Intervention expansion will need more scenes or the same scenes with multiple intervention candidates.

---

## 2. Freeze / Reuse Decisions

### 2.1 FREEZE (do not modify)

| Module | Reason |
|--------|--------|
| `artifacts/smoke/` | Release-verified smoke artifacts |
| `artifacts/learning_stage1/` | Completed M1/M2 experimental results |
| `artifacts/context_forcing_v1/` | Completed M0 diagnostic results |
| `src/validation/dataset_validator.py` | Exhaustively validated, should only be *extended* for new record types |
| `src/validation/demonstration_validator.py` | Completed demo validation |
| `src/validation/demonstration_distinctness.py` | Completed distinctness analysis |
| `tests/test_benchmark.py` | 19 regression tests, all passing |
| `tests/test_final_release_regressions.py` | Release regression suite |
| Existing learning configs under `configs/learning/` | Reproducibility of existing runs |
| Existing context_forcing configs under `configs/context_forcing_v1/` | Reproducibility |

### 2.2 REUSE AS-IS

| Module | Reason |
|--------|--------|
| `src/environment/scene_builder.py` | Core scene construction is mature; intervention gen will *call* it, not modify it |
| `src/environment/scene_utils.py` | Geometry queries are canonical and complete for current tasks |
| `src/environment/renderer.py` | Rendering pipeline is stable |
| `src/environment/observation_rig.py` | Camera rigs are stable |
| `src/environment/robot_integration.py` | Robot initialization is stable |
| `src/learning/vision_encoder.py` | DINOv2 extraction is validated |
| `src/learning/text_encoder.py` | MiniLM extraction is validated |
| `src/generation/background_randomization.py` | Background sampling is complete |
| `src/generation/split_planner.py` | Split construction logic is validated |
| `src/generation/scene_config.py` | EpisodeSpec is sound for current tasks |
| `src/generation/demonstration_generator.py` | Demo generation is complete |

### 2.3 EXTEND

| Module | What to add | Reason |
|--------|-------------|--------|
| `src/validation/occupancy_checks.py` | Add `check_action_feasibility(model, data, action)` wrapper that dispatches to lid/target checks based on action type. Add `evaluate_intervention_effect()`. | Needed for intervention validation. Current functions are sound but need a unified interface. |
| `src/learning/dataset.py` | Create `InterventionDataset` as a sibling class (not modifying `LearningDataset`). | New data format for intervention records. Original dataset must remain for reproducibility. |
| `src/learning/metrics.py` | Add intervention-specific metrics: culprit accuracy, intervention ranking accuracy, CFR. | Core metrics infrastructure. |
| `src/learning/losses.py` | Add intervention loss terms. Consider creating `InterventionLoss` as sibling. | New loss terms for intervention prediction. |
| `src/learning/trainer.py` | Add model dispatch for new architectures (replace `inspect.signature` hack with protocol). | Clean model interface needed for V2/V3 architectures. |
| `scripts/precompute_features.py` | Add intervention-scene feature extraction mode. | Same DINOv2 pipeline for new images. |
| `scripts/build_learning_index.py` | Add intervention index builder. | New manifest format for intervention records. |

### 2.4 REPLACE LATER (not now)

| Module | When | Reason |
|--------|------|--------|
| Patch-based heatmap as primary localization | After M7 (object-centric model) | Heatmaps remain as baseline/diagnostic; object-token relevance becomes primary |
| Natural-language-only action representation | After M3 | Explicit `Action` encoding needed for plan conditioning |
| Global feature pooling in pooled_multimodal | After M7 | Object-factored representation replaces global pool |

---

## 3. Scientific Claim Ladder

Each milestone answers **one** scientific question. They are sequenced so that each depends on the previous.

### M0 — CONTEXT CONDITIONING (COMPLETE)
**Claim**: The relational cross-attention architecture can condition predictions on task/demo context when the dataset prevents visual shortcuts.
**Evidence**: Context Forcing V1 pilot — relational model achieves 1.000 val accuracy on deconfounded data.
**Status**: ✅ Verified at `f7439e2`.

### M1 — INTERVENTION DATASET VALIDITY
**Question**: Can we generate rigorously validated candidate interventions for STOP states and measure their ground-truth effect on feasibility?
**Claim**: For each STOP scene with N candidate objects, simulator-validated interventions produce correct Δ_i ∈ {0, 1} labels, including correct=1, irrelevant=0, and hard-negative=0 cases.
**Gate**: 100% agreement between programmatic Δ_i and manual inspection on smoke set; all validation tests pass.

### M2 — SHORTCUT BASELINE
**Question**: Can a visual-only model solve intervention ranking without actually understanding the causal structure?
**Claim**: A query-only baseline cannot significantly exceed chance on intervention ranking when the dataset is properly deconfounded.
**Gate**: Query-only intervention top-1 accuracy ≤ 1/N_candidates + ε.

### M3 — CAUSAL INTERVENTION PREDICTION
**Question**: Given scene, action, and demo context, can the model identify which candidate intervention restores feasibility?
**Claim**: The intervention-conditioned model significantly outperforms query-only and feasibility-only baselines on intervention top-1 accuracy and CFR.
**Gate**: Top-1 intervention accuracy > 80% on ID split; CFR > 70%.

### M4 — CAUSAL OBJECT/RELATION LOCALIZATION
**Question**: Can the model identify the responsible object and violated relation?
**Claim**: High culprit top-1 accuracy and relation accuracy, exceeding direct classification baselines lacking intervention supervision.
**Gate**: Culprit top-1 > 85% on ID; demonstrably better than B4 (no intervention supervision).

### M5 — CORRECTIVE REPAIR VALIDATION
**Question**: Does the model-selected intervention actually restore feasibility when executed in MuJoCo?
**Claim**: Oracle-selected interventions restore feasibility at near-100% rates; model-predicted interventions restore at rates significantly above random.
**Gate**: Oracle CFR ≥ 95%; model CFR > 70%.

### M6 — IRRELEVANT INTERVENTION INVARIANCE
**Question**: Can irrelevant interventions/distractors be correctly ignored?
**Claim**: Feasibility prediction and causal attribution are invariant under irrelevant interventions and distractor additions.
**Gate**: False-relevance rate < 5%; prediction Δ under irrelevant intervention < 0.05.

### M7 — OBJECT-CENTRIC REPRESENTATION
**Question**: Does an object-factored model improve causal generalization over patch/global representations?
**Claim**: Object-centric (V3) shows improved OOD generalization vs. patch-based (V2).
**Gate**: ≥ 5% improvement on unseen-object split intervention accuracy.

### M8 — TRAJECTORY DEMONSTRATION CONDITIONING
**Question**: What does target-relative robot trajectory add beyond video-only demonstration features?
**Claim**: EE pose/gripper trajectory features improve feasibility prediction.
**Gate**: Statistically significant improvement on at least one OOD split.

### M9 — RESIDUAL PLAN CONDITIONING
**Question**: Can the representation reason about feasibility w.r.t. a later action?
**Claim**: Plan-conditioned model correctly identifies objects affecting future subgoals.
**Gate**: Multi-step plan accuracy > 75% on divergent immediate/residual feasibility cases.

### M10 — DISCOVERY GATING
**Question**: Can newly discovered objects be partitioned into irrelevant, locally repairable, and replanning-relevant?
**Claim**: Correct CONTINUE / LOCAL_REPAIR / GLOBAL_REPLAN classification.
**Gate**: F1 > 0.8 for three-way classification.

### M11 — TAMP INTEGRATION
**Question**: Does the learned causal module reduce unnecessary VLM/planner calls?
**Claim**: Selective repair/replan reduces total planner calls vs. always-replan baseline.
**Gate**: ≥ 30% reduction in planner calls; no decrease in task success rate.

### M12 — WITHIN-EPISODE EXPERIENCE REUSE
**Question**: Can successful earlier manipulations be retrieved and adapted for local repair?
**Claim**: Episodic experience reuse reduces motion-planning calls and repair latency.
**Gate**: Measurable latency reduction; no decrease in repair success rate.

> **Ordering revision note**: The user's proposed M3 and M2 are swapped here. Rationale: intervention ranking is the more direct output of the intervention-supervision framework and should be verified before derived culprit localization. Object localization emerges from *which intervention has the largest Δ*, so intervention prediction logically precedes localization as a separate claim.

---

## 4. Dataset V2 Design

### 4.1 Intervention Ontology

For early tasks (M1–M6), intervention vocabulary is deliberately constrained:

```
InterventionOperator:
    NONE            # null intervention (control)
    RELOCATE        # move object to a different location
```

Each candidate intervention specifies:
- **object**: which scene object is being acted upon
- **operator**: RELOCATE (or NONE)
- **target_location**: categorical/sampled destination (e.g., `safe_region_beside_box`, `safe_region_far`, `still_on_lid`, `still_in_target`)
- **destination_type**: `CLEAR` (clears obstruction), `STILL_OBSTRUCTING` (hard negative), `IRRELEVANT` (distractor object)

### 4.2 Candidate Intervention Generation Per Scene

For each STOP scene with candidate objects O = {o_1, ..., o_N}:

**Correct interventions** (Δ_i = 1):
- `RELOCATE(culprit, safe_region)` — moves the actually-obstructing object to a location that clears the obstruction

**Irrelevant interventions** (Δ_i = 0):
- `RELOCATE(distractor, safe_region)` — moves a non-obstructing object
- `NONE` — no intervention (identity)

**Hard-negative interventions** (Δ_i = 0):
- `RELOCATE(culprit, still_obstructing_pose)` — moves culprit but insufficiently (still on lid / still in target)
- `RELOCATE(distractor, anywhere)` — moves a visually similar but non-causal object
- `RELOCATE(culprit, other_obstruction_pose)` — moves culprit from one obstructing pose to another

For PROCEED scenes: all candidate interventions yield Δ_i = 0 (feasibility already 1). Important control.

### 4.3 Record Schema

**Decision: One record per (scene, intervention)**. Rationale:
1. Enables standard DataLoader batching without ragged lists
2. Simplifies pairing logic for contrastive losses
3. Natural for intervention ranking: model scores individual interventions

```python
@dataclass
class InterventionRecord:
    # Scene identity
    scene_id: str
    pair_id: str
    task_id: str                    # "task_1" / "task_2"
    instruction: str
    query_action: str               # "OPEN(box)" / "PLACE(object1, target)"

    # Pre-intervention state
    pre_rgb_path: str
    pre_segmentation_path: str
    pre_feasible: bool              # F(s, a)
    pre_label: str                  # "STOP" / "PROCEED"

    # Candidate objects
    candidate_objects: List[CandidateObjectRecord]

    # Demonstration
    demonstration_id: str
    demonstration_path: str

    # Intervention specification
    intervention_id: str
    intervention_idx: int           # randomized index
    intervention_object: str        # body name
    intervention_operator: str      # "NONE" / "RELOCATE"
    intervention_target_location: str
    intervention_type: str          # "correct" / "irrelevant" / "hard_negative" / "none"
    intervention_parameters: dict   # destination pose, etc.

    # Post-intervention state
    post_rgb_path: str
    post_segmentation_path: str
    post_feasible: bool             # F(T(s, ρ_i), a)

    # Causal labels
    causal_effect: int              # Δ_i = post_feasible - pre_feasible ∈ {-1, 0, 1}
    is_culprit: bool
    culprit_object: Optional[str]
    culprit_relation: str           # "ON_TOP_OF" / "OCCUPIES" / "NONE"

    # Masks
    causal_mask_path: str
    candidate_mask_path: str
    target_mask_path: str
    object_masks: Dict[str, str]    # per-object mask paths

    # Provenance
    generation_seed: int
    intervention_seed: int
    split: str
    manifest_hash: str
```

### 4.4 Counterfactual Controls

1. **Same-RGB / different-task reversal**: Inherited from context_forcing_v1 design
2. **Matched intervention pairs**: For every correct intervention, an incorrect intervention on the same object type exists elsewhere
3. **Object identity balancing**: Each type appears as culprit and non-culprit
4. **Intervention-type balancing**: Correct/irrelevant/hard-negative balanced per split
5. **Candidate ordering randomization**: `intervention_idx` uncorrelated with type
6. **Target-location balancing**: Safe region locations varied
7. **No leakage between paired scenes**: All interventions from a scene → same split
8. **Culprit/non-culprit identity swapping**: Same object type is culprit in one scene, distractor in another

### 4.5 Expected Sample Counts

| Profile | Scenes | Interventions/scene | Total records | Purpose |
|---------|--------|--------------------|-|---------|
| Smoke | 8 | 4 | 32 | CI/CD testing |
| Pilot | 60 | 4–6 | 240–360 | Architecture validation |
| Full | 400+ | 4–6 | 1600–2400 | Paper results |

### 4.6 Growing Object Count

Strategy: **factored growth**
- For N=2,3,4 objects per scene, fixed scenes per (task, count, split) cell
- Interventions generated for ALL candidate objects → N interventions per scene
- Linear growth: doubling objects doubles interventions per scene, not scene count

---

## 5. Architecture V2 Design

### 5.1 Design Philosophy

Architecture V2 is the **minimal modification** of the current relational model enabling intervention scoring. Reuses existing cross-attention; adds intervention conditioning as additional input.

### 5.2 Approach Selection

| Option | Description | Selected? |
|--------|-------------|-----------|
| A: Predict culprit directly | scene → culprit classification | No — too weak |
| B: Score candidate interventions | (scene, action, interv) → Δ̂_i | Considered |
| **C: Predict post-intervention feasibility** | **(scene, action, interv) → F̂'** | **Yes** |
| D: Predict Δ directly | (scene, action, interv) → Δ̂ | Less info than C |
| E: Latent transition + feasibility | T_ψ(Z, ρ) → Z' → F̂' | Deferred to V3/V4 |

**Rationale for Option C**:
- Δ̂ = F̂' - F̂ is computed from output, not baked in
- Reuses existing feasibility classifier head
- Naturally supports PROCEED-scene controls (all interventions predict F=1)
- For NONE intervention, recovers standard feasibility prediction

### 5.3 V2 Tensor Shapes and Information Flow

```
INPUTS:
    text_feat:           (B, 384)            # MiniLM instruction embedding
    demo_global:         (B, K, 768)         # K=4 DINOv2 demo frame globals
    query_patch:         (B, 256, 768)       # 16×16 DINOv2 patch tokens
    intervention_feat:   (B, D_interv)       # intervention descriptor

INTERVENTION ENCODING:
    intervention descriptor = concat(
        object_visual_crop_feat,             # (768,) DINOv2 crop of intervened object
        intervention_type_embed,             # (64,) learned embed for NONE/RELOCATE
        destination_type_embed,              # (64,) learned embed for clear/still_obstructing
    )                                        # total D_interv = 896

PROJECTIONS:
    z_t   = proj_t(text_feat)               # (B, 256)
    z_d   = proj_v_global(demo_global)      # (B, K, 256)
    z_int = proj_interv(intervention_feat)  # (B, 256)

TEMPORAL ENCODER (self-attention over context tokens):
    seq = [z_t, z_d_1, ..., z_d_K, z_int]  # (B, 1+K+1, 256)
    Z_S = temporal_encoder(seq)             # (B, 1+K+1, 256)
    z_S = mean_pool(Z_S)                    # (B, 256)

CROSS-ATTENTION (query patches attend to context):
    Z_Q = proj_v_patch(query_patch)         # (B, 256, 256)
    Z_R = cross_attention(Z_Q, Z_S)         # (B, 256, 256)
    z_R = mean_pool(Z_R)                    # (B, 256)

OUTPUTS:
    logit_post = classifier(z_R)            # (B, 1) — predicted post-intervention feasibility
    s = cosine_sim(z_S_norm, z_R_norm)      # (B,) — ranking score
    Z_R                                     # (B, 256, 256) — for heatmap decoder
```

**Key change**: intervention descriptor token appended to temporal context sequence. Cross-attention naturally conditions query reasoning on the intervention.

For pre-intervention (no-intervention) pass: intervention = NONE with zero object crop.

### 5.4 Training Protocol

Each batch contains:
- Pre-intervention samples (intervention=NONE) with label = pre_feasible
- Intervention samples with label = post_feasible

Predicted effect: `Δ̂_i = σ(logit_post(intervention_i)) - σ(logit_post(NONE))`

### 5.5 Compatibility with Baselines

- **B0 Query-only**: No intervention, no context → unchanged
- **B1 Pooled multimodal**: Extended with intervention concatenation
- **B2 Relational**: Extended as above
- **B3 Feasibility-only**: Same architecture, no intervention loss
- **B4 Direct culprit**: Culprit head from scene features, no interventions

---

## 6. Architecture V3 Design

### 6.1 Object-Centric Architecture

V3 replaces patch-token reasoning with object-token reasoning. Likely paper architecture.

### 6.2 Object Tokens

```
z_oi = concat(
    visual_crop_feat,       # (768,) DINOv2 crop of object bounding box
    geometry_feat,          # (16,)  bbox size/aspect, image position
    relative_pose_feat,     # (16,)  position relative to target/lid
    category_embed          # (32,)  learned object type embedding (optional)
)                           # total: 832
z_oi_proj = proj_obj(z_oi)  # (256,)
```

Scene: `O = {z_o1, ..., z_oN}` — variable N.

### 6.3 Action/Plan Tokens

Single action:
```
z_a = proj_action(concat(
    action_type_embed,       # (64,) OPEN / PLACE
    action_target_embed,     # (64,) box / target_region
    text_feat                # (384,) instruction
))                           # → (256,)
```

Residual plan (M9):
```
Z_pi = [z_a_t, ..., z_a_H]
Z_pi = plan_encoder(Z_pi)   # (H, 256)
```

### 6.4 Demonstration Trajectory Tokens (M8)

```
d_t = concat(
    visual_embed,                # (768,) DINOv2 frame global
    ee_pos_target_relative,      # (3,) EE position in target frame
    ee_orn_target_relative,      # (6,) rotation 6D representation
    gripper_state                # (1,) open/closed
)                                # → (778,)
z_dt = proj_demo(d_t)            # (256,)
```

### 6.5 Relational Encoder

```
tokens = concat(
    [z_a],                       # 1 action token
    [z_d1, ..., z_dK],          # K demo tokens
    [z_o1, ..., z_oN],          # N object tokens
)                                # (1 + K + N, 256)

type_embed: action=0, demo=1, object=2
position_embed: learned per-slot for action/demo, shared for objects

Z = relational_encoder(tokens)   # (1+K+N, 256) — 4-layer pre-norm transformer

z_action = Z[0]
z_objects = Z[1+K:]              # per-object representations
```

### 6.6 Feasibility Head

```
z_global = mean_pool(z_objects)   # (256,)
logit = clf_head(concat(z_action, z_global))  # (1,)
```

### 6.7 Per-Object Causal Relevance

```
for each z_tilde_oi:
    r_i = relevance_head(concat(z_tilde_oi, z_action))  # (1,)
```

### 6.8 Intervention Mechanism

```
z_rho = concat(z_tilde_oi, interv_type_embed, destination_embed)
z_rho_proj = proj_intervention(z_rho)  # (256,)

z_objects_prime = z_objects.clone()
z_objects_prime[i] = transition_net(z_tilde_oi, z_rho_proj)

z_global_prime = mean_pool(z_objects_prime)
logit_post = clf_head(concat(z_action, z_global_prime))
```

### 6.9 Variable Object Count

Standard transformer masking: pad to max_objects with zeros; attention mask excludes padding; mean pool over real tokens only.

---

## 7. Loss Design

### 7.1 Feasibility Loss (L_feas)

```
L_feas = BCE(σ(logit), y_feasible)
```
Applied to both pre-intervention (NONE) and post-intervention samples.

### 7.2 Intervention Ranking Loss (L_rank_interv)

```
For matched (correct, incorrect) interventions from same scene:
    Δ̂_correct = σ(logit_post_correct) - σ(logit_pre)
    Δ̂_incorrect = σ(logit_post_incorrect) - σ(logit_pre)
    L_rank_interv = MarginRankingLoss(Δ̂_correct, Δ̂_incorrect, margin=0.3)
```

### 7.3 Causal Relevance Loss (L_causal) — V3 only

```
L_causal = BCE(σ(r_i), 1[Δ_i > 0])
```
Per-object relevance supervision from GT intervention effects.

### 7.4 Counterfactual Effect Loss (L_effect)

```
L_effect = MSE(Δ̂_i, Δ_i)
```

### 7.5 Latent Transition Consistency (L_transition) — V3 only

```
L_transition = MSE(transition_net(z_oi, z_rho), encoder(post_scene).objects[i].detach())
```
Applied only to intervened object's token.

### 7.6 Irrelevant Intervention Invariance (L_invariance) — V3 only

```
For irrelevant intervention j (GT Δ_j = 0):
    L_invariance = MSE(σ(logit_post_j), σ(logit_pre))
```

### 7.7 Heatmap Loss (L_heat) — V2 only

```
L_heat = BCE(heatmap, causal_mask) + DiceLoss(heatmap, causal_mask)
```

### 7.8 Total Loss (V2)

```
L = λ_feas·L_feas + λ_rank·L_rank_interv + λ_effect·L_effect + λ_heat·L_heat
```
Recommended: λ_feas=1.0, λ_rank=0.3, λ_effect=0.5, λ_heat=0.3

### 7.9 Total Loss (V3)

```
L = λ_feas·L_feas + λ_rank·L_rank_interv + λ_effect·L_effect
  + λ_causal·L_causal + λ_transition·L_transition + λ_invariance·L_invariance
```

---

## 8. Data / Model Interfaces

### 8.1 Action

```python
@dataclass
class Action:
    action_type: str          # "OPEN" / "PLACE"
    target: str               # "box" / "target_region"
    arguments: Dict[str, str] # {"object": "object1"}
```

### 8.2 CandidateObject

```python
@dataclass
class CandidateObject:
    name: str                 # MuJoCo body name
    object_type: str          # "coffee_can", etc.
    role: str                 # "blocker"/"occupant"/"distractor"
    position: List[float]
    orientation: List[float]
    is_culprit: bool
    mask_path: str
```

### 8.3 Intervention

```python
@dataclass
class Intervention:
    intervention_id: str
    operator: str             # "NONE" / "RELOCATE"
    object_name: str
    destination: str          # "safe_region" / "still_on_lid"
    destination_pose: List[float]
    intervention_type: str    # "correct"/"irrelevant"/"hard_negative"/"none"
```

### 8.4 InterventionOutcome

```python
@dataclass
class InterventionOutcome:
    intervention: Intervention
    post_rgb_path: str
    post_segmentation_path: str
    post_feasible: bool
    causal_effect: int        # Δ = post_feasible - pre_feasible
    validation_details: Dict
```

### 8.5 Model Batch (V2)

```python
@dataclass
class InterventionBatch:
    text_feat: Tensor         # (B, 384)
    demo_global: Tensor       # (B, K, 768)
    query_patch: Tensor       # (B, 256, 768)
    query_global: Tensor      # (B, 768)
    interv_object_crop: Tensor  # (B, 768)
    interv_type_idx: Tensor   # (B,) long
    interv_dest_idx: Tensor   # (B,) long
    pre_feasible: Tensor      # (B,) float
    post_feasible: Tensor     # (B,) float
    causal_effect: Tensor     # (B,) float
    scene_ids: List[str]
    intervention_ids: List[str]
    intervention_types: List[str]
    pair_ids: List[str]
    task_ids: List[str]
```

### 8.6 Prediction Output (V2)

```python
@dataclass
class InterventionPrediction:
    logit_pre: Tensor         # (B, 1)
    logit_post: Tensor        # (B, 1)
    delta_hat: Tensor         # (B,)
    ranking_score: Tensor     # (B,)
    Z_R: Optional[Tensor]
    heatmap: Optional[Tensor]
```

---

## 9. Repository Change Plan

### 9.1 New Files

| Path | Purpose | Key classes/functions |
|------|---------|----------------------|
| `src/interventions/__init__.py` | Package | — |
| `src/interventions/intervention_types.py` | Dataclasses | `Action`, `Intervention`, `InterventionOutcome`, `CandidateObject` |
| `src/interventions/intervention_generator.py` | Generate interventions per scene | `InterventionGenerator.generate_candidates()` |
| `src/interventions/intervention_validator.py` | Validate via simulator | `validate_intervention_effect()` |
| `src/interventions/intervention_scene_generator.py` | Full pipeline | `InterventionSceneGenerator` |
| `src/learning/intervention_dataset.py` | PyTorch dataset | `InterventionDataset`, `InterventionBatchSampler` |
| `src/learning/intervention_losses.py` | Loss functions | `InterventionLoss` |
| `src/learning/intervention_metrics.py` | Metrics | `compute_intervention_metrics()` |
| `src/learning/models/intervention_relational.py` | V2 | `InterventionConditionedRelationalModel` |
| `src/learning/models/intervention_object_centric.py` | V3 (later) | `ObjectCentricCausalModel` |
| `src/learning/models/model_registry.py` | Model dispatch | `get_model(config)` |
| `src/evaluation/__init__.py` | Package | — |
| `src/evaluation/intervention_evaluator.py` | Evaluation | `InterventionEvaluator` |
| `src/evaluation/representation_probes.py` | Probes | `LinearProbe`, `run_probes()` |
| `scripts/generate_intervention_dataset.py` | CLI generation | `main()` |
| `scripts/train_intervention_model.py` | CLI training | `main()` |
| `scripts/evaluate_intervention_model.py` | CLI evaluation | `main()` |
| `scripts/validate_intervention_dataset.py` | CLI validation | `main()` |
| `scripts/run_intervention_smoke.sh` | E2E smoke | — |
| `configs/intervention/smoke.yaml` | Smoke config | — |
| `configs/intervention/pilot.yaml` | Pilot config | — |
| `configs/intervention/learning/*.yaml` | Training configs | — |
| `tests/test_interventions.py` | Unit tests | — |
| `tests/test_intervention_dataset.py` | Dataset tests | — |
| `tests/test_intervention_models.py` | Model tests | — |

### 9.2 Modified Files

| Path | Modification |
|------|-------------|
| `src/validation/occupancy_checks.py` | Add `check_action_feasibility()` dispatcher |
| `src/learning/metrics.py` | Add AUROC/AUPRC |
| `scripts/precompute_features.py` | Add `--intervention-manifest` mode |

### 9.3 Config Structure

```
configs/intervention/
├── smoke.yaml
├── pilot.yaml
├── full.yaml
└── learning/
    ├── intervention_relational.yaml
    ├── intervention_query_only.yaml
    ├── intervention_feasibility_only.yaml
    ├── intervention_direct_culprit.yaml
    └── intervention_object_centric.yaml    # later
```

### 9.4 Artifact Directories

```
artifacts/intervention_v1/
├── smoke_report.json
├── pilot_report.json
├── dataset_validation.json
├── oracle_verification.json
├── training_results/
│   ├── {model}_seed{s}/
│   └── aggregate_metrics.json
└── evaluation_results/
```

---

## 10. Experiment Plan

### Exp 0: Oracle Intervention Verification
- **Hypothesis**: Simulator-validated interventions produce correct Δ_i labels at 100%.
- **Inputs**: Smoke intervention dataset (32 records)
- **Model**: None (simulator)
- **Metric**: Agreement rate
- **Expected**: 100%
- **Failure**: Generation bug → fix
- **Next**: Exp 1

### Exp 1: Query-Only Shortcut Baseline
- **Hypothesis**: Query-only cannot rank interventions above chance.
- **Model**: QueryOnlyBaseline
- **Metric**: Intervention top-1 accuracy
- **Expected**: ≤ 1/N + ε (≤ 35% for N=4)
- **Failure**: Dataset has shortcuts → add controls
- **Next**: Exp 2

### Exp 2: Feasibility-Only Baseline (B3)
- **Hypothesis**: Feasibility-only model cannot rank interventions well.
- **Model**: Relational model, L_feas only, no intervention input
- **Metric**: Intervention top-1 accuracy
- **Expected**: Moderate (above chance, below V2)
- **Next**: Exp 3

### Exp 3: Direct Culprit Classification (B4)
- **Hypothesis**: Direct culprit without intervention supervision is inferior.
- **Model**: Relational + culprit head, no intervention conditioning
- **Metric**: Culprit top-1, CFR via oracle on predicted culprit
- **Expected**: Reasonable culprit accuracy, lower CFR than V2

### Exp 4: Intervention-Conditioned Relational V2
- **Hypothesis**: Intervention conditioning enables better ranking and CFR.
- **Model**: V2
- **Metric**: Top-1, CFR, culprit top-1, AUROC
- **Expected**: Top-1 > 80% ID; CFR > 70%
- **Failure**: Check gradient flow through intervention tokens

### Exp 5: Irrelevance Invariance
- **Hypothesis**: Irrelevant interventions don't change predictions.
- **Model**: Trained V2
- **Metric**: Mean |Δ̂| for irrelevant; false-relevance rate
- **Expected**: |Δ̂| < 0.05; FRR < 5%

### Exp 6: Hard Negative Discrimination
- **Hypothesis**: Correct vs. hard-negative separated.
- **Metric**: Δ̂ separation
- **Expected**: Clear separation

### Exp 7: 5-Seed Ablation Study
- 10 ablations × 5 seeds (see Section 11)

### Exp 8: OOD Generalization
- **Splits**: unseen_object, compositional
- **Expected**: Graceful degradation

---

## 11. Ablation Matrix

| ID | What is removed | Expected effect |
|----|----------------|----------------|
| A0 | (Full V2) | Best |
| A1 | Intervention token zeroed | Falls to B3 |
| A2 | Text zeroed | Can't distinguish tasks |
| A3 | Demo zeroed | No visual context |
| A4 | λ_rank = 0 | Ranking degrades |
| A5 | λ_effect = 0 | Δ̂ calibration degrades |
| A6 | λ_heat = 0 | Heatmaps degrade |
| A7 | No hard negatives in data | Easier task |
| A8 | Object crop zeroed | Can't identify object |
| A9 | Type/dest embed zeroed | Can't reason about semantics |

---

## 12. Generalization Matrix

| Axis | ID | OOD |
|------|------|-----|
| Object identity | coffee_can, sugar_box, mug | cup, bowl |
| Background | bg_neutral_wood | bg_blue_counter, bg_granite_dark |
| Object count | 1 blocker | 2+, 3+ |
| Composition | Seen factor tuples | Novel combinations |
| Intervention dest | Seen safe regions | Novel poses |

---

## 13. Metric Definitions

### Intervention Top-1 Accuracy
```
For each STOP scene s: î = argmax_i Δ̂_i
Top1 = (1/|S_stop|) Σ 1[Δ_{î} = 1]
```

### Counterfactual Repair Rate (CFR)
```
CFR = P(F(T(s, ρ̂), a) = 1 | F(s, a) = 0)
where ρ̂ = argmax_i Δ̂_i, evaluated in simulator
```

### Culprit Top-1 Accuracy
```
Predicted culprit = argmax_i Δ̂_i (or argmax_i r_i)
CulpritTop1 = (1/|S_stop|) Σ 1[pred_culprit == GT_culprit]
```

### False Relevance Rate
```
FRR = (1/|I_irr|) Σ 1[|Δ̂_j| > τ] for j ∈ irrelevant interventions
```

### Effect Separation (AUC)
```
AUC of classifying interventions as relevant vs irrelevant using |Δ̂_i|
```

---

## 14. Test Plan

### Unit Tests

1. `test_correct_intervention_flips_feasibility`
2. `test_irrelevant_preserves_infeasibility`
3. `test_hard_negative_preserves_infeasibility`
4. `test_none_preserves_state`
5. `test_intervention_scene_deterministic`
6. `test_only_declared_object_changes`
7. `test_causal_effect_matches_simulator`
8. `test_post_rgb_matches_state`
9. `test_no_pair_leakage_across_splits`
10. `test_ordering_randomized`
11. `test_ordering_not_leaked_to_labels`
12. `test_feature_files_exist`
13. `test_label_balance`
14. `test_v2_forward_shapes`
15. `test_none_recovers_baseline`
16. `test_ablation_removes_modality`
17. `test_gradient_through_intervention`

### Integration Tests

18. `test_smoke_end_to_end`
19. `test_oracle_cfr`
20. `test_query_only_shortcut_bound`

### Scientific Tests

21. `test_same_rgb_different_task`
22. `test_culprit_matches_relation`
23. `test_no_cache_contamination`
24. `test_hard_negative_valid`
25. `test_intervention_frequency_balance`

---

## 15. Compute Plan

| Task | Relative cost | Bottleneck | Smoke-testable? |
|------|--------------|-----------|-----------------|
| Intervention scene gen (60 scenes) | LOW | CPU | Yes (8) |
| Feature extraction (~720 images) | LOW-MED | GPU | Yes (32) |
| Single training (50 epochs, pilot) | LOW | GPU | Yes (tiny) |
| 5-seed runs | MEDIUM | GPU | Yes (1 seed) |
| Full ablation (10×5) | MED-HIGH | GPU | Yes (2×1) |
| Full dataset (400+ scenes) | MEDIUM | CPU | Yes (smoke) |
| Full paper suite | HIGH | GPU | — |

Hardware: RTX 5090 32GB, 48 CPU, 125GB RAM — more than sufficient.

---

## 16. Checkpoint / Go-No-Go Gates

### G1 (M1): 100% Δ_i agreement + all tests pass
### G2 (M2): Query-only ≤ 1/N + 0.05
### G3 (M3): V2 top-1 > 80%, CFR > 70%, significantly > B3/B4
### G4 (M5): Model-predicted intervention → simulator CFR > 70%
### G5 (M6): False-relevance rate < 5%
### G6 (M7): V3 ≥ V2 on unseen-object split

---

## 17. Paper-Facing Experiment Story

### Figures
| # | Content |
|---|---------|
| 1 | Architecture diagram |
| 2 | Intervention pair examples (correct/hard-neg/irrelevant) |
| 3 | Latent sensitivity: per-object Δ̂ visualization |
| 4 | Qualitative intervention ranking results |
| 5 | Representation probes |
| 6 | OOD generalization |
| 7 | (Optional) Discovery gating |

### Tables
| # | Content |
|---|---------|
| 1 | Main results: all model variants |
| 2 | Ablation matrix |
| 3 | OOD generalization |
| 4 | Irrelevant intervention invariance |
| 5 | (Optional) TAMP integration |

---

## 18. Failure Modes

1. **Visual shortcut in intervention data** — post-intervention images have systematic bias
2. **Trivial intervention discrimination** — model uses object-type frequency heuristic
3. **Leakage through pair structure** — model memorizes pair associations
4. **Feasibility shortcut** — model ignores intervention tokens
5. **Object-position heuristic** — "on lid → culprit" without relational understanding
6. **Intervention-type leakage** — intervention encoding reveals type
7. **Latent collapse** — object tokens converge
8. **Simulator overfitting** — CFR high only because evaluator = data generator
9. **Causal overclaiming** — Δ_i is operational, not Pearl-style
10. **Publication bias** — reporting only positive results

---

## 19. Implementation Order for Gemini 3.1 Pro

| Phase | Objective | Prerequisite | Files | Tests | Output |
|-------|-----------|-------------|-------|-------|--------|
| P1 | Intervention types + generator | None | `intervention_types.py`, `intervention_generator.py` | 1-4 | Candidate interventions generated |
| P2 | Intervention validator | P1 | `intervention_validator.py` | 5-8 | Effects validated |
| P3 | Scene generator pipeline | P1,P2 | `intervention_scene_generator.py` | smoke | Records + images |
| P4 | Smoke generation + validation | P3 | `generate_intervention_dataset.py`, configs | smoke pass | 32 validated records |
| P5 | Dataset class + features | P4 | `intervention_dataset.py`, precompute | 9-13 | Loadable dataset |
| P6 | V2 architecture | P5 | `intervention_relational.py`, `model_registry.py` | 14-17 | Forward pass correct |
| P7 | Losses + metrics | P6 | `intervention_losses.py`, `intervention_metrics.py` | — | Loss computes |
| P8 | Training script + smoke train | P5-P7 | `train_intervention_model.py` | convergence | Model checkpoint |
| P9 | Evaluation + oracle verify | P8 | `evaluate_intervention_model.py` | 18-20 | Oracle CFR verified |
| P10 | Pilot dataset | P4 | pilot config | validation | ~360 records |
| P11 | Multi-seed + baselines | P8,P10 | — | — | Exp 1-4 results |
| P12 | Ablation studies | P11 | — | — | Ablation table |
| P13 | Probes + analysis | P11 | `representation_probes.py` | — | Probe results |

---

## 20. Master TODO

```
Phase 0: Preparation
  [ ] Create feature branch: feature/intervention-v1
  [ ] Finalize this plan document
  [ ] Create companion doc: CAUSAL_INTERVENTION_DATASET_SPEC.md
  [ ] Create companion doc: CAUSAL_INTERVENTION_ARCHITECTURE.md
  [ ] Create companion doc: EXPERIMENT_ROADMAP.md
  [ ] Create companion doc: GEMINI_IMPLEMENTATION_PHASES.md

Phase 1: Intervention Infrastructure (M1)
  [ ] src/interventions/__init__.py
  [ ] src/interventions/intervention_types.py
  [ ] src/interventions/intervention_generator.py
  [ ] src/interventions/intervention_validator.py
  [ ] tests/test_interventions.py (tests 1-8)
  [ ] Gate G1: All tests pass

Phase 2: Scene Generation Pipeline (M1)
  [ ] src/interventions/intervention_scene_generator.py
  [ ] configs/intervention/smoke.yaml
  [ ] scripts/generate_intervention_dataset.py
  [ ] scripts/validate_intervention_dataset.py
  [ ] scripts/run_intervention_smoke.sh
  [ ] Gate: Smoke dataset validation passes

Phase 3: Learning Infrastructure (M2-M3)
  [ ] src/learning/intervention_dataset.py
  [ ] src/learning/intervention_losses.py
  [ ] src/learning/intervention_metrics.py
  [ ] src/learning/models/model_registry.py
  [ ] src/learning/models/intervention_relational.py
  [ ] tests/test_intervention_dataset.py (tests 9-13)
  [ ] tests/test_intervention_models.py (tests 14-17)
  [ ] Gate: Shapes correct, dataset loads

Phase 4: Training and Evaluation (M2-M3)
  [ ] scripts/train_intervention_model.py
  [ ] scripts/evaluate_intervention_model.py
  [ ] configs/intervention/learning/*.yaml
  [ ] Gate: Smoke training converges; oracle CFR verified

Phase 5: Pilot Experiments (M2-M5)
  [ ] configs/intervention/pilot.yaml
  [ ] Generate pilot dataset
  [ ] Experiments 1-6
  [ ] Gate G3: V2 top-1 > 80%, CFR > 70%

Phase 6: Multi-Seed + Ablations (M3-M6)
  [ ] 5-seed runs for V2 + baselines
  [ ] Full ablation matrix
  [ ] OOD evaluation
  [ ] Representation probes
  [ ] Gates G4, G5

Phase 7: Object-Centric (M7)
  [ ] src/learning/models/intervention_object_centric.py
  [ ] Object crop features
  [ ] V2 vs V3 comparison
  [ ] Gate G6

Phase 8-12: Later milestones
  [ ] M8: Trajectory conditioning
  [ ] M9: Residual plan
  [ ] M10: Discovery gating
  [ ] M11: TAMP integration
  [ ] M12: Experience reuse
```

---

*This document was generated as a planning artifact. No source code was modified.*
