# Causal Intervention Research Plan — Master Blueprint

> **Repository**: `/home/projects/long-horizon/infeasibiilty-latent`
> **Branch**: `feature/context-forcing-v1-diagnostic`
> **HEAD commit**: `35758b5` ("docs: finalize causal intervention plan and add demo scripts")
> **Audit date**: 2026-08-14
> **Auditor**: Senior Research-Engineering Planner (Packet 0)

---

## Table of Contents

1. [Current-State Audit](#1-current-state-audit)
2. [Freeze / Reuse Decisions](#2-freeze--reuse-decisions)
3. [Scientific Claim Ladder & Feasibility Hierarchy](#3-scientific-claim-ladder--feasibility-hierarchy)
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
| HEAD | `35758b5` |
| Commit message | "docs: finalize causal intervention plan and add demo scripts" |
| Working tree | Clean |
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

| Pilot | 152 (120 pairs + 32 controls) | Primary learning dataset | `data/` (generated) |
| Context Challenge (Benchmark B) | 80 (40 physical scenes × 2 tasks) | Same-RGB context-reversal test | `data/context_challenge/` |
| Context Forcing V1 | ~128 train + ~64 val (native splits) | Deconfounded forcing dataset | `data/context_forcing_v1/` |

### 1.4 Current Experimental Findings (Verified)

#### Milestone 1 — Representation Learning (COMPLETE)
- 30 runs: 6 model families × 5 seeds (11, 23, 42, 67, 101)
- **Key finding**: All models achieve 0.0 reversal accuracy on Benchmark B
- Query-only achieves high accuracy on Benchmark A → visual shortcuts (approaching ~0.75 visual ceiling on balanced A/B/C/D context sets)
- Relational model: 100% logical test accuracy, but shortcut-dependent

#### Milestone 2 — Context Challenge (COMPLETE)
- Benchmark B demonstrates shortcut exploitation definitively
- Models cannot flip predictions for identical RGB with swapped task context
- Strong evidence of shortcut-permissive data, not architectural inability

#### Context Forcing V1 (COMPLETE — current branch)
- **query_only**: 0.703 val accuracy (approaching visual-only ceiling of ~0.75 for forced-context task)
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

## 3. Scientific Claim Ladder & Feasibility Hierarchy

### 3.0 Feasibility Level Hierarchy
To prevent conflating symbolic preconditions with execution dynamics, we formally distinguish four levels:
- **$F_R(s, a)$ — Relational Precondition Feasibility**: Relational preconditions for action $a$ (e.g., lid is clear for `OPEN(box)`, target region is unoccupied for `PLACE(object, target)`).
- **$F_M(s, a, \theta)$ — Continuous Motion Feasibility**: Kinematic reachability, collision avoidance, and IK feasibility for execution parameter $\theta$.
- **$F_A(s, a) = F_R(s, a) \land F_M(s, a, \theta)$ — Executable Action Feasibility**: Conjunction of relational preconditions and valid motion plan.
- **$F_\pi(s, \pi_{\text{remaining}})$ — Residual-Plan Feasibility**: Multi-step downstream execution feasibility w.r.t. remaining plan prefix $\pi_{\text{remaining}}$.

The current research phase focuses strictly on learning **intervention-grounded relational feasibility $F_R$**. Motion ($F_M$) and residual-plan ($F_\pi$) conditioning are deferred to subsequent project phases.

Each milestone answers **one** scientific question:

### M0 — CONTEXT CONDITIONING (COMPLETE)
**Claim**: The relational cross-attention architecture can condition predictions on task/demo context when the dataset prevents visual shortcuts.
**Evidence**: Context Forcing V1 pilot — relational model achieves 1.000 val accuracy on deconfounded data.
**Status**: ✅ Verified at `35758b5`.

### M1 — INTERVENTION DATASET VALIDITY
**Question**: Can we generate rigorously validated candidate interventions and measure their ground-truth effect on feasibility?
**Claim**: For each base scene with 1 causal culprit and $N$ distractors (STOP) or $N$ distractors (PROCEED), simulator-validated interventions produce correct ground-truth $\Delta_i = F_R(T(s, \rho_i), a) - F_R(s, a) \in \{-1, 0, +1\}$ labels across all 5 semantic categories (repair=+1, hard-negative=0, irrelevant=0, identity=0, harmful=-1).
**Gate**: 100% agreement between simulator $F_R$ evaluations and labels across the smoke dataset (6 scenes, 22 records); all validation tests pass.

### M2 — SHORTCUT BASELINES (B0 & B0b)
**Question**: Can a visual-only or simple non-relational model solve candidate ranking without relational context?
**Claim**: A query-only baseline (B0) cannot rank candidate interventions (receives identical scene inputs for all candidates). A simple candidate-aware MLP baseline (B0b) cannot significantly exceed chance on candidate ranking when the dataset is deconfounded ($Top\text{-}1 \le 1/N + \epsilon$).
**Gate**: B0b intervention top-1 accuracy $\le 1/N_{\text{candidates}} + \epsilon$.

### M3 — INTERVENTION-GROUNDED PREDICTION (V2 vs. B3)
**Question**: Does candidate ranking supervision improve reparative action prediction over passive feasibility training alone?
**Claim**: The intervention-conditioned relational model trained with ranking loss (V2) significantly outperforms the feasibility-only ablation (B3, trained with $\lambda_{rank}=0$) and the non-relational candidate baseline (B0b) on Top-1 intervention accuracy and Counterfactual Repair Rate (CFR).
**Provisional Target**: Top-1 intervention accuracy > 80% on ID split; CFR > 70%.

### M4 — CAUSAL OBJECT/RELATION LOCALIZATION
**Question**: Can the model identify the responsible object and violated relation?
**Claim**: High culprit top-1 accuracy and relation accuracy, exceeding direct classification baselines (B4) lacking intervention supervision.
**Provisional Target**: Culprit top-1 > 85% on ID; demonstrably better than B4.

### M5 — CORRECTIVE REPAIR VALIDATION
**Question**: Does the model-selected intervention actually restore feasibility when executed in MuJoCo?
**Claim**: Oracle-selected interventions restore feasibility at $\ge 95\%$ rates; model-predicted interventions restore at rates significantly above random baselines.
**Provisional Target**: Oracle CFR $\ge 95\%$; model CFR > 70%.

### M6 — IRRELEVANT INTERVENTION INVARIANCE
**Question**: Can irrelevant interventions and identity controls be correctly ignored?
**Claim**: Feasibility prediction is invariant under irrelevant interventions and distractor manipulations ($|\hat{\Delta}_j| \approx 0$).
**Provisional Target**: False-relevance rate (FRR) < 5%; mean $|\hat{\Delta}|$ for irrelevant interventions < 0.05.

### M7 — OBJECT-CENTRIC REPRESENTATION (V3 CONCEPT)
**Question**: Does an object-factored model improve causal generalization over patch/global representations?
**Claim**: Object-centric relational re-encoding (V3) shows improved OOD generalization vs. patch-based V2.
**Provisional Target**: $\ge 5\%$ improvement on unseen-object split intervention accuracy.

### M8 — TRAJECTORY DEMONSTRATION CONDITIONING
**Question**: What does target-relative robot trajectory add beyond video-only demonstration features?
**Claim**: EE pose/gripper trajectory features improve feasibility prediction.

### M9 — RESIDUAL PLAN CONDITIONING
**Question**: Can the representation reason about feasibility w.r.t. a later action in a multi-step plan?
**Claim**: Plan-conditioned model correctly identifies objects affecting future subgoals.

### M10 — DISCOVERY GATING
**Question**: Can newly discovered objects be partitioned into irrelevant, locally repairable, and replanning-relevant?
**Claim**: Correct CONTINUE / LOCAL_REPAIR / GLOBAL_REPLAN classification.

### M11 — TAMP INTEGRATION
**Question**: Does the learned causal module reduce unnecessary VLM/planner calls?
**Claim**: Selective repair/replan reduces total planner calls vs. always-replan baseline.

### M12 — WITHIN-EPISODE EXPERIENCE REUSE
**Question**: Can successful earlier manipulations be retrieved and adapted for local repair?
**Claim**: Episodic experience reuse reduces motion-planning calls and repair latency.

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

In early representation learning (V2/V3), each infeasible scene contains **exactly ONE causal culprit** and $N$ distractors.

**STOP Scenes (1 culprit + $N$ distractors)**:
- **Repair** ($\Delta_i = +1$): `RELOCATE(culprit, safe_region)` — moves the single culprit to a clear area, restoring feasibility ($F_R \to 1$).
- **Hard Negative** ($\Delta_i = 0$): `RELOCATE(culprit, still_obstructing_pose)` — moves the culprit but leaves it in an obstructing configuration ($F_R \to 0$).
- **Irrelevant** ($\Delta_i = 0$): `RELOCATE(distractor_k, safe_region)` — moves a non-causal distractor ($F_R \to 0$). ($N$ such interventions).
- **Identity Control** ($\Delta_i = 0$): `NONE` — evaluates pre-intervention state ($F_R \to 0$).

**PROCEED Scenes ($N$ distractors)**:
- **Harmful** ($\Delta_i = -1$): `RELOCATE(distractor_k, obstructing_pose)` — moves a distractor into an obstructing pose, breaking feasibility ($F_R \to 0$).
- **Irrelevant** ($\Delta_i = 0$): `RELOCATE(distractor_k, safe_region)` — moves a distractor to another clear area ($F_R \to 1$). ($N$ such interventions).
- **Identity Control** ($\Delta_i = 0$): `NONE` — evaluates pre-intervention state ($F_R \to 1$).

All semantic category labels (`repair`, `hard_negative`, `irrelevant`, `identity`, `harmful`) and spatial designations are `PRIVILEGED_GT_ONLY` metadata.

### 4.3 Record Schema & API Partitioning

**Decision: One record per (scene, candidate intervention)**. Rationale:
1. Enables standard DataLoader batching without ragged lists
2. Simplifies grouping logic for contrastive ranking losses
3. Natural for scoring individual candidate interventions

The schema strictly separates model inputs, supervision targets, and privileged metadata:

```python
@dataclass
class InterventionRecord:
    # --- MODEL INPUT FIELDS (Pre-intervention observations only) ---
    task_id: str                    # "task_1" / "task_2"
    instruction: str                # e.g., "Open the box."
    pre_rgb_path: str
    pre_segmentation_path: str
    candidate_crop_path: str        # DINOv2 crop of candidate object
    current_geometry: Dict[str, Any]      # Bounding box & target-relative pose
    destination_geometry: Dict[str, Any]  # Proposed target-relative destination pose
    operator: str                   # "NONE" / "RELOCATE"
    demonstration_path: str

    # --- SUPERVISION TARGETS ---
    pre_feasible: bool              # F_R(s, a)
    post_feasible: bool             # F_R(T(s, rho_i), a)
    causal_effect: int              # Δ_i = post_feasible - pre_feasible in {-1, 0, +1}

    # --- PRIVILEGED METADATA / EVALUATION (GT Oracle only) ---
    scene_id: str
    pair_id: str
    query_action: str               # "OPEN(box_B1)" / "PLACE(object1, target)"
    intervention_id: str
    intervention_idx: int           # Uniformly randomized per scene
    intervention_type: str          # PRIVILEGED_GT_ONLY ("correct", "hard_negative", etc.)
    destination_type: str           # PRIVILEGED_GT_ONLY ("safe_region", "still_obstructing", etc.)
    is_culprit: bool                # PRIVILEGED_GT_ONLY
    culprit_object: str             # PRIVILEGED_GT_ONLY
    culprit_relation: str           # PRIVILEGED_GT_ONLY
    candidate_objects: List[Dict]   # PRIVILEGED_GT_ONLY
    post_rgb_path: str              # PRIVILEGED_GT_ONLY (validation/visualization only)
    post_segmentation_path: str     # PRIVILEGED_GT_ONLY
    causal_mask_path: str           # PRIVILEGED_GT_ONLY
    target_mask_path: str           # PRIVILEGED_GT_ONLY
    generation_seed: int
    intervention_seed: int
    split: str                      # "id", "unseen_object", etc.

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
    text_feat:              (B, 384)                     # MiniLM instruction embedding
    demo_global:            (B, K, 768)                  # K=4 DINOv2 demo frame globals
    query_patch:            (B, 256, 768)                # 16×16 DINOv2 pre-scene patch tokens
    interv_object_crop:     (B, D_crop)                  # D_crop=768 DINOv2 crop of candidate object
    interv_current_geom:    (B, D_current_geom)          # Candidate localization & target-relative current pose
    interv_dest_geom:       (B, D_dest_geom)             # Proposed target-relative destination pose
    interv_operator_idx:    (B,) long                    # 0=NONE, 1=RELOCATE

INTERVENTION ENCODING:
    operator_embed = embed_operator(interv_operator_idx) # (B, D_operator=64)
    z_rho_raw = concat(
        interv_object_crop,                              # (B, 768)
        interv_current_geom,                             # (B, D_current_geom)
        interv_dest_geom,                                # (B, D_dest_geom)
        operator_embed,                                  # (B, 64)
    )                                                    # D_interv = D_crop + D_current_geom + D_dest_geom + D_operator
                                                         # (e.g. 768 + 16 + 16 + 64 = 864 in default config)

PROJECTIONS:
    z_t   = proj_t(text_feat)                            # (B, 256)
    z_d   = proj_v_global(demo_global)                   # (B, K, 256)
    z_int = proj_interv(z_rho_raw)                       # (B, 256), proj_interv = nn.Linear(D_interv, 256)

TEMPORAL ENCODER (self-attention over context tokens):
    seq = [z_t, z_d_1, ..., z_d_K, z_int]               # (B, 1+K+1, 256)
    Z_S = temporal_encoder(seq)                          # (B, 1+K+1, 256)
    z_S = mean_pool(Z_S)                                 # (B, 256)

CROSS-ATTENTION (query patches attend to context):
    Z_Q = proj_v_patch(query_patch)                      # (B, 256, 256)
    Z_R = cross_attention(Z_Q, Z_S)                      # (B, 256, 256)
    z_R = mean_pool(Z_R)                                 # (B, 256)

OUTPUTS:
    logit_post = classifier(z_R)                         # (B, 1) — predicted post-intervention feasibility
    s = cosine_sim(z_S_norm, z_R_norm)                   # (B,) — ranking score
    Z_R                                                  # (B, 256, 256)
```

**Key property**: Neutral intervention descriptor token appended to temporal context sequence. Cross-attention conditions query patch reasoning on the candidate intervention. Pre-scene observations only; post-state images/features NEVER enter forward().

For pre-intervention (`NONE`) pass: `interv_object_crop`, `interv_current_geom`, and `interv_dest_geom` are zero tensors, and `interv_operator_idx` is `0` (`NONE`).

### 5.4 Training Protocol

Each batch contains candidate interventions scored for post-feasibility:
- `NONE` intervention samples predict $P(F_{pre}=1 \mid s_{pre}, a)$
- Relocation candidate samples predict $P(F_{post}=1 \mid s_{pre}, a, \rho_i)$
- Predicted effect: $\hat{\Delta}_i = P(F_{post}=1 \mid \rho_i) - P(F_{pre}=1 \mid \text{NONE})$

### 5.5 Compatibility with Baselines

- **B0 Query-only Feasibility**: Pre-scene only, no intervention descriptor $\to$ visual feasibility baseline (cannot rank candidates).
- **B0b Simple Intervention Baseline**: 3-layer MLP over `concat(scene_global, crop_feat, current_geom, dest_geom, operator_embed)` $\to$ post-feasibility. Weak candidate-ranking baseline without relational cross-attention.
- **B3 Feasibility-Only Relational**: Same architecture and inputs as V2, trained with $\lambda_{rank} = 0$. Direct ablation for candidate ranking loss.
- **B4 Direct Culprit**: Relational model with static `is_culprit` head, no intervention conditioning.
- **V2 Intervention Relational (Proposed)**: Full model with ranking supervision ($\lambda_{rank} > 0$).

---

## 6. Architecture V3 Design (Future Concept)

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
    [x_1, ..., x_N],            # N raw object tokens
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

### 6.8 Counterfactual Relational Mechanism (Intervention Reasoning)

For candidate relocation $\rho_i = \text{RELOCATE}(o_i, \text{dest\_geom}_i)$:
1. Construct the counterfactual raw object token using proposed destination geometry:
   $$x_i' = E_{obj}(\text{visual}_i, \text{destination\_geometry}_i)$$
2. Replace the raw entity token in the scene set:
   $$x_i \to x_i'$$
3. Rerun the relational encoder over ALL entities and context tokens:
   $$Z_{out}' = \mathcal{R}_\theta([z_a, z_{d1..K}, x_1, \dots, x_i', \dots, x_N])$$
4. Evaluate post-intervention feasibility from $Z_{out}'$.

**Rationale**: Moving object $i$ physically alters pairwise geometric and clearance relations with *all other scene entities*. Rather than freezing already-contextualized latents and mutating an isolated slot via a separate transition network, replacing the raw factored entity and re-running relational encoding naturally propagates global relational updates. (Learned latent transition networks may be reserved for contact-rich dynamics where post-state geometry cannot be analytically specified).

### 6.9 Variable Object Count

Standard transformer masking: pad to max_objects with zeros; attention mask excludes padding; mean pool over real tokens only.

---

## 7. Loss Design

### 7.1 Feasibility Loss (L_feas)

```
L_post_feas = BCEWithLogitsLoss(logit_post, y_post_feasible)
```
Applied to all candidate samples (recovering pre-feasibility for `NONE`).

### 7.2 Intervention Ranking Loss (L_rank_interv)

```
For matched (correct, incorrect) interventions from same scene:
    Δ̂_correct = σ(logit_post_correct) - σ(logit_none)
    Δ̂_incorrect = σ(logit_post_incorrect) - σ(logit_none)
    L_rank_interv = MarginRankingLoss(Δ̂_correct, Δ̂_incorrect, target=1, margin=0.3)
```

### 7.3 Canonical Loss (V2)

```
L_V2 = L_post_feas + λ_rank · L_rank_interv
```
Default: $\lambda_{feas}=1.0$, $\lambda_{rank}=0.3$ for V2; $\lambda_{rank}=0.0$ for B3.

Auxiliary losses (effect MSE, heatmap, direct culprit, transition, invariance) are marked as deferred/optional ablations and are not required in the canonical V2 model.

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
    intervention_type: str    # PRIVILEGED_GT_ONLY
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
    # --- MODEL INPUTS (Observed by forward pass) ---
    text_feat: Tensor               # (B, 384)
    demo_global: Tensor             # (B, K, 768)
    query_patch: Tensor             # (B, 256, 768)
    interv_object_crop: Tensor      # (B, 768)
    interv_current_geom: Tensor     # (B, D_current_geom)
    interv_dest_geom: Tensor        # (B, D_dest_geom)
    interv_operator_idx: Tensor     # (B,) long [0=NONE, 1=RELOCATE]

    # --- SUPERVISION TARGETS ---
    pre_feasible: Tensor            # (B,) float
    post_feasible: Tensor           # (B,) float
    causal_effect: Tensor           # (B,) float {-1, 0, +1}

    # --- PRIVILEGED METADATA (Evaluation, Invariants, GT Oracle only) ---
    scene_ids: List[str]
    intervention_ids: List[str]
    intervention_types: List[str]   # PRIVILEGED_GT_ONLY
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
- **Hypothesis**: Simulator-validated interventions produce correct $\Delta_i \in \{-1, 0, +1\}$ labels across all 5 semantic categories at 100%.
- **Inputs**: Smoke intervention dataset (6 scenes, 22 records)
- **Model**: None (MuJoCo simulator oracle)
- **Metric**: Simulator validation agreement rate
- **Expected**: 100%
- **Failure**: Generator/validator bug $\to$ fix before learning

### Exp 1: Non-Relational Shortcut Baselines (B0 & B0b)
- **Hypothesis**: Query-only (B0) cannot rank candidate interventions (pre-scene visual feature only). Simple candidate-aware MLP (B0b) cannot rank interventions significantly above chance without relational task/demo context.
- **Model**: `QueryOnlyBaseline` (B0), `SimpleInterventionMLP` (B0b)
- **Metric**: Intervention top-1 ranking accuracy
- **Expected**: B0b Top-1 $\le 1/N + \epsilon$ ($\le 35\%$ on 4-candidate sets)
- **Failure**: Shortcut in candidate crops/geometry $\to$ refine deconfounding

### Exp 2: Feasibility-Only Relational Baseline (B3)
- **Hypothesis**: Relational model trained solely on post-feasibility ($\lambda_{rank}=0$) acquires some candidate awareness but lacks the margin separation needed for optimal repair ranking.
- **Model**: `InterventionConditionedRelationalModel` with $\lambda_{rank} = 0$ (B3)
- **Metric**: Intervention top-1 accuracy, CFR
- **Expected**: Moderate ranking performance (exceeds B0b, below V2)

### Exp 3: Direct Culprit Classification (B4)
- **Hypothesis**: Direct culprit classification without candidate intervention grounding fails to generalize to complex scene geometry.
- **Model**: Relational model + static `is_culprit` auxiliary head
- **Metric**: Culprit top-1, CFR via oracle execution on predicted culprit
- **Expected**: Reasonable culprit accuracy, lower CFR than V2

### Exp 4: Intervention-Conditioned Relational Model (V2)
- **Hypothesis**: Intervention conditioning + grouped candidate-ranking supervision yields superior repair ranking, CFR, and OOD generalization.
- **Model**: V2 (`InterventionConditionedRelationalModel` with $\lambda_{rank} > 0$)
- **Metric**: Top-1 candidate accuracy, Counterfactual Repair Rate (CFR), False Relevance Rate (FRR)
- **Provisional Targets**: Top-1 > 80% ID; CFR > 70%

### Exp 5: Irrelevance Invariance
- **Hypothesis**: Irrelevant interventions and identity controls do not perturb feasibility predictions.
- **Model**: Trained V2
- **Metric**: Mean $|\hat{\Delta}|$ for irrelevant candidates; False-Relevance Rate (FRR)
- **Provisional Target**: Mean $|\hat{\Delta}| < 0.05$; FRR < 5%

### Exp 6: Hard Negative Discrimination
- **Hypothesis**: Model distinguishes true repair interventions ($\Delta=+1$) from hard negatives ($\Delta=0$, still obstructing).
- **Metric**: Margin separation between $\hat{\Delta}_{repair}$ and $\hat{\Delta}_{hard\_neg}$
- **Expected**: Statistically significant margin $> 0.2$

### Exp 7: 5-Seed Protocol & Ablations
- 5 seeds (11, 23, 42, 67, 101) $\times$ 10,000 bootstrap resamples across all ablations.

### Exp 8: OOD Generalization
- **Splits**: `unseen_object`, `compositional`, clutter scaling ($1 \text{ culprit} + N \text{ distractors}$)
- **Expected**: Graceful degradation; V2 maintains statistically significant lead over B0b and B3.

---

## 11. Ablation Matrix

| ID | What is removed | Expected effect |
|----|----------------|----------------|
| A0 | (Full V2) | Best |
| A1 | Intervention token zeroed | Falls to B0 pre-scene baseline |
| A2 | Text feature zeroed | Cannot distinguish Task 1 (`OPEN`) vs Task 2 (`PLACE`) |
| A3 | Demo features zeroed | Empirical exploration: assesses role of visual demo context in Task 1/2 |
| A4 | $\lambda_{rank} = 0$ (B3) | Isolates contribution of candidate ranking loss |
| A5 | Candidate crop zeroed | Cannot visually identify candidate object properties |
| A6 | Geometry features zeroed | Cannot reason about candidate spatial displacement / target relations |
| A7 | No hard negatives in training | Overfits to object identity heuristic; fails on hard negatives |

---

## 12. Generalization Matrix

| Axis | In-Distribution (ID) | Out-of-Distribution (OOD) |
|------|----------------------|---------------------------|
| Object Identity | `coffee_can`, `sugar_box`, `mug` | `cup`, `bowl` |
| Background / Lighting | `bg_neutral_wood`, standard lighting | `bg_blue_counter`, `bg_granite_dark`, randomized lighting |
| Clutter Scaling | 1 culprit + 1 distractor | 1 culprit + 3+ distractors (increasing distractor count) |
| Composition | Seen factor combinations | Novel attribute combinations |
| Intervention Destination | Canonical safe regions | Novel safe region poses |

---

## 13. Metric Definitions

### 13.1 Causal vs. Repair Relevance
- **Causal Relevance**: $y_i^{causal} = \mathbb{I}[|\Delta_i| > 0]$ (identifies any candidate that alters feasibility: repair $\Delta=+1$ or harmful $\Delta=-1$).
- **Repair Relevance**: $y_i^{repair} = \mathbb{I}[\Delta_i > 0]$ (identifies valid corrective interventions: repair $\Delta=+1$ only).

### 13.2 Intervention Top-1 Accuracy
For each STOP scene $s$, candidate ranking selects $\hat{i} = \arg\max_i \hat{\Delta}_i$:
$$\text{Top1} = \frac{1}{|S_{\text{stop}}|} \sum_{s \in S_{\text{stop}}} \mathbb{I}[\Delta_{\hat{i}} = 1]$$

### 13.3 Counterfactual Repair Rate (CFR)
$$\text{CFR} = P\left(F_R(T(s, \hat{\rho}), a) = 1 \mid F_R(s, a) = 0\right)$$
where $\hat{\rho} = \arg\max_i \hat{\Delta}_i$, evaluated in the MuJoCo simulator.

### 13.4 False Relevance Rate (FRR)
$$\text{FRR} = \frac{1}{|I_{\text{irr}}|} \sum_{j \in I_{\text{irr}}} \mathbb{I}[|\hat{\Delta}_j| > \tau] \quad (\tau = 0.1)$$

---

## 14. Test Plan

### Unit Tests
1. `test_generate_candidates_stop_count`
2. `test_generate_candidates_proceed_count`
3. `test_correct_intervention_safe_region`
4. `test_hard_negative_still_obstructs`
5. `test_none_intervention_is_identity`
6. `test_intervention_ids_unique`
7. `test_intervention_idx_randomized`
8. `test_correct_intervention_flips_feasibility`
9. `test_irrelevant_preserves_infeasibility`
10. `test_harmful_intervention_breaks_feasibility`
11. `test_hard_negative_preserves_infeasibility`
12. `test_none_preserves_state`
13. `test_only_declared_object_changes`
14. `test_v2_forward_shapes`
15. `test_v2_none_intervention`
16. `test_v2_gradient_through_intervention`
17. `test_no_post_state_leakage`

### Integration & Baseline Tests
18. `test_smoke_validation_passes` (6 base scenes, 22 records)
19. `test_oracle_cfr_upper_bound` ($\ge 95\%$)
20. `test_b0b_candidate_ranking_baseline` ($\le 1/N + \epsilon$)
21. `test_b3_feasibility_only_ablation`

---

## 15. Compute Plan

| Task | Relative cost | Bottleneck | Smoke-testable? |
|------|--------------|-----------|-----------------|
| Intervention scene gen (60 scenes) | LOW | CPU | Yes (6 scenes / 22 records) |
| Feature extraction (~720 images) | LOW-MED | GPU | Yes (22 records) |
| Single training (50 epochs, pilot) | LOW | GPU | Yes (smoke) |
| 5-seed runs | MEDIUM | GPU | Yes (1 seed) |
| Full ablation (8×5) | MED-HIGH | GPU | Yes (2×1) |
| Full dataset (400+ scenes) | MEDIUM | CPU | Yes (smoke) |

---

## 16. Checkpoint / Go-No-Go Gates (Provisional Engineering Targets)

### G1 (M1): 100% $\Delta_i$ agreement on smoke set (6 scenes, 22 records)
### G2 (M2): B0b candidate baseline $\le 1/N + 0.05$
### G3 (M3): V2 top-1 > 80%, CFR > 70%, significantly > B0b / B3
### G4 (M5): Model-predicted intervention $\to$ simulator CFR > 70%
### G5 (M6): False-relevance rate (FRR) < 5%
### G6 (M7): V3 $\ge$ V2 on unseen-object split

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

### Tables
| # | Content |
|---|---------|
| 1 | Main results: all model variants (B0, B0b, B3, B4, V2) |
| 2 | Ablation matrix (A0-A7) |
| 3 | OOD generalization (unseen object, clutter scaling) |
| 4 | Irrelevant intervention invariance |

---

## 18. Failure Modes

1. **Visual shortcut in intervention data** — post-intervention images have systematic bias (prevented by strict pre-scene forward input contract)
2. **Trivial intervention discrimination** — model uses object-type frequency heuristic (prevented by matched culprit/distractor role swapping)
3. **Leakage through pair structure** — model memorizes pair associations (prevented by scene-level split grouping)
4. **Feasibility shortcut** — model ignores intervention tokens (prevented by ranking loss $\lambda_{rank} > 0$)
5. **Simulator overfitting** — CFR evaluated on independently constructed validation environments

---

## 19. Implementation Order for Gemini 3.1 Pro

| Phase | Objective | Prerequisite | Files | Tests | Output |
|-------|-----------|-------------|-------|-------|--------|
| P1 | Intervention types + generator | Repo clean | `intervention_types.py`, `intervention_generator.py` | 1-7 | Candidate interventions generated |
| P2 | Intervention validator | P1 | `intervention_validator.py`, `occupancy_checks.py` | 8-13 | Effects validated against $F_R$ |
| P3 | Scene generator pipeline | P1, P2 | `intervention_scene_generator.py` | — | Pre/post RGB, masks, crops |
| P4 | Smoke generation + validation | P3 | `validate_intervention_dataset.py`, `run_intervention_smoke.sh` | 14-15 | 6 scenes / 22 records validated |
| P5 | Dataset class + features | P4 | `intervention_dataset.py`, `precompute_features.py` | 16-19 | Strict 3-way partitioned dataset |
| P6 | V2 architecture | P5 | `intervention_relational.py`, `model_registry.py` | 20-23 | Forward pass & gradients verified |
| P7 | Losses + metrics | P6 | `intervention_losses.py`, `intervention_metrics.py` | — | Minimal canonical loss computes |
| P8 | Training script + smoke train | P5-P7 | `train_intervention_model.py`, configs | 24-25 | Convergence on smoke data |
| P9 | Evaluation + oracle verify | P8 | `evaluate_intervention_model.py` | 26-28 | Oracle CFR verified $\ge 95\%$ |
| P10 | Pilot dataset | P4 | Pilot config | — | ~360 records generated |
| P11 | Multi-seed + baselines | P8, P10 | — | — | Exp 1-4 results (B0, B0b, B3, B4, V2) |
| P12 | Ablation studies | P11 | — | — | Ablation table (A0-A7) |
| P13 | Probes + analysis | P11 | `representation_probes.py` | — | Latent probe results |

---

## 20. Master TODO

```
Phase 0: Preparation
  [ ] Create feature branch: feature/causal-intervention-v2
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
