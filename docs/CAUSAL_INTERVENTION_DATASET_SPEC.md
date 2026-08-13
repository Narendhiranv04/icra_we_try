# Causal Intervention Dataset Specification

> Companion to the master causal intervention research plan.
> Details the ontology, records, structure, and generation strategy for the intervention dataset.

---

## 1. Motivation and Framing

The existing Benchmark A and Benchmark B datasets evaluate *whether* a scene is feasible for an action. The Causal Intervention dataset evaluates *why* a scene is infeasible by proposing minimal candidate interventions and verifying their causal effect.

Instead of just labeling a scene `STOP=1`, we generate a set of candidate interventions $\rho_i$ (e.g., "move object X out of the way") and evaluate whether the post-intervention state $T(s, \rho_i)$ becomes feasible for the action $a$ under relational precondition feasibility $F_R$.

The causal effect of intervention $i$ is:
$$\Delta_i = F_R(T(s, \rho_i), a) - F_R(s, a) \in \{-1, 0, +1\}$$

Interpretation:
- $\Delta_i = +1$: **Repair** (STOP $\to$ PROCEED)
- $\Delta_i = 0$: **Unchanged** (STOP $\to$ STOP or PROCEED $\to$ PROCEED)
- $\Delta_i = -1$: **Harmful** (PROCEED $\to$ STOP)

Authoritative ground truth is established exclusively via simulator execution and physical predicate checks ($F_R$). Semantic category names (`repair`, `hard_negative`, `irrelevant`, `identity`, `harmful`) are strictly `PRIVILEGED_GT_ONLY` metadata.

---

## 2. Intervention Ontology

### 2.1 Objects and Candidates
A scene contains candidate objects $O = \{o_1, \dots, o_N\}$.
- **Culprit**: The single object physically violating the precondition (e.g., resting on the lid in Task 1, or occupying the target in Task 2). For early V2/V3 experiments, each infeasible scene contains **exactly ONE causal culprit** and $N$ distractors.
- **Distractor**: An object present in the scene that does not violate the precondition.

### 2.2 Operators
For early tasks, we restrict interventions to simple physical relocations:
- `RELOCATE(object_name, destination_pose)`
- `NONE` (Identity intervention, control)

### 2.3 Destinations
Destinations are sampled from semantically defined regions in the simulator coordinate frame:
- Clear area on the table (safe region outside the target/lid footprint).
- Obstruction region (on the lid / inside the target region).

---

## 3. Candidate Generation Strategy

For a given scene, the dataset generator produces a set of candidate interventions designed to test specific causal and spatial reasoning capabilities:

| Semantic Category (`PRIVILEGED_GT_ONLY`) | Target Object | Destination Geometry | Pre $F_R$ | Post $F_R$ | Ground Truth $\Delta_i$ | Purpose |
|---|---|---|---|---|---|---|
| **Repair / Correct** | Culprit | Safe region | 0 (STOP) | 1 (PROCEED) | **+1** | Positive reparative example. Tests identification of culprit and valid repair. |
| **Hard Negative** | Culprit | Obstructing pose | 0 (STOP) | 0 (STOP) | **0** | Tests understanding of *spatial relation*, not just object identity. |
| **Irrelevant** | Distractor | Safe region | 0 (STOP) or 1 (PROCEED) | Unchanged | **0** | Tests invariance to non-causal objects. |
| **Identity (None)** | N/A | N/A | Any | Unchanged | **0** | Control. Pre-state equals post-state. |
| **Harmful** | Distractor | Obstructing pose | 1 (PROCEED) | 0 (STOP) | **-1** | Negative intervention on feasible scene. Tests detection of newly introduced obstructions. |

### Generation Rules:
1. Every STOP scene (1 culprit + $N$ distractors) generates:
   - 1 Repair intervention ($\Delta = +1$)
   - 1 Hard Negative intervention ($\Delta = 0$)
   - $N$ Irrelevant interventions (1 per distractor, $\Delta = 0$)
   - 1 Identity intervention ($\Delta = 0$)
   - *Total for 1 culprit + 1 distractor = 4 candidate records*.
2. Every PROCEED scene ($N$ distractors) generates:
   - 1 Harmful intervention ($\Delta = -1$)
   - $N$ Irrelevant interventions (1 per distractor, $\Delta = 0$)
   - 1 Identity intervention ($\Delta = 0$)
   - *Total for 1 distractor = 3 candidate records*.

---

## 4. Record Schema & API Partitioning

The dataset is stored as JSON Lines (`.jsonl`), where **each line is a single (scene, candidate intervention) pair**. The schema enforces strict three-way partitioning:

```json
{
  /* ================================================================
     1. MODEL FORWARD INPUTS (Pre-intervention observations only)
     ================================================================ */
  "instruction": "Open the box.",
  "task_id": "task_1",
  "pre_rgb_path": "data/intervention_v1/scenes/scene_t1_001/pre_rgb.png",
  "pre_segmentation_path": "data/intervention_v1/scenes/scene_t1_001/pre_segmentation.png",
  "candidate_object_crop_path": "data/intervention_v1/features/crop_scene_t1_001_sugar_box.png",
  "current_geometry": {
    "relative_position": [0.02, -0.01, 0.08],
    "bounding_extent": [0.08, 0.05, 0.12],
    "orientation_quat": [1.0, 0.0, 0.0, 0.0]
  },
  "destination_geometry": {
    "relative_position": [0.25, 0.15, 0.00],
    "orientation_quat": [1.0, 0.0, 0.0, 0.0]
  },
  "intervention_operator": "RELOCATE",
  "demonstration_path": "data/demos/demo_t1_open.mp4",

  /* ================================================================
     2. SUPERVISION TARGETS (Used strictly in loss computation)
     ================================================================ */
  "pre_feasible": false,
  "post_feasible": true,
  "causal_effect": 1,

  /* ================================================================
     3. PRIVILEGED METADATA (Evaluation, Invariants, GT Oracle only)
     ================================================================ */
  "scene_id": "scene_t1_001",
  "pair_id": "pair_task_1_001",
  "query_action": "OPEN(box_B1)",
  "intervention_id": "scene_t1_001_inv_0",
  "intervention_idx": 2,
  "intervention_type": "correct",
  "destination_type": "safe_region",
  "is_culprit": true,
  "culprit_object": "sugar_box",
  "culprit_relation": "ON_TOP_OF",
  "candidate_objects": [
    {
      "name": "sugar_box",
      "object_type": "sugar_box",
      "role": "blocker",
      "position": [0.5, 0.1, 0.8],
      "orientation": [1, 0, 0, 0],
      "is_culprit": true,
      "mask_path": "data/intervention_v1/scenes/scene_t1_001/masks/sugar_box.png"
    },
    {
      "name": "mug",
      "object_type": "mug",
      "role": "distractor",
      "position": [0.3, 0.4, 0.7],
      "orientation": [1, 0, 0, 0],
      "is_culprit": false,
      "mask_path": "data/intervention_v1/scenes/scene_t1_001/masks/mug.png"
    }
  ],
  "post_rgb_path": "data/intervention_v1/scenes/scene_t1_001/post_inv_0_rgb.png",
  "post_segmentation_path": "data/intervention_v1/scenes/scene_t1_001/post_inv_0_segmentation.png",
  "causal_mask_path": "data/intervention_v1/scenes/scene_t1_001/causal_mask.png",
  "target_mask_path": "data/intervention_v1/scenes/scene_t1_001/target_mask.png",
  "generation_seed": 42,
  "intervention_seed": 101,
  "split": "id",
  "manifest_hash": "a1b2c3d4..."
}
```

> [!IMPORTANT]
> `post_rgb_path` and `post_segmentation_path` exist strictly for simulator ground-truth verification and qualitative visualization. The PyTorch `InterventionDataset` must NEVER expose post-state images or features to the forward inference path.

---

## 5. Directory Structure

```
data/intervention_v1/
├── manifests/
│   ├── intervention_manifest.jsonl
│   └── split_assignments.json
├── features/
│   ├── query_scene_t1_001_features.pt
│   ├── post_scene_t1_001_inv_0_features.pt      # (For GT validation/teacher only)
│   └── crop_scene_t1_001_sugar_box_features.pt  # DINOv2 crop of candidate object
└── scenes/
    ├── scene_t1_001/
    │   ├── pre_rgb.png
    │   ├── pre_segmentation.png
    │   ├── post_inv_0_rgb.png
    │   ├── post_inv_0_segmentation.png
    │   └── masks/
    │       ├── sugar_box.png
    │       ├── mug.png
    │       ├── causal_mask.png
    │       └── target_mask.png
    └── ...
```

---

## 6. Controls and Deconfounding

To prevent models from exploiting visual or positional shortcuts:
1. **Intervention Index Randomization**: `intervention_idx` is uniformly shuffled per scene so the correct repair does not appear at a fixed index.
2. **Object Role Swapping**: An object type (e.g., `coffee_can`) acting as culprit in scene A must appear as a distractor in scene B.
3. **Split Isolation**: All candidate records originating from the same base scene reside strictly within the same dataset split.
4. **Target Location Variance**: Coordinates in `destination_geometry` are sampled with continuous spatial jitter to prevent fixed-coordinate memorization.
5. **PROCEED Controls**: PROCEED scenes generate Harmful ($\Delta = -1$), Irrelevant ($\Delta = 0$), and Identity ($\Delta = 0$) interventions to evaluate sensitivity to introduced obstructions.

---

## 7. Scaling Object Counts

In early representation learning (V2/V3), scenes contain **exactly 1 causal culprit** and $N$ distractors:
- Clutter scaling evaluates scaling distractor count $N \in \{1, 2, 3, 4\}$.
- Total candidate records per STOP scene: $N + 3$ (linear scaling, avoiding combinatorial explosion).
- Conjunctions of multiple simultaneous blockers and minimal intervention set search are deferred to later milestones.
