# Causal Intervention Dataset Specification

> Companion to the master causal intervention research plan.
> Details the ontology, records, structure, and generation strategy for the intervention dataset.

---

## 1. Motivation and Framing

The existing Benchmark A and Benchmark B datasets evaluate *whether* a scene is feasible for an action. The Causal Intervention dataset evaluates *why* a scene is infeasible by proposing minimal interventions and verifying their causal effect.

Instead of just labeling a scene `STOP=1`, we generate a set of candidate interventions $\rho_i$ (e.g., "move object X out of the way") and evaluate whether the post-intervention state $T(s, \rho_i)$ becomes feasible for the action $a$.

The causal effect of intervention $i$ is:
$\Delta_i = F(T(s, \rho_i), a) - F(s, a) \in \{0, 1\}$

(Note: we assume pre_feasible $F(s,a)=0$ for STOP scenes. For PROCEED scenes, $F=1$ so $\Delta=0$ for all interventions, which serves as a critical control.)

## 2. Intervention Ontology

### 2.1 Objects and Candidates
A scene contains a set of candidate objects $O = \{o_1, \dots, o_N\}$.
- **Culprit**: The object that is physically causing the infeasibility (e.g., blocking the lid).
- **Distractor**: An object that is present but not causing infeasibility.

### 2.2 Operators
For early tasks, we restrict interventions to simple physical relocations:
- `RELOCATE(object_name, destination_pose)`
- `NONE` (Identity intervention, control)

### 2.3 Destinations
Instead of arbitrary coordinates, destinations are sampled from semantically meaningful regions:
- `SAFE_REGION_1` / `SAFE_REGION_2`: A known clear area on the table.
- `STILL_OBSTRUCTING`: A different pose that still violates the precondition.
- `OTHER_OBSTRUCTION`: A pose that violates a *different* precondition (for later).

## 3. Candidate Generation Strategy

For a given STOP scene (with $N$ objects), the dataset generator produces a set of candidate interventions designed to test specific model reasoning capabilities:

| Intervention Type | Target Object | Destination | Ground Truth $\Delta_i$ | Purpose |
|-------------------|---------------|-------------|-------------------------|---------|
| **Correct** | Culprit | Safe region | 1 | Positive example. Tests if model identifies the correct object and a valid repair. |
| **Irrelevant** | Distractor | Safe region | 0 | Tests if model ignores objects that don't affect feasibility. |
| **Hard Negative** | Culprit | Still obstructing | 0 | Tests if model understands the *spatial* condition, not just the object identity. |
| **Identity (None)** | N/A | N/A | 0 | Control. Pre-state equals post-state. |

### Generation Rules:
1. Every STOP scene must have exactly 1 Correct intervention.
2. Every distractor object gets 1 Irrelevant intervention.
3. Every culprit object gets 1 Hard Negative intervention (if physically possible to sample).
4. Every scene gets 1 Identity intervention.

For a scene with 1 culprit and 1 distractor, this yields exactly 4 interventions.

## 4. Record Schema

The dataset is stored as JSON Lines (`.jsonl`), where **each line is a single (scene, intervention) pair**. 

```json
{
  "scene_id": "scene_t1_001",
  "pair_id": "pair_task_1_001",
  "task_id": "task_1",
  "instruction": "Open the box.",
  "query_action": "OPEN(box_B1)",
  
  "pre_feasible": false,
  "pre_rgb_path": "data/intervention_v1/scene_t1_001/pre_rgb.png",
  "pre_segmentation_path": "data/intervention_v1/scene_t1_001/pre_segmentation.png",
  
  "candidate_objects": [
    {
      "name": "sugar_box",
      "object_type": "sugar_box",
      "role": "blocker",
      "position": [0.5, 0.1, 0.8],
      "orientation": [1, 0, 0, 0],
      "is_culprit": true,
      "mask_path": "data/intervention_v1/scene_t1_001/masks/sugar_box.png"
    },
    {
      "name": "mug",
      "object_type": "mug",
      "role": "distractor",
      "position": [0.3, 0.4, 0.7],
      "orientation": [1, 0, 0, 0],
      "is_culprit": false,
      "mask_path": "data/intervention_v1/scene_t1_001/masks/mug.png"
    }
  ],
  
  "demonstration_id": "demo_t1_open",
  "demonstration_path": "data/demos/demo_t1_open.mp4",
  
  "intervention_id": "scene_t1_001_inv_0",
  "intervention_idx": 2, 
  "intervention_object": "sugar_box",
  "intervention_operator": "RELOCATE",
  "intervention_target_location": "safe_region",
  "intervention_type": "correct",
  "intervention_parameters": {
    "destination_pose": [0.2, 0.6, 0.72]
  },
  
  "post_feasible": true,
  "post_rgb_path": "data/intervention_v1/scene_t1_001/post_inv_0_rgb.png",
  "post_segmentation_path": "data/intervention_v1/scene_t1_001/post_inv_0_segmentation.png",
  
  "causal_effect": 1,
  "is_culprit": true,
  "culprit_object": "sugar_box",
  "culprit_relation": "ON_TOP_OF",
  
  "causal_mask_path": "data/intervention_v1/scene_t1_001/causal_mask.png",
  "target_mask_path": "data/intervention_v1/scene_t1_001/target_mask.png",
  
  "generation_seed": 42,
  "intervention_seed": 101,
  "split": "id",
  "manifest_hash": "a1b2c3d4..."
}
```

## 5. Directory Structure

```
data/intervention_v1/
├── manifests/
│   ├── intervention_manifest.jsonl
│   └── split_assignments.json
├── features/
│   ├── query_scene_t1_001_features.pt
│   ├── post_scene_t1_001_inv_0_features.pt
│   ├── post_scene_t1_001_inv_1_features.pt
│   └── crop_scene_t1_001_sugar_box_features.pt
└── scenes/
    ├── scene_t1_001/
    │   ├── pre_rgb.png
    │   ├── pre_segmentation.png
    │   ├── post_inv_0_rgb.png
    │   ├── post_inv_0_segmentation.png
    │   ├── post_inv_1_rgb.png
    │   ├── post_inv_1_segmentation.png
    │   └── masks/
    │       ├── sugar_box.png
    │       ├── mug.png
    │       ├── causal_mask.png
    │       └── target_mask.png
    └── scene_t1_002/
        ...
```

## 6. Controls and Deconfounding

To prevent the model from learning statistical shortcuts, the dataset generation must enforce:

1. **Intervention Index Randomization**: `intervention_idx` must be uncorrelated with `intervention_type` (e.g., the correct intervention shouldn't always be index 0).
2. **Object Role Swapping**: An object type (e.g., `coffee_can`) that is a culprit in scene A must appear as a distractor in scene B.
3. **No Leakage Across Splits**: All interventions originating from the same base scene must be assigned to the same dataset split.
4. **Target Location Variance**: `safe_region` poses must be sampled with variance so the model doesn't learn a fixed "good coordinate" heuristic.
5. **PROCEED Controls**: For PROCEED scenes, generate RELOCATE interventions for all objects. They should all yield $\Delta = 0$.

## 7. Scaling Object Counts

To avoid combinatorial explosion when moving from 1-distractor to N-distractor scenes:
- Do not generate all possible pairwise swap interventions.
- Adhere strictly to the generation rules (1 correct, N distractors=N irrelevant, 1 hard negative, 1 identity).
- Total interventions per scene = $N + 3$. Linear growth with object count.
