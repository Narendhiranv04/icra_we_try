# Causal Intervention Architecture

> Companion to the master causal intervention research plan.
> Details the transition from the current relational cross-attention architecture (V1) to the intervention-conditioned model (V2) and eventually the object-centric causal model (V3).

---

## 1. Existing Architecture (V1: Relational Cross-Attention)

The current model (`DemoLanguageConditionedRelationalModel`) operates by embedding task context into a temporal sequence and allowing query image patches to attend to that context.

1. **Vision Encoder**: Frozen DINOv2 (`ViT-B/14`). Extracts 768-d global token and $16 \times 16$ patch tokens.
2. **Text Encoder**: Frozen `all-MiniLM-L6-v2`. Extracts 384-d sentence embedding.
3. **Temporal Encoder**: 2-layer Transformer encoder processing `[text_token, demo_global_1, ..., demo_global_K]`.
4. **Cross Attention**: 2-layer Transformer cross-attention where query patch tokens (Q) attend to the full sequence output of the temporal encoder (K, V).
5. **Output**: Mean pooling of the updated patch tokens feeds into an MLP to predict feasibility.

**Limitation**: V1 can only predict *whether* a scene is feasible. It has no mechanism to reason about counterfactual interventions.

---

## 2. Architecture V2: Intervention-Conditioned Relational Model

**Goal**: Minimal extension to V1 that enables scoring candidate interventions $\rho_i$.

**Approach**: We append an intervention descriptor token to the temporal context sequence. The cross-attention then conditions the query patch processing on *both* the task context and the proposed intervention. The model predicts the *post-intervention* feasibility.

### 2.1 Intervention Descriptor Token

An intervention is described by the object being acted upon and the semantic destination.

```
z_interv = concat(
    proj_visual(object_crop_feature),      # (768,) DINOv2 feature of cropped object
    current_geometry_feature,              # (16,) e.g., target-relative current pose
    proposed_destination_geometry,         # (16,) e.g., target-relative destination pose
    embed_operator(intervention_operator)  # (64,) e.g., NONE, RELOCATE
)
z_int = proj_interv(z_interv)              # (256,)
```

For the `NONE` intervention (pre-intervention state evaluation), the `object_crop_feature` is zeroed out, and the `intervention_operator` is `NONE`.

### 2.2 Sequence Assembly

The temporal sequence is extended:
```
seq = [z_t, z_d_1, ..., z_d_K, z_int]  # Shape: (1 + K + 1, 256)
```

### 2.3 Forward Pass

1. Pass `seq` through the Temporal Encoder.
2. Query patches attend to the updated `seq` in Cross-Attention.
3. Mean pool the updated query patches.
4. Pass through classifier MLP to get `logit_post` (predicted post-intervention feasibility).

The predicted causal effect of the intervention is derived by comparing it to the `NONE` intervention output:
$\hat{\Delta}_i = \sigma(\text{logit\_post\_interv}_i) - \sigma(\text{logit\_post\_none})$

---

## 3. Architecture V3: Object-Centric Causal Model (Future)

**Goal**: Move from implicit patch-based reasoning to explicit factored object reasoning, enabling latent transitions and explicit irrelevance priors.

### 3.1 Tokenization

Instead of patch tokens, the scene is tokenized into explicit entities.

**Object Tokens** ($N$ tokens):
```
z_oi = proj_obj(concat(
    object_crop_feature,
    bbox_geometry_features,
    relative_pose_to_target
))
```

**Action Token** (1 token):
```
z_a = proj_action(concat(
    action_type_embed,
    action_target_embed,
    text_instruction_embed
))
```

**Demo Tokens** ($K$ tokens):
```
z_dk = proj_demo(concat(
    frame_global_feature,
    ee_pose_relative_to_target,
    gripper_state
))
```

### 3.2 Relational Encoder

A Transformer processes the unified set of tokens: `[z_a, z_d1..K, z_o1..N]`.
Through self-attention, object tokens contextualize each other relative to the action and demonstration.

```
Z_out = relational_encoder([z_a, z_d1..K, z_o1..N])
z_action = Z_out[0]
z_objects = Z_out[1+K:]
```

### 3.3 Output Heads

1. **Feasibility**: `clf(concat(z_action, mean_pool(z_objects)))`
2. **Causal Relevance**: For each object, `rel_clf(concat(z_action, z_objects[i]))` predicting ground-truth $\Delta_i$.

### 3.4 Latent Transition (Intervention Mechanism)

To evaluate an intervention $\rho_i$ on object $i$:
1. Update object $i$'s token using a latent transition network:
   `z_objects_prime[i] = transition_net(z_objects[i], embed_intervention(rho_i))`
2. Keep all other object tokens unchanged:
   `z_objects_prime[j] = z_objects[j]` for $j \neq i$
3. Re-evaluate feasibility using `z_objects_prime`.

This enforces a strong architectural prior: an intervention on object $i$ only changes object $i$'s latent state, and if the overall feasibility changes, object $i$ must be causally relevant.
