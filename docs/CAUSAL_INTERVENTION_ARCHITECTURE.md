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

**Approach**: We append a neutral intervention descriptor token to the temporal context sequence. The cross-attention then conditions the query patch processing on *both* the task context and the proposed candidate intervention. The model predicts the *post-intervention* feasibility $P(F_{post}=1 \mid s_{pre}, \text{context}, \rho_i)$.

### 2.1 Intervention Descriptor Token

A candidate intervention is described neutrally by the visual appearance of the acted object, its current geometry in the scene, its proposed target-relative destination geometry, and the operator type (`NONE` vs. `RELOCATE`). It contains NO privileged outcome labels or semantic destination tokens (e.g., `SAFE_REGION`, `STILL_OBSTRUCTING`, or `is_culprit` are strictly prohibited from entering `forward()`).

```
z_rho_raw = concat(
    object_crop_feature,            # (D_crop=768,) DINOv2 feature of cropped candidate object
    current_geometry_feature,       # (D_current_geom,) candidate localization & target-relative current pose/extent
    proposed_destination_geometry,  # (D_dest_geom,) proposed target-relative destination pose
    embed_operator(operator_idx)    # (D_operator=64,) learned embedding for NONE (0) / RELOCATE (1)
)

D_interv = D_crop + D_current_geom + D_dest_geom + D_operator
z_int = proj_interv(z_rho_raw)      # nn.Linear(D_interv, latent_dim=256)
```

#### Geometry Feature Semantics:
- **`current_geometry_feature` ($D_{current\_geom}$)**: Parameterized schema carrying candidate localization in the current observation, target-relative current 3D position/orientation, and bounding extent.
- **`proposed_destination_geometry` ($D_{dest\_geom}$)**: Parameterized schema carrying the proposed target-relative 3D destination coordinate and orientation.
- Dimensionalities $D_{current\_geom}$ and $D_{dest\_geom}$ are schema/config-driven (e.g. 16-d each in prototype configuration), yielding $D_{interv} = 768 + 16 + 16 + 64 = 864$. In code, `interv_input_dim = vision_dim + current_geom_dim + dest_geom_dim + operator_embed_dim` dynamically defines `proj_interv = nn.Linear(interv_input_dim, latent_dim)`.

For the `NONE` intervention (pre-intervention state evaluation), `object_crop_feature`, `current_geometry_feature`, and `proposed_destination_geometry` are zero-tensors, and `operator_idx` is `0` (`NONE`).

### 2.2 Sequence Assembly

The temporal sequence is extended:
```
seq = [z_t, z_d_1, ..., z_d_K, z_int]  # Shape: (1 + K + 1, 256)
```

### 2.3 Forward Pass

1. Pass `seq` through the Temporal Encoder (self-attention).
2. Query image patches attend to the updated `seq` in Cross-Attention.
3. Mean pool the updated query patches.
4. Pass through classifier MLP to get `logit_post` (predicted post-intervention feasibility logit).

The model forward path observes strictly pre-intervention observations. Post-intervention images/features NEVER enter inference.

The predicted causal effect of candidate intervention $i$ is derived by comparing it to the `NONE` intervention output:
$$\hat{\Delta}_i = \sigma(\text{logit\_post\_interv}_i) - \sigma(\text{logit\_post\_none})$$

---

## 3. Architecture V3: Object-Centric Causal Model (Future Concept)

**Goal**: Move from implicit patch-based reasoning to explicit factored object reasoning, enabling global relational counterfactual re-encoding and explicit irrelevance priors.

### 3.1 Tokenization

Instead of patch tokens, the scene is tokenized into explicit entities.

**Object Tokens** ($N$ tokens):
```
x_i = proj_obj(concat(
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

A Transformer processes the unified set of raw tokens: `[z_a, z_d1..K, x_1..N]`.
Through self-attention, object tokens contextualize each other relative to the action and demonstration.

```
Z_out = relational_encoder([z_a, z_d1..K, x_1..N])
z_action = Z_out[0]
z_objects = Z_out[1+K:]
```

### 3.3 Output Heads

1. **Feasibility**: `clf(concat(z_action, mean_pool(z_objects)))`
2. **Causal Relevance**: For each object, `rel_clf(concat(z_action, z_objects[i]))` predicting ground-truth causal sensitivity $|\Delta_i| > 0$ and repair relevance $\Delta_i > 0$.

### 3.4 Counterfactual Relational Mechanism (Intervention Reasoning)

For candidate relocation $\rho_i = \text{RELOCATE}(o_i, \text{dest\_geom}_i)$ with known proposed destination geometry:
1. Construct the counterfactual raw object token using the proposed destination geometry:
   $$x_i' = E_{obj}(\text{visual}_i, \text{destination\_geometry}_i)$$
2. Replace the raw entity token in the scene set:
   $$x_i \to x_i'$$
3. Rerun the relational encoder over ALL entities and context tokens:
   $$Z_{out}' = \mathcal{R}_\theta([z_a, z_{d1..K}, x_1, \dots, x_i', \dots, x_N])$$
4. Evaluate post-intervention feasibility from $Z_{out}'$.

**Rationale**: Moving object $i$ physically alters pairwise geometric and clearance relations with *all other scene entities*. Rather than freezing already-contextualized latents and mutating an isolated slot via a separate transition network, replacing the raw factored entity and re-running relational encoding naturally propagates global relational updates. (Learned latent transition networks may be reserved for contact-rich dynamics where post-state geometry cannot be analytically specified).
