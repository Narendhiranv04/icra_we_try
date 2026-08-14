# Causal Intervention Architecture

> Companion to the master causal intervention research plan.
> Details the transition from the legacy cross-attention baseline (V1) to the intervention-conditioned model (V2) and eventually the object-centric causal model (V3).

---

## 1. Legacy Architecture (V1: Demonstration-Conditioned Relational Model)

The legacy model (`DemoLanguageConditionedRelationalModel`) operated by embedding task text and multi-step demonstration frames into a temporal sequence and allowing query image patches to cross-attend to that sequence:

1. **Vision Encoder**: Frozen DINOv2 (`ViT-B/14`). Extracts 768-d global token and $16 \times 16 = 256$ patch tokens.
2. **Text Encoder**: Frozen `all-MiniLM-L6-v2`. Extracts 384-d sentence embedding.
3. **Temporal Encoder**: 2-layer Transformer encoder processing `[text_token, demo_global_1, ..., demo_global_K]`.
4. **Cross Attention**: 2-layer Transformer cross-attention where query patch tokens (Q) attend to the full sequence output of the temporal encoder (K, V).
5. **Output**: Mean pooling of updated patch tokens feeds into an MLP to predict feasibility.

**Limitation**: V1 can only predict *whether* a scene is feasible under demonstrated context. It has no mechanism to reason about counterfactual physical interventions.

---

## 2. Architecture V2: Intervention-Conditioned Relational Model (Current Implemented)

**Goal**: Enable scoring candidate interventions $\rho_i$ without privileged simulator metadata, predicting post-intervention feasibility $P(F_{post}=1 \mid s_{pre}, \text{text}, \rho_i)$.

### 2.1 Seven-Field Model-Visible Input Contract

The model observes strictly pre-intervention observations across 7 fields:
1. `text_feat` (384-d float32): Frozen `all-MiniLM-L6-v2` mean-pooled text embedding.
2. `scene_global` (768-d float32): Frozen DINOv2 CLS token of the pre-intervention scene RGB.
3. `scene_patch` ($256 \times 768$-d float32): Frozen DINOv2 patch tokens of the pre-intervention scene RGB.
4. `candidate_visual` (768-d float32): Frozen DINOv2 CLS token of the candidate object crop (zeros for `NONE`).
5. `current_geometry` (3-d float32): Task-relative current 3D position $(x, y, z)$ in meters (zeros for `NONE`).
6. `destination_geometry` (3-d float32): Task-relative destination 3D position $(x, y, z)$ in meters (zeros for `NONE`).
7. `operator_idx` (scalar long): Operator type index ($0 = \text{NONE}, 1 = \text{RELOCATE}$).

No demonstration tokens, privileged object IDs, ground-truth culprit labels, or post-intervention visual features are visible to the model.

### 2.2 Intervention Descriptor Token

The raw intervention descriptor $h_\rho$ concatenates candidate visual and spatial parameters:
```
h_rho = concat(
    candidate_visual,       # (768,)
    current_geometry,       # (3,)
    destination_geometry,   # (3,)
    operator_embed(op_idx)  # (64,) learned embedding for NONE (0) / RELOCATE (1)
)  # Dimension D_interv = 768 + 3 + 3 + 64 = 838

z_int = proj_intervention(h_rho)  # nn.Linear(838, latent_dim=256)
```

For the `NONE` operator, candidate visual and geometries are zero-tensors and `operator_idx = 0`.

### 2.3 Context Sequence & Relational Cross-Attention

The context sequence comprises 3 tokens:
```
Z_C = [proj_text(text_feat), proj_scene_global(scene_global), z_int] + pos_embed  # Shape: (B, 3, 256)
```
1. **Context Self-Attention**: 2-layer Transformer encoder processes $Z_C$.
2. **Patch Cross-Attention**: Query pre-scene patches $Z_R = \text{proj\_patch}(\text{scene\_patch})$ ($B \times 256 \times 256$) cross-attend to context keys/values $Z_C$ via 2-layer Transformer cross-attention.
3. **Pooling & Classification**: Mean pooling over the 256 cross-attended patch tokens yields relational embedding $z_R \in \mathbb{R}^{256}$, classified by MLP head to produce unbounded post-feasibility logit `post_logit`.
4. **Ranking Score**: By construction, `ranking_score == post_logit`.

### 2.4 Objective Contract & Architectural Ablations

- **B0 (`query_only`)**: Feasibility predicted purely from `scene_global` via MLP. Zero candidate inputs by construction (candidate-invariant within scene). Objective: BCE only ($\lambda_{rank} = 0$).
- **B0b (`simple_intervention`)**: Candidate-aware MLP over all 7 fields concatenated without patch cross-attention. Objective: BCE + scene-normalized pairwise ranking loss ($\lambda_{rank} = 1.0$).
- **B3 (`intervention_relational`)**: Relational cross-attention architecture with feasibility BCE only ($\lambda_{rank} = 0$).
- **V2 (`intervention_relational`)**: Relational cross-attention architecture with feasibility BCE + scene-normalized pairwise ranking loss ($\lambda_{rank} = 1.0$).

B3 and V2 share the identical `InterventionRelationalModel` class, identical hyperparameters, and identical initial parameter fingerprints under fixed seed.

---

## 3. Architecture V3: Object-Centric Causal Model (Future Extension)

**Goal**: Factor the scene into explicit object entity tokens and reason over counterfactual mutations with graph/relational propagation.

- **Object Entity Tokens**: Factored per-object visual crop + 3D bounding geometry.
- **Relational Counterfactual Re-Encoding**: Replace raw object token $x_i \to x_i'$ with proposed destination geometry and rerun the relational encoder over all scene entities to propagate global clearance updates.
- **Demonstration Conditioning**: Multi-step demonstration trajectory tokens may be reintroduced as an optional context conditioning mechanism in V3.
