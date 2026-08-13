# Causal Intervention Experiment Roadmap

> Companion to the master causal intervention research plan.
> Details the sequence of empirical tests and ablation studies necessary to validate the causal representation learning claims.

---

## 1. Primary Empirical Claims & Feasibility Hierarchy

### 1.1 Feasibility Level Hierarchy
To ensure architectural and conceptual clarity, we formally distinguish four levels of feasibility:
- **$F_R(s, a)$ — Relational Precondition Feasibility**: Symbolic and spatial preconditions for action $a$ (e.g., lid is clear for `OPEN(box)`, target region is unoccupied for `PLACE(object, target)`).
- **$F_M(s, a, \theta)$ — Continuous Motion Feasibility**: Kinematic reachability, collision avoidance, and IK feasibility for execution parameter $\theta$.
- **$F_A(s, a) = F_R(s, a) \land F_M(s, a, \theta)$ — Executable Action Feasibility**: Conjunction of relational preconditions and motion execution.
- **$F_\pi(s, \pi_{\text{remaining}})$ — Residual-Plan Feasibility**: Multi-step downstream execution feasibility w.r.t. a remaining plan prefix $\pi_{\text{remaining}}$.

The current research phase focuses strictly on learning **intervention-grounded relational feasibility $F_R$**. Motion ($F_M$) and residual-plan ($F_\pi$) conditioning are deferred to subsequent project phases.

### 1.2 Primary Empirical Claims
1. **Shortcut-Free Representation**: Visual-only pre-scene models cannot reliably identify reparative interventions in our deconfounded dataset.
2. **Intervention Grounding**: Conditioning on candidate interventions and their simulator-validated effects allows the model to learn true causal relevance and predict valid repairs (high Counterfactual Repair Rate).
3. **Irrelevance Invariance**: The learned representation is invariant to interventions on causally irrelevant objects.

---

## 2. Baselines and Model Variants

To isolate the source of performance, we compare several model variants:

| ID | Name | Candidate Intervention Input? | Training Loss | Architecture & Purpose |
|---|---|---|---|---|
| **B0** | **Query-Only Feasibility** | No | $\mathcal{L}_{feas}$ | 2-layer MLP from global DINOv2 feature. Pre-scene feasibility / visual shortcut baseline. **Cannot rank candidates** (receives identical inputs for all candidates of a scene). |
| **B0b** | **Simple Intervention Baseline** | Yes | $\mathcal{L}_{post\_feas}$ | 3-layer MLP over `concat(scene_global, crop_feat, current_geom, dest_geom, operator_embed)`. Weak candidate-ranking baseline without relational cross-attention. |
| **B3** | **Intervention Relational (Feas-Only)** | Yes (same as V2) | $\mathcal{L}_{post\_feas}$ ($\lambda_{rank}=0$) | Same intervention-conditioned relational architecture as V2, but trained strictly on post-feasibility BCE without candidate ranking loss. Direct ablation for the ranking objective. |
| **B4** | **Direct Culprit Relational** | No | $\mathcal{L}_{feas} + \mathcal{L}_{culprit}$ | Relational cross-attention model with a head predicting `is_culprit` directly from query patch tokens. Tests whether static culprit identification suffices vs. intervention effect prediction. |
| **V2** | **Intervention Relational** (Proposed) | Yes | $\mathcal{L}_{post\_feas} + \lambda_{rank}\mathcal{L}_{rank}$ | Proposed intervention-conditioned relational model. Appends neutral intervention descriptor token to temporal sequence before cross-attention. |

---

## 3. Core Experiment Suite

All core experiments are run using a 5-seed protocol (seeds: 11, 23, 42, 67, 101) with metrics aggregated using 10,000 bootstrap resamples to compute 95% confidence intervals.

> [!NOTE]
> Numerical thresholds listed below represent **provisional engineering targets**, not rigid publication claims. Scientific validation rests on high simulator oracle upper bounds, zero leakage, outperforming candidate-aware baselines (B0b / B3), and successful simulator repair execution.

### Exp 1: Shortcut Baselines (B0, B0b, B3 vs. V2)
- **Goal**: Establish that intervention ranking cannot be solved purely by static visual shortcuts (B0) or candidate-aware MLPs without relational reasoning (B0b), and isolate the gain from candidate ranking loss (B3 vs. V2).
- **Metric**: Intervention Top-1 Accuracy and Post-Feasibility AUROC on ID split.
- **Success Criteria**: 
  - B0 cannot produce candidate-specific rankings.
  - B0b accuracy $\le 1/N + \epsilon$ on deconfounded candidates.
  - V2 achieves significantly higher Top-1 accuracy and CFR than B3 and B0b.

### Exp 2: Direct Culprit vs. Intervention Conditioning (B4 vs. V2)
- **Goal**: Demonstrate that predicting the *counterfactual effect of an intervention* ($\hat{\Delta}_i$) leads to better actionable repairs than predicting a static culprit label.
- **Metric**: Counterfactual Repair Rate (CFR) via oracle simulation of the highest-ranked intervention. Culprit Top-1 Accuracy.
- **Success Criteria**: V2 achieves significantly higher CFR and Culprit Top-1 than B4.

### Exp 3: Main Efficacy (V2)
- **Goal**: Validate that V2 reliably identifies correct interventions and predicts post-intervention feasibility.
- **Metric**: Intervention Top-1, CFR, AUROC.
- **Provisional Targets**: Intervention Top-1 $> 80\%$, CFR $> 70\%$ on ID split.

### Exp 4: Irrelevance Invariance
- **Goal**: Show that the model ignores distractors and identity controls.
- **Protocol**: Evaluate V2 on the subset of *irrelevant* and *identity* interventions ($\Delta_i=0$).
- **Metric**: False Relevance Rate (FRR) — proportion of irrelevant interventions where $|\hat{\Delta}_i| > \tau$ (e.g., $\tau=0.2$). Mean $|\hat{\Delta}_i|$.
- **Provisional Targets**: FRR $< 5\%$, Mean $|\hat{\Delta}_i| < 0.05$.

### Exp 5: Hard Negative Discrimination
- **Goal**: Show the model understands spatial relations, not just object identity.
- **Protocol**: Compare predicted $\hat{\Delta}_i$ between Correct interventions (move culprit to safe region) and Hard Negative interventions (move culprit but still obstruct).
- **Metric**: AUC of separating Correct vs. Hard Negative based on $\hat{\Delta}_i$.
- **Provisional Target**: AUC $> 0.90$.

---

## 4. Ablation Matrix

To isolate the contribution of specific architectural components and loss terms in V2, we conduct a leave-one-out ablation study (5 seeds per ablation).

| ID | Ablation | Mechanism | Hypothesis |
|---|---|---|---|
| **A1** | No Intervention Input | Zero out `z_int` during forward pass. | Performance degrades to pre-scene level (cannot rank candidates). |
| **A2** | No Text/Instruction | Zero out `text_feat`. | Model fails to distinguish tasks in multi-task evaluation. |
| **A3** | No Demonstration | Zero out `demo_global`. | Empirical exploration: measures visual goal grounding contribution when text instruction is present. |
| **A4** | No Ranking Loss ($\lambda_{rank}=0$) | Train with $\lambda_{rank} = 0$ (Model B3). | Candidate ranking accuracy and CFR degrade compared to full V2. |
| **A8** | No Object Crop | Zero out `object_crop_feature` in `z_interv`. | Model cannot identify *which* object is being intervened upon; Top-1 drops. |
| **A9** | No Geometry Features | Zero out `current_geometry` and `destination_geometry`. | Model cannot reason about spatial feasibility; Hard Negative discrimination fails. |

---

## 5. Out-of-Distribution (OOD) Generalization

Evaluate the trained V2 model (from ID training) on held-out splits to test robustness.

| Split | Challenge | Expected Behavior |
|---|---|---|
| **unseen_object** | Novel object geometries/textures (e.g., cup, bowl) | Graceful degradation. V3 (object-centric) expected to improve this in later work. |
| **compositional** | Novel combinations of seen factors | Minor degradation. |
| **higher_count** | Scenes with 1 culprit + 3+ distractors (clutter scaling) | Degradation due to visual clutter in patch attention. |

---

## 6. Representation Probes

To understand *how* the model represents causality internally, we train linear probes on frozen latent representations (e.g., the pooled context vector `z_S` or the cross-attention output vector `z_R`).

**Probes:**
1. **Feasibility Probe**: Predict $F_R(s, a)$.
2. **Culprit Probe**: Predict identity of the culprit object.
3. **Relation Probe**: Predict the violated relational predicate.

**Hypothesis**: The latents of the intervention-supervised V2 model will be significantly more linearly separable for Culprit and Relation probes than the feasibility-only model, indicating that intervention supervision induces structured causal representations.
