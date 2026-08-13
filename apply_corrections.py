import re
import os

def update_architecture():
    path = "docs/CAUSAL_INTERVENTION_ARCHITECTURE.md"
    with open(path, "r") as f:
        content = f.read()
    
    # Correction 6: V1 Arch
    content = content.replace("4-layer Transformer encoder", "2-layer Transformer encoder")
    content = content.replace("where query patch tokens (Q) attend to the pooled output of the temporal encoder (K, V)", "where query patch tokens (Q) attend to the full sequence output of the temporal encoder (K, V)")
    
    # Correction 1: Intervention Descriptor
    old_descriptor = """z_interv = concat(
    proj_visual(object_crop_feature),      # (768,) DINOv2 feature of cropped object
    embed_operator(intervention_operator), # (64,) e.g., NONE, RELOCATE
    embed_dest(destination_type)           # (64,) e.g., CLEAR, STILL_OBSTRUCTING
)"""
    new_descriptor = """z_interv = concat(
    proj_visual(object_crop_feature),      # (768,) DINOv2 feature of cropped object
    current_geometry_feature,              # (16,) e.g., target-relative current pose
    proposed_destination_geometry,         # (16,) e.g., target-relative destination pose
    embed_operator(intervention_operator)  # (64,) e.g., NONE, RELOCATE
)"""
    content = content.replace(old_descriptor, new_descriptor)
    
    content = content.replace("concat(crop, type_embed, dest_embed)", "concat(crop, current_geom, dest_geom, operator_embed)")
    content = content.replace("concat(z_tilde_oi, interv_type_embed, destination_embed)", "concat(z_tilde_oi, current_geom, dest_geom, operator_embed)")
    
    with open(path, "w") as f:
        f.write(content)

def update_dataset_spec():
    path = "docs/CAUSAL_INTERVENTION_DATASET_SPEC.md"
    with open(path, "r") as f:
        content = f.read()
    
    # Correction 2: Delta in {-1, 0, 1}
    content = content.replace("\\in \\{0, 1\\}", "\\in \\{-1, 0, +1\\}")
    content = content.replace("so $\\Delta=0$ for all interventions, which serves as a critical control", "so $\\Delta=0$ for irrelevant interventions, and $\\Delta=-1$ for harmful interventions")
    
    # Correction 13/2: Update Generation rules
    old_rules = """| **Identity (None)** | N/A | N/A | 0 | Control. Pre-state equals post-state. |"""
    new_rules = """| **Identity (None)** | N/A | N/A | 0 | Control. Pre-state equals post-state. |
| **Harmful** | Distractor | Obstructing pose | -1 | Negative example on PROCEED scenes. Tests if model recognizes when an action breaks feasibility. |"""
    content = content.replace(old_rules, new_rules)
    
    content = content.replace("For PROCEED scenes, generate RELOCATE interventions for all objects. They should all yield $\\Delta = 0$.", 
                              "For PROCEED scenes, generate RELOCATE interventions for distractors that move them into obstructing poses (Harmful, $\\Delta = -1$), as well as Irrelevant and Identity interventions ($\\Delta = 0$).")
    
    # Correction 3: One culprit
    content = content.replace("For a given STOP scene (with $N$ objects),", "For a given STOP scene (with exactly 1 causal culprit and $N$ distractors),")
    content = content.replace("Total interventions per scene = $N + 3$", "Total interventions per scene = $N + 3$ (for STOP scenes)")
    
    # Privileged labels
    content = content.replace("intervention_type: str", "intervention_type: str          # PRIVILEGED_GT_ONLY: \"correct\" / \"irrelevant\" / \"hard_negative\" / \"harmful\" / \"none\"")
    content = content.replace("\"intervention_target_location\": \"safe_region\",", "\"intervention_target_location\": \"safe_region\",  # PRIVILEGED_GT_ONLY")
    content = content.replace("\"intervention_type\": \"correct\",", "\"intervention_type\": \"correct\",  # PRIVILEGED_GT_ONLY")
    
    with open(path, "w") as f:
        f.write(content)

def update_roadmap():
    path = "docs/EXPERIMENT_ROADMAP.md"
    with open(path, "r") as f:
        content = f.read()
    
    # Correction 7 & 11: Baselines
    old_b0 = "| B0 | Query-Only | No | No | MLP from global DINOv2 feature. Baseline for visual shortcuts. |"
    new_b0 = """| B0 | Query-Only | No | No | MLP from global DINOv2 feature. Baseline for visual shortcuts (ceiling ~0.75). |
| B0b | Simple Intervention | Yes | No | MLP from (scene_global, candidate_geom, operator). Weak baseline for candidate ranking. |"""
    content = content.replace(old_b0, new_b0)
    
    content = content.replace("B0 (Query-Only) accuracy $\\le 1/N + \\epsilon$ (approx. random chance)", "B0b (Simple Intervention) accuracy $\\le 1/N + \\epsilon$. (Note: pure B0 pre-scene only cannot rank interventions. For context forcing, B0 accuracy approaches the visual ceiling of ~0.75, not 0.50)")
    
    # Correction 2
    content = content.replace("$\\Delta_i=0$", "$\\Delta_i=0$")
    content = content.replace("subset of *irrelevant* interventions", "subset of *irrelevant* and *identity* interventions")
    
    # Correction 1
    content = content.replace("Zero out embeddings in `z_interv`.", "Zero out geometric features and operator embeddings in `z_interv`.")
    content = content.replace("Type/Dest Embeddings", "Geometry/Operator Embeddings")
    
    with open(path, "w") as f:
        f.write(content)

def update_phases():
    path = "docs/GEMINI_IMPLEMENTATION_PHASES.md"
    with open(path, "r") as f:
        content = f.read()
        
    content = content.replace("feature/intervention-v1", "feature/causal-intervention-v2")
    
    # P1 Update
    content = content.replace("correct, irrelevant, hard-negative", "correct, irrelevant, hard-negative, harmful")
    content = content.replace("- Generate NONE — identity control", "- Generate NONE — identity control\n  - For PROCEED scenes: generate RELOCATE(distractor, obstructing_pose) — harmful")
    content = content.replace("get exactly 4 interventions (correct, hard_neg, irrelevant, none)", "get exactly 4 interventions for STOP (correct, hard_neg, irrelevant, none)")
    
    # P2 Update
    content = content.replace("7. `test_correct_intervention_flips_feasibility` — Build a STOP scene for Task 1 with a blocker on lid. Apply RELOCATE(blocker, safe_region). Verify feasibility changes from False to True.",
                              "7. `test_correct_intervention_flips_feasibility` — Build a STOP scene for Task 1 with a blocker on lid. Apply RELOCATE(blocker, safe_region). Verify feasibility changes from False to True (causal_effect = +1).")
    content = content.replace("8. `test_irrelevant_preserves_infeasibility` — Same STOP scene. Apply RELOCATE(distractor, safe_region). Verify feasibility remains False.",
                              "8. `test_irrelevant_preserves_infeasibility` — Same STOP scene. Apply RELOCATE(distractor, safe_region). Verify feasibility remains False (causal_effect = 0).\n8b. `test_harmful_intervention_breaks_feasibility` — Build a PROCEED scene. Apply RELOCATE(distractor, onto_lid). Verify feasibility changes from True to False (causal_effect = -1).")
    
    # P5 Update
    content = content.replace("- Post-intervention query features (global + patch) — new", "- Post-intervention query features — (Note: For SUPERVISION/EVAL only, NOT model input)")
    content = content.replace("Loads intervention JSONL records", "Loads intervention JSONL records. Ensures post-intervention state is separate from forward inputs.")
    
    # P6 Update
    content = content.replace("interv_type_embed = nn.Embedding(2, 64) — NONE/RELOCATE", "interv_operator_embed = nn.Embedding(2, 64) — NONE/RELOCATE\n  - No destination or type embeddings. Use explicit geometric parameters.")
    content = content.replace("interv_dest_embed = nn.Embedding(4, 64) — destination types", "")
    content = content.replace("forward(self, text_feat, demo_global, query_patch, interv_object_crop, interv_type_idx, interv_dest_idx)",
                              "forward(self, text_feat, demo_global, query_patch, interv_object_crop, interv_current_geom, interv_dest_geom, interv_operator_idx)")
    content = content.replace("concat(crop, type_embed, dest_embed)", "concat(crop, current_geom, dest_geom, operator_embed)")
    
    with open(path, "w") as f:
        f.write(content)

def update_plan():
    path = "docs/CAUSAL_INTERVENTION_RESEARCH_PLAN.md"
    with open(path, "r") as f:
        content = f.read()
    
    content = content.replace("feature/intervention-v1", "feature/causal-intervention-v2")
    content = content.replace("f7439e2", "a362018")
    content = content.replace('HEAD commit**: `a362018` ("Complete context_forcing_v1 diagnostic pilot")', 'HEAD commit**: `a362018` ("docs: add causal intervention implementation plan")')
    
    # Corrections
    content = content.replace("4-layer Transformer encoder processing", "2-layer Transformer encoder processing")
    content = content.replace("2-layer Transformer cross-attention where query patch tokens (Q) attend to the pooled output of the temporal encoder", "2-layer Transformer cross-attention where query patch tokens (Q) attend to the full sequence output of the temporal encoder")
    content = content.replace("at chance boundary for forced-context task", "approaching visual-only ceiling of ~0.75 for forced-context task")
    content = content.replace("Query-only achieves high accuracy on Benchmark A → visual shortcuts", "Query-only achieves high accuracy on Benchmark A → visual shortcuts (approaching ~0.75 visual ceiling on balanced A/B/C/D context sets)")
    content = content.replace("Model loses visual goal grounding; performance drops.", "Model performance may drop. (Empirical question: demo may not be strictly necessary for early tasks).")
    
    content = content.replace("Δ_i ∈ {0, 1}", "Δ_i ∈ {-1, 0, +1}")
    content = content.replace("{-1, 0, 1}", "{-1, 0, +1}")
    
    content = content.replace("For each STOP scene with N candidate objects", "For each STOP scene with exactly 1 causal culprit and N distractors")
    
    # V2 Targets
    content = content.replace("L = λ_feas·L_feas + λ_rank·L_rank_interv + λ_effect·L_effect + λ_heat·L_heat",
                              "L = λ_feas·L_feas + λ_rank·L_rank_interv\nNote: Redundant effect/heatmap losses removed or optional. Target is P(F_post=1 | rho_i).")
    content = content.replace("Predicted effect: `Δ̂_i = σ(logit_post(intervention_i)) - σ(logit_post(NONE))`",
                              "Predicted effect: `Δ̂_i = P(F_post=1 | rho_i) - P(F_post=1 | NONE)`")
                              
    content = content.replace("Top-1 > 80% ID; CFR > 70%", "Top-1 > 80% ID; CFR > 70% (PROVISIONAL GATES)")
    content = content.replace("Top-1 > 80% on ID split; CFR > 70%", "Top-1 > 80% on ID split; CFR > 70% (PROVISIONAL GATES)")
    content = content.replace("Oracle CFR ≥ 95%; model CFR > 70%", "Oracle CFR ≥ 95%; model CFR > 70% (PROVISIONAL GATES)")
    
    content = content.replace("destination_type_embed,              # (64,) learned embed for clear/still_obstructing", 
                              "current_geometry_feat,               # (16,) target-relative geometry\n        proposed_destination_geometry,       # (16,) proposed target-relative geometry")
    content = content.replace("intervention_type_embed,", "operator_embed,")
    content = content.replace("destination_type: str          # \"CLEAR\" (clears obstruction), \"STILL_OBSTRUCTING\" (hard negative), \"IRRELEVANT\" (distractor object)",
                              "destination_type: str          # PRIVILEGED_GT_ONLY")
    content = content.replace("intervention_type: str    # \"correct\"/\"irrelevant\"/\"hard_negative\"/\"none\"", "intervention_type: str    # PRIVILEGED_GT_ONLY")
    content = content.replace("intervention_target_location: str", "intervention_target_location: str  # PRIVILEGED_GT_ONLY")
    
    content = content.replace("B0 Query-only**: No intervention, no context → unchanged", "B0 Query-only**: Pre-scene only, no intervention descriptor → unchanged\n- **B0b Simple Intervention**: MLP over (scene_global, candidate_geom, operator), no relational context.")
    
    content = content.replace("For PROCEED scenes: all candidate interventions yield Δ_i = 0", "For PROCEED scenes: generate Harmful interventions (Δ_i = -1) by moving distractors to obstructing poses, plus controls (Δ_i = 0)")

    with open(path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    update_architecture()
    update_dataset_spec()
    update_roadmap()
    update_phases()
    update_plan()
    print("Corrections applied.")
