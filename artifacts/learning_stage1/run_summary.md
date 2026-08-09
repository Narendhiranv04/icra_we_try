# Milestone 2 Final Run Summary

## Overview
This run represents the final scientific conclusion for Milestone 2, evaluating multiple models on their ability to condition representations on multimodal inputs (language, video demonstrations) using the `full_no_ranking`, `demo_query`, `language_query`, `pooled_multimodal`, and `relational_heatmap` architectures.

A total of 30 experiments (6 model types x 5 random seeds) were rigorously evaluated on Benchmark A (standard infeasibility prediction) and Benchmark B (a counterfactual dataset isolating contextual dependence). 

## Primary Scientific Conclusion
**Finding:** All multimodal models, including the newly proposed `relational_heatmap` v1, rely entirely on visual and spatial shortcuts (e.g. geometric configurations of objects in the RGB frame) rather than correctly processing the conditioning multimodal queries.

**Evidence:**
1. **Zero Reversal Accuracy:** On Benchmark B, which evaluates whether models flip their predictions when given an identical RGB frame but a different conditional query (e.g. text or demonstration), the `reversal_accuracy` is 0.0 for all models across all seeds.
2. **Generic Query Insensitivity:** The `mean_flip_rate` when swapping a specific task demonstration for the same generic text ("Perform the demonstrated task.") on an identical RGB frame is 0.0 (except for a trivial 0.005 on relational_heatmap). The mean absolute delta probability is <0.03 for all models. This proves that models output identical scores regardless of the actual conditional query provided.

**Interpretation:** The learning formulation in v1 fundamentally failed to enforce context dependence. While the models achieved varying levels of success on Benchmark A, they simply memorized spatial shortcuts in the visual observation that correlated with the target task rather than learning a conditional representation.

## Limitations & Missing Provenance
- The dataset provenance uses exact commit tracking for `TRAINED_CODE_COMMIT`, but lacks the raw `run_final_experiment.sh` execution log, as it was run via standard bash rather than a tracked framework.
- Missing specific random seed bounds for the original `Benchmark-B` candidate generation pass.
- Missing exhaustive hardware metrics (VRAM consumption over time) during model training.

## Artifacts Generated
- `dataset_provenance.json`: Metadata, commits, and environment variables.
- `experiment_manifest.json`: List of all runs generated.
- `aggregate_metrics.json` / `bootstrap_results.json`: Full metrics aggregated across 5 random seeds.
- `generic_demo_pair_sensitivity.json`: Analysis comparing same-RGB states given distinct task demos under generic text conditioning.
- `INVALIDATED_RUNS.md`: Record of historically invalidated pilot runs.
