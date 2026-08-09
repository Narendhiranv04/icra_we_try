# Milestone 2 Final Run Summary

## Overview
This run represents the final scientific conclusion for Milestone 2, evaluating multiple models on their ability to condition representations on multimodal inputs (language, video demonstrations) using the `full_no_ranking`, `demo_query`, `language_query`, `pooled_multimodal`, and `relational_heatmap` architectures.

A total of 30 experiments (6 model types x 5 random seeds) were rigorously evaluated on Benchmark A (standard infeasibility prediction) and Benchmark B (a counterfactual dataset isolating contextual dependence). 

## Primary Scientific Conclusion
**Finding:** The `query_only` model performs extremely strongly on Benchmark A, establishing that Benchmark A is visually shortcut-solvable. However, all six model families have a 0.0 reversal accuracy on Benchmark B across the five-seed aggregate.

**Evidence:**
1. **Zero Reversal Accuracy:** On Benchmark B, which evaluates whether models flip their predictions when given an identical RGB frame but a different conditional query (e.g. text or demonstration), the `reversal_accuracy` is 0.0 for all models across all seeds.
2. **Generic Query Insensitivity:** Generic-demo sensitivity remains small. Mean prediction flip rates are 0.005 for demo_query, 0.015 for full_no_ranking, 0.01 for pooled_multimodal, and 0.0 for relational_heatmap. Mean absolute probability changes remain below 0.03 for all model families.

**Interpretation:** The v1 formulation fails to use the task/demo context sufficiently to perform the required same-RGB decision reversal. This is strong evidence of shortcut exploitation, but Benchmark B alone does not prove that every output is produced exclusively from RGB. Distribution-shift or dominant-class decision collapse must remain a possible explanation for some Benchmark-B behavior.

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
