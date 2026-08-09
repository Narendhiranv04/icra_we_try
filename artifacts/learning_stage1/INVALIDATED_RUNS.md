# Invalidated Historical Runs

## 24-Pair Pilot Run
The existing historical result in run_summary.md based on "24 matched pairs" and "96 malformed records removed" is invalid. The true historical issue was stale/deleted pilot files from an old smoke/pilot output collision. This run should not be treated as a valid scientific result.

## Initial Final-Run ModuleNotFound Failure
The initial attempt at the final 30-run training script failed immediately due to a `ModuleNotFound` error (incorrect PYTHONPATH). No valid training results or checkpoints were produced by that attempt.

## BF16 Heatmap Visualization Failure
During Phase 12 postprocessing, heatmap generation failed with `TypeError: Got unsupported ScalarType BFloat16`. This failure was entirely local to the visualization script. All 30 model training, evaluation, and bootstrap metric runs had already completed successfully. This postprocessing crash did NOT invalidate the checkpoints or scientific metrics. It was fixed by adding `.float()` before NumPy conversion.
