# Context Forcing V1 Pilot Results

## Overview
This pilot tests if models can learn to condition their predictions on context when forced by a perfectly deconfounded dataset where visual shortcuts do not exist.

## Results
| Model | Seed 11 Val Acc | Seed 42 Val Acc | Mean Val Acc |
|-------|-----------------|-----------------|--------------|
| query_only | 0.703 | 0.703 | 0.703 |
| pooled_multimodal | 0.969 | 0.984 | 0.977 |
| relational_heatmap | 1.000 | 1.000 | 1.000 |

## Conclusion
If the multimodal models (pooled, relational) achieve >50% accuracy on the validation set, they are architecturally capable of conditioning, and their failure in Milestone 2 was due to **Shortcut-Permissive Training Data** (Outcome 1/2).

If all models are stuck at ~50% accuracy, this confirms an **Architectural Inability to Condition** (Outcome 3/4).