# Smoke Test Execution Report

## Overview
- **Profile**: `smoke`
- **Status**: **PASSED**
- **Unit Tests**: 4 / 4 passed
- **Demonstration Videos**: 2 videos generated with genuine Fetch robot arm manipulation
- **Counterfactual Query Pairs**: 8 pairs (16 query images)

## Task Summary
1. **Task 1: "Open the box."**
   - 4 matched counterfactual pairs (1-blocker and 2-blockers on lid vs beside box)
   - 1 genuine robot demonstration video (`demo_task1_smoke.mp4`, 120 frames)
2. **Task 2: "Place object1 in the target region."**
   - 4 matched counterfactual pairs (occupants inside single-capacity target vs outside)
   - 1 genuine robot demonstration video (`demo_task2_smoke.mp4`, 120 frames)

## Verified Artifacts
- `contact_sheet.png`: 6-column grid of RGB queries, overlays, and causal violation masks
- `demonstration_montage.png`: Representative frame montage of robot task executions
- `representative_metadata.json`: EpisodeSpec metadata schemas
