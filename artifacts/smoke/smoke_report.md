# Smoke Test Execution Report

## Overview
- **Profile**: `smoke`
- **Status**: **PASSED**
- **Unit Tests**: 14 / 14 passed
- **Demonstration Videos**: 2 videos generated with genuine Fetch robot arm manipulation
- **Counterfactual Query Pairs**: 8 pairs (16 query images)
- **Standalone Positive Controls**: 4 controls

## Task Summary
1. **Task 1: "Open the box."**
   - Matched counterfactual pairs across splits (`id`, `unseen_object`, `unseen_background`, `compositional`)
   - 1 genuine robot demonstration video (`open_box/demo_task1_smoke/rgb.mp4`)
2. **Task 2: "Place object1 in the target region."**
   - Matched counterfactual pairs across splits (`id`, `unseen_object`, `unseen_background`, `compositional`)
   - 1 genuine robot demonstration video (`place_object/demo_task2_smoke/rgb.mp4`)

## Verified Artifacts
- `contact_sheet.png`: Grid layout of RGB queries, overlays, and causal violation masks
- `demonstration_montage.png`: Representative frame montage of robot task executions
- `representative_metadata.json`: EpisodeSpec metadata schemas
