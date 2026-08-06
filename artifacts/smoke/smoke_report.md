# Smoke Test Execution Report

## Overview
- **Commit Hash**: `d35a0cd829b5bf4353b61996ee78bc96a3fe6922`
- **Profile**: `smoke`
- **Overall Status**: **PASSED**
- **Unit Tests**: 23 / 23 passed (PASSED)
- **Dataset Validation**: PASSED
- **Split Validation**: PASSED
- **Reproducibility Regeneration**: PASSED
- **Demonstration Distinctness**: PASSED
- **Counterfactual Query Pairs**: 8 pairs (16 query images)
- **Standalone Positive Controls**: 4 controls

## Task Summary
1. **Task 1: "Open the box."**
   - Matched counterfactual pairs across splits (`id`, `unseen_object`, `unseen_background`, `compositional`)
   - Robot demonstration video (`open_box/demo_task1_smoke/rgb.mp4`)
2. **Task 2: "Place object1 in the target region."**
   - Matched counterfactual pairs across splits (`id`, `unseen_object`, `unseen_background`, `compositional`)
   - Robot demonstration video (`place_object/demo_task2_smoke/rgb.mp4`)

## Verified Artifacts
- `contact_sheet.png`: Grid layout of RGB queries, overlays, and causal violation masks
- `demonstration_montage.png`: Representative frame montage of robot task executions
- `representative_metadata.json`: EpisodeSpec metadata schemas
