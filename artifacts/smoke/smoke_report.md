# Smoke Test Execution Report

## Overview
- **Tested Code Commit**: `4d11c118652fd8182429d277f56211285c07c700`
- **Report Generation Commit**: `4d11c118652fd8182429d277f56211285c07c700`
- **Profile**: `smoke`
- **Overall Status**: **FAILED**
- **Unit Tests**: 19 / 19 passed (PASSED)
- **Counterfactual Query Pairs**: 8 pairs (16 query images)
- **Standalone Positive Controls**: 4 controls

## Required Reports Status
- `dataset_validation.json`: **PASSED**
- `split_validation.json`: **PASSED**
- `reproducibility_report.json`: **PASSED**
- `demonstration_validation.json`: **PASSED**
- `demonstration_distinctness.json`: **PASSED**
- `control_distribution.json`: **FAILED**

## Verified Artifacts
- `contact_sheet.png`: Grid layout of RGB queries, overlays, and causal violation masks
- `demonstration_montage.png`: Representative frame montage of robot task executions
- `representative_metadata.json`: EpisodeSpec metadata schemas
