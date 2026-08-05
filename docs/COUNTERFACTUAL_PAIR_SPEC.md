# Counterfactual Pair Specification

## Pair Invariants
A matched counterfactual pair (`STOP` and `PROCEED`) is constructed by:
1. Building the `STOP` `EpisodeSpec`.
2. Cloning the `EpisodeSpec` identically (background, camera, lighting, robot pose, distractors, object identities, non-target object poses).
3. Modifying **only** the position of the relation-defining object(s):
   - For Task 1: Move blocker object from `B1_lid_panel` to beside `box_B1`.
   - For Task 2: Move occupant object from `target_region_geom` to outside `target_region`.
4. Generating paired RGB images and masks under identical camera perspective (`front_camera`).

## Mask Semantics per Sample
1. `instance_segmentation.png`: Encoded simulator instance segmentation map.
2. `candidate_object_mask.png`: Binary mask highlighting candidate blocker/occupant object (present in both STOP and PROCEED).
3. `relation_target_mask.png`: Binary mask highlighting relation target surface (`B1_lid_panel` or `target_region_geom`, present in both STOP and PROCEED).
4. `causal_violation_mask.png`:
   - For `STOP`: Union of candidate object mask and relation target mask.
   - For `PROCEED`: All zeros (`0` non-zero pixels).
5. `combined_relation_visualization.png`: RGB overlay showing relation target in cyan and candidate object in red.
