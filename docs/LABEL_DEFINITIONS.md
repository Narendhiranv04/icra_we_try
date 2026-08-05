# Label & Relational Predicate Definitions

## Decision Labels

- **`PROCEED`**: The intended action should be executed directly because the required relational precondition is satisfied.
- **`STOP`**: The intended next action should **not** be executed directly under the current scene state because a relational precondition is violated.

## Task 1: "Open the box."
- **Relational Predicate**: `ON_TOP_OF(object_i, B1_lid)`
- **STOP Condition**: At least one designated movable object rests on `B1_lid_panel`.
- **PROCEED Condition**: `B1_lid_panel` is clear (`BESIDE(object_i, box_B1)`).
- **Lid Frame Bounds**: Local lid extent `x in [-0.178, 0.178]`, `y in [-0.093, 0.093]`, `z in [0.0, 0.35]`.

## Task 2: "Place object1 in the target region."
- **Relational Predicate**: `OCCUPIES(object_2, target_region)`
- **STOP Condition**: A non-target object occupies `target_region_body` / `target_region_geom`.
- **PROCEED Condition**: `target_region_geom` is empty and available (`OUTSIDE(object_2, target_region)`).
- **Target Frame Bounds**: Local single-capacity target region center `[-0.10, -0.20, 0.581]`, half-extent `[0.10, 0.10, 0.001]`.
