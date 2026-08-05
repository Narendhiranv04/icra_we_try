# Demonstration Generation Specification

## Overview
All demonstration videos show genuine Fetch mobile manipulator robot arm executions (`robot0:...`), using `VerticalIK` inverse kinematics and joint position actuators rather than direct free-joint coordinate overwrites.

## Open Box Execution Protocol (`BoxOpenExecutor`)
1. **Initial Phase**: Closed box, clear lid. Robot in `right_side` base pose.
2. **Approach Phase**: Robot arm moves end-effector from hover pose to `B1_lid_handle_grasp` handle site via IK trajectory.
3. **Attachment Phase**: Gripper reaches handle; temporary MuJoCo equality weld constraint `robot0:open_weld_B1_lid` activates at exact contact pose.
4. **Opening Motion Phase**: Robot arm joint trajectory follows the circular hinge arc from 0 to 1.57 rad (90 degrees). Hinge position actuator assists lid motion.
5. **Release & Retreat**: Weld constraint deactivates (`eq_active = 0`); arm retreats back to hover pose.
6. **Final Static Phase**: Lid remains open at target angle.

## Pick and Place Execution Protocol (`PlaceObjectExecutor`)
1. **Initial Phase**: `object1` starts outside target region; target region empty. Robot in `home` base pose.
2. **Approach Phase**: Robot arm moves end-effector to hover above `object1` and descends to pick height.
3. **Grasp Phase**: Temporary equality weld constraint `robot0:grasp_weld_target` activates at end-effector contact.
4. **Transport & Placement Phase**: Robot arm lifts `object1`, moves along 3D parabolic trajectory, and places `object1` on `target_region_body` surface (`[-0.10, -0.20, 0.65]`).
5. **Release & Retreat**: Weld constraint deactivates (`eq_active = 0`); arm lifts vertically and retreats.
6. **Final Static Phase**: `object1` rests stably inside `target_region`.
