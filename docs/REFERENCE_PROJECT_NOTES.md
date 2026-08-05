# Reference Project Notes (`/home/naren/RA_iiith`)

## Relationship to Reference Project
- **Reference Project Path**: `/home/naren/RA_iiith` (READ-ONLY, UNMODIFIED)
- **Active Workspace**: `/home/naren/RA_iiith_new`
- **GitHub Repository**: `https://github.com/Narendhiranv04/icra_we_try.git`

## Adapted Components from Reference Project
1. **Fetch Manipulator XML Integration**: Merged Farama's Gymnasium-Robotics Fetch mobile manipulator assets into `kitchen_base.xml`.
2. **Inverse Kinematics**: Adapted `VerticalIK` damped least-squares solver for Fetch 7-DOF arm from `pick_motion.py`.
3. **Container & Object Manipulation**: Adapted `BoxOpenExecutor` and `PlaceExecutor` trajectory stepping and relative pose weld grasp assistance from `open_motion.py` and `place_motion.py`.
4. **Target Region**: Standardized single-capacity target region on countertop surface at `pos=[-0.10, -0.20, 0.581]`.
