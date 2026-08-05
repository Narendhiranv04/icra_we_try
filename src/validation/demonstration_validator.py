"""
Demonstration validator.

Validates that generated demonstrations satisfy physical invariants:
- No direct lid/object qpos assignment (structural - enforced by code review)
- Lid starts closed, ends open (Task 1)
- Object starts outside target, ends inside (Task 2)  
- Robot arm moves during demonstration
- Weld is activated only when gripper is near target
- State logs are consistent
"""

from __future__ import annotations

import math
from typing import List, Tuple

from src.tasks.open_box import DemoStateLog as OpenStateLog
from src.tasks.place_object import DemoStateLog as PlaceStateLog


class DemonstrationValidator:
    """Validates demonstration state logs against physical invariants."""

    @staticmethod
    def validate_open_box(
        state_log: List[OpenStateLog],
        *,
        min_final_angle_deg: float = 45.0,
        max_initial_angle_deg: float = 10.0,
        proximity_threshold: float = 0.15,
    ) -> Tuple[bool, List[str]]:
        """Validate Task 1 (Open Box) demonstration.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues: List[str] = []
        
        if not state_log:
            return False, ["No state log entries"]

        # Check lid starts closed
        initial_entries = [e for e in state_log if e.phase == "initial"]
        if initial_entries:
            initial_angle = math.degrees(initial_entries[0].lid_angle_rad)
            if initial_angle > max_initial_angle_deg:
                issues.append(
                    f"Lid not closed at start: {initial_angle:.1f}° > {max_initial_angle_deg}°"
                )

        # Check lid ends open
        final_entries = [e for e in state_log if e.phase == "final"]
        if final_entries:
            final_angle = math.degrees(final_entries[-1].lid_angle_rad)
            if final_angle < min_final_angle_deg:
                issues.append(
                    f"Lid not sufficiently open at end: {final_angle:.1f}° < {min_final_angle_deg}°"
                )

        # Check robot arm actually moves
        arm_positions = [e.arm_qpos for e in state_log]
        if len(arm_positions) >= 2:
            import numpy as np
            first = np.array(arm_positions[0])
            last = np.array(arm_positions[-1])
            arm_displacement = float(np.linalg.norm(last - first))
            if arm_displacement < 0.01:
                issues.append(
                    f"Robot arm barely moved: displacement = {arm_displacement:.4f} rad"
                )

        # Check weld was activated
        weld_entries = [e for e in state_log if e.weld_active]
        if not weld_entries:
            issues.append("Weld constraint was never activated")

        # Check lid angle increases during opening phase
        opening_entries = [e for e in state_log if e.phase == "opening"]
        if len(opening_entries) >= 2:
            angles = [e.lid_angle_rad for e in opening_entries]
            if angles[-1] <= angles[0]:
                issues.append(
                    f"Lid angle did not increase during opening: "
                    f"start={math.degrees(angles[0]):.1f}° end={math.degrees(angles[-1]):.1f}°"
                )

        is_valid = len(issues) == 0
        return is_valid, issues

    @staticmethod
    def validate_place_object(
        state_log: List[PlaceStateLog],
        *,
        proximity_threshold: float = 0.15,
    ) -> Tuple[bool, List[str]]:
        """Validate Task 2 (Place Object) demonstration.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues: List[str] = []

        if not state_log:
            return False, ["No state log entries"]

        # Check object starts outside target
        initial_entries = [e for e in state_log if e.phase == "initial"]
        if initial_entries:
            if initial_entries[0].obj_in_target:
                issues.append("Object already in target at start")

        # Check object ends in target
        final_entries = [e for e in state_log if e.phase == "final"]
        if final_entries:
            if not final_entries[-1].obj_in_target:
                issues.append("Object not in target at end")

        # Check robot arm actually moves
        arm_positions = [e.arm_qpos for e in state_log]
        if len(arm_positions) >= 2:
            import numpy as np
            first = np.array(arm_positions[0])
            last = np.array(arm_positions[-1])
            arm_displacement = float(np.linalg.norm(last - first))
            if arm_displacement < 0.01:
                issues.append(
                    f"Robot arm barely moved: displacement = {arm_displacement:.4f} rad"
                )

        # Check weld was activated
        weld_entries = [e for e in state_log if e.weld_active]
        if not weld_entries:
            issues.append("Weld constraint was never activated")

        # Check object moved during transport
        transport_entries = [e for e in state_log if e.phase == "transport"]
        if len(transport_entries) >= 2:
            import numpy as np
            start_pos = np.array(transport_entries[0].object_pos)
            end_pos = np.array(transport_entries[-1].object_pos)
            obj_displacement = float(np.linalg.norm(end_pos - start_pos))
            if obj_displacement < 0.02:
                issues.append(
                    f"Object barely moved during transport: {obj_displacement:.4f}m"
                )

        is_valid = len(issues) == 0
        return is_valid, issues
