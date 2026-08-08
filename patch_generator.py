def patch_gen():
    with open("src/generation/context_challenge_generator.py", "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "targ_mask_1 = renderer.render_region_mask(data, [\"B1_lid_panel\"])" in line:
            insert_str = """
        # Visibility validation
        if targ_mask_1.sum() < 500:
            renderer.close()
            return self.generate_scene(scene_id, state, seed=seed+1)
        if state in ["A", "C"] and cand_mask_1.sum() < 50:
            renderer.close()
            return self.generate_scene(scene_id, state, seed=seed+1)
"""
            lines.insert(i+1, insert_str)
            break
            
    for i, line in enumerate(lines):
        if "targ_mask_2 = renderer.render_region_mask(data, [\"target_region_geom\"])" in line:
            insert_str = """
        if targ_mask_2.sum() < 500:
            renderer.close()
            return self.generate_scene(scene_id, state, seed=seed+1)
        if state in ["B", "C"] and cand_mask_2.sum() < 50:
            renderer.close()
            return self.generate_scene(scene_id, state, seed=seed+1)
"""
            lines.insert(i+1, insert_str)
            break

    with open("src/generation/context_challenge_generator.py", "w") as f:
        f.writelines(lines)

patch_gen()
