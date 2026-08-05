"""
Contact sheet visualizer for creating composite grid images of benchmark pairs.
"""

from pathlib import Path
from typing import List, Union
import json
import PIL.Image as Image
import numpy as np


class ContactSheetGenerator:
    """Generator for rendering tiled image contact sheets of query RGBs and masks."""

    def __init__(self, output_dir: Union[str, Path] = "data/previews"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_contact_sheet(
        self,
        manifest_path: Union[str, Path],
        output_filename: str = "contact_sheet.png",
        thumb_size: tuple[int, int] = (320, 240),
    ) -> str:
        """Create composite grid contact sheet image of benchmark pairs."""
        manifest_path = Path(manifest_path)
        records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "stop" in rec and "proceed" in rec:
                        records.append(rec)

        if not records:
            return ""

        rows = len(records)
        cols = 6  # STOP RGB, STOP Overlay, STOP Causal, PROCEED RGB, PROCEED Overlay, PROCEED Causal
        tw, th = thumb_size
        canvas = Image.new("RGB", (cols * tw, rows * th), color=(15, 23, 42))

        for r_idx, rec in enumerate(records):
            stop_meta = rec["stop"]
            proceed_meta = rec["proceed"]

            stop_rgb = Image.open(stop_meta["rgb_path"]).resize(thumb_size)
            stop_vis_p = stop_meta.get("combined_visualization_path", stop_meta["rgb_path"])
            stop_vis = Image.open(stop_vis_p).resize(thumb_size)
            stop_caus_p = stop_meta.get("causal_violation_mask_path", stop_meta.get("culprit_mask_path"))
            stop_caus = Image.open(stop_caus_p).resize(thumb_size).convert("RGB")

            proceed_rgb = Image.open(proceed_meta["rgb_path"]).resize(thumb_size)
            proceed_vis_p = proceed_meta.get("combined_visualization_path", proceed_meta["rgb_path"])
            proceed_vis = Image.open(proceed_vis_p).resize(thumb_size)
            proceed_caus_p = proceed_meta.get("causal_violation_mask_path", proceed_meta.get("culprit_mask_path"))
            proceed_caus = Image.open(proceed_caus_p).resize(thumb_size).convert("RGB")

            canvas.paste(stop_rgb, (0 * tw, r_idx * th))
            canvas.paste(stop_vis, (1 * tw, r_idx * th))
            canvas.paste(stop_caus, (2 * tw, r_idx * th))
            canvas.paste(proceed_rgb, (3 * tw, r_idx * th))
            canvas.paste(proceed_vis, (4 * tw, r_idx * th))
            canvas.paste(proceed_caus, (5 * tw, r_idx * th))

        out_path = self.output_dir / output_filename
        canvas.save(out_path)
        return str(out_path)
