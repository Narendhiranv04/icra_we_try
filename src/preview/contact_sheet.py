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
        """Create composite grid contact sheet image of benchmark pairs.
        
        Args:
            manifest_path: Path to manifest JSONL file.
            output_filename: Filename for contact sheet PNG.
            thumb_size: (width, height) thumbnail resize tuple.
            
        Returns:
            Saved contact sheet image path.
        """
        manifest_path = Path(manifest_path)
        records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if not records:
            return ""

        rows = len(records)
        cols = 4 # STOP RGB, STOP Mask, PROCEED RGB, PROCEED Mask
        tw, th = thumb_size
        canvas = Image.new("RGB", (cols * tw, rows * th), color=(15, 23, 42))

        for r_idx, rec in enumerate(records):
            stop_rgb = Image.open(rec["stop"]["rgb_path"]).resize(thumb_size)
            stop_mask = Image.open(rec["stop"]["culprit_mask_path"]).resize(thumb_size).convert("RGB")
            proceed_rgb = Image.open(rec["proceed"]["rgb_path"]).resize(thumb_size)
            proceed_mask = Image.open(rec["proceed"]["culprit_mask_path"]).resize(thumb_size).convert("RGB")

            canvas.paste(stop_rgb, (0 * tw, r_idx * th))
            canvas.paste(stop_mask, (1 * tw, r_idx * th))
            canvas.paste(proceed_rgb, (2 * tw, r_idx * th))
            canvas.paste(proceed_mask, (3 * tw, r_idx * th))

        out_path = self.output_dir / output_filename
        canvas.save(out_path)
        return str(out_path)
