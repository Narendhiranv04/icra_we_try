"""
HTML preview generator for visualizing benchmark query pairs, mask overlays, and decision labels.
"""

from pathlib import Path
from typing import List, Dict, Union
import json
import base64


def image_to_base64(img_path: Union[str, Path]) -> str:
    """Convert an image file to base64 data URI format for inline HTML rendering.
    
    Args:
        img_path: Path to image file.
        
    Returns:
        Base64 string data URI.
    """
    img_path = Path(img_path)
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class HTMLPreviewGenerator:
    """Generator for rendering standalone interactive HTML preview pages for dataset inspection."""

    def __init__(self, output_dir: Union[str, Path] = "data/previews"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(
        self,
        manifest_path: Union[str, Path],
        output_filename: str = "benchmark_preview.html",
    ) -> str:
        """Build HTML preview page from dataset manifest.
        
        Args:
            manifest_path: Path to manifest JSONL file.
            output_filename: Name of preview HTML file.
            
        Returns:
            Saved HTML file path string.
        """
        manifest_path = Path(manifest_path)
        records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        cards_html = []
        for rec in records:
            pair_id = rec.get("pair_id", "Unknown")
            task_id = rec.get("task_id", "")
            instruction = rec.get("instruction", "")

            stop_rgb_b64 = image_to_base64(rec["stop"]["rgb_path"])
            stop_mask_b64 = image_to_base64(rec["stop"]["culprit_mask_path"])

            proceed_rgb_b64 = image_to_base64(rec["proceed"]["rgb_path"])
            proceed_mask_b64 = image_to_base64(rec["proceed"]["culprit_mask_path"])

            card = f"""
            <div class="pair-card">
                <h3>Pair ID: {pair_id} | Task: {task_id}</h3>
                <p class="instruction"><strong>Instruction:</strong> "{instruction}"</p>
                <div class="comparison-grid">
                    <div class="column stop-col">
                        <span class="badge badge-stop">STOP (Precondition Violated)</span>
                        <div class="image-box">
                            <p>RGB Query</p>
                            <img src="{stop_rgb_b64}" alt="STOP RGB" />
                        </div>
                        <div class="image-box">
                            <p>Culprit Mask</p>
                            <img src="{stop_mask_b64}" alt="STOP Mask" />
                        </div>
                    </div>
                    <div class="column proceed-col">
                        <span class="badge badge-proceed">PROCEED (Precondition Satisfied)</span>
                        <div class="image-box">
                            <p>RGB Counterfactual</p>
                            <img src="{proceed_rgb_b64}" alt="PROCEED RGB" />
                        </div>
                        <div class="image-box">
                            <p>Culprit Mask</p>
                            <img src="{proceed_mask_b64}" alt="PROCEED Mask" />
                        </div>
                    </div>
                </div>
            </div>
            """
            cards_html.append(card)

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MuJoCo Relational Precondition Benchmark Preview</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            color: #38bdf8;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .pair-card {{
            background-color: #1e293b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        .instruction {{
            font-size: 1.1em;
            color: #cbd5e1;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }}
        .column {{
            background-color: #0f172a;
            padding: 15px;
            border-radius: 8px;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .badge-stop {{
            background-color: #ef4444;
            color: #ffffff;
        }}
        .badge-proceed {{
            background-color: #22c55e;
            color: #ffffff;
        }}
        .image-box img {{
            width: 100%;
            border-radius: 6px;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MuJoCo Relational Precondition Benchmark - Query Pair Preview</h1>
        {"".join(cards_html)}
    </div>
</body>
</html>
"""

        out_path = self.output_dir / output_filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        return str(out_path)
