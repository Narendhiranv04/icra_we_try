"""
HTML preview generator for visualizing benchmark query pairs, mask overlays, decision labels, and metadata.
"""

from pathlib import Path
from typing import List, Dict, Union
import json
import base64


def image_to_base64(img_path: Union[str, Path]) -> str:
    """Convert an image file to base64 data URI format for inline HTML rendering."""
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
        """Build HTML preview page from dataset manifest."""
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
            split = rec.get("split", "id")

            # STOP images
            stop_rgb_b64 = image_to_base64(rec["stop"]["rgb_path"])
            stop_cand_b64 = image_to_base64(rec["stop"].get("candidate_object_mask_path", rec["stop"].get("culprit_mask_path")))
            stop_target_b64 = image_to_base64(rec["stop"].get("relation_target_mask_path", rec["stop"].get("region_mask_path")))
            stop_causal_b64 = image_to_base64(rec["stop"].get("causal_violation_mask_path", ""))
            stop_vis_b64 = image_to_base64(rec["stop"].get("combined_visualization_path", rec["stop"]["rgb_path"]))

            # PROCEED images
            proceed_rgb_b64 = image_to_base64(rec["proceed"]["rgb_path"])
            proceed_cand_b64 = image_to_base64(rec["proceed"].get("candidate_object_mask_path", rec["proceed"].get("culprit_mask_path")))
            proceed_target_b64 = image_to_base64(rec["proceed"].get("relation_target_mask_path", rec["proceed"].get("region_mask_path")))
            proceed_causal_b64 = image_to_base64(rec["proceed"].get("causal_violation_mask_path", ""))
            proceed_vis_b64 = image_to_base64(rec["proceed"].get("combined_visualization_path", rec["proceed"]["rgb_path"]))

            stop_spec = rec["stop"].get("spec", {})
            rel_before = stop_spec.get("relation_before", "")
            rel_after = stop_spec.get("relation_after", "")

            card = f"""
            <div class="pair-card">
                <div class="card-header">
                    <h2>Pair ID: {pair_id} | Task: {task_id} | Split: <span class="split-tag">{split}</span></h2>
                    <p class="instruction"><strong>Goal Instruction:</strong> "{instruction}"</p>
                </div>
                
                <div class="comparison-grid">
                    <!-- STOP COLUMN -->
                    <div class="column stop-col">
                        <span class="badge badge-stop">STOP (Precondition Violated)</span>
                        <p class="rel-pred"><strong>Relation:</strong> <code>{rel_before}</code></p>
                        <div class="image-grid">
                            <div class="image-box"><p>RGB Query Image</p><img src="{stop_rgb_b64}" alt="STOP RGB" /></div>
                            <div class="image-box"><p>Combined Relation Overlay</p><img src="{stop_vis_b64}" alt="STOP Overlay" /></div>
                            <div class="image-box"><p>Candidate Object Mask</p><img src="{stop_cand_b64}" alt="STOP Candidate Mask" /></div>
                            <div class="image-box"><p>Relation Target Mask</p><img src="{stop_target_b64}" alt="STOP Target Mask" /></div>
                            <div class="image-box"><p>Causal Violation Mask (Union)</p><img src="{stop_causal_b64}" alt="STOP Causal Mask" /></div>
                        </div>
                    </div>

                    <!-- PROCEED COLUMN -->
                    <div class="column proceed-col">
                        <span class="badge badge-proceed">PROCEED (Precondition Satisfied)</span>
                        <p class="rel-pred"><strong>Relation:</strong> <code>{rel_after}</code></p>
                        <div class="image-grid">
                            <div class="image-box"><p>RGB Counterfactual Image</p><img src="{proceed_rgb_b64}" alt="PROCEED RGB" /></div>
                            <div class="image-box"><p>Combined Relation Overlay</p><img src="{proceed_vis_b64}" alt="PROCEED Overlay" /></div>
                            <div class="image-box"><p>Candidate Object Mask</p><img src="{proceed_cand_b64}" alt="PROCEED Candidate Mask" /></div>
                            <div class="image-box"><p>Relation Target Mask</p><img src="{proceed_target_b64}" alt="PROCEED Target Mask" /></div>
                            <div class="image-box"><p>Causal Violation Mask (Empty)</p><img src="{proceed_causal_b64}" alt="PROCEED Causal Mask" /></div>
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
    <title>MuJoCo Relational Precondition Benchmark - Query Pair Preview</title>
    <style>
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        h1 {{
            text-align: center;
            color: #38bdf8;
            margin-bottom: 24px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .pair-card {{
            background-color: #1e293b;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 32px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
            border: 1px solid #334155;
        }}
        .card-header {{
            border-bottom: 1px solid #334155;
            padding-bottom: 12px;
            margin-bottom: 16px;
        }}
        .instruction {{
            font-size: 1.15em;
            color: #cbd5e1;
            margin: 4px 0 0 0;
        }}
        .split-tag {{
            background-color: #0284c7;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .rel-pred {{
            color: #94a3b8;
            font-size: 0.95em;
            margin: 8px 0;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        .column {{
            background-color: #0f172a;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #1e293b;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .badge-stop {{
            background-color: #dc2626;
            color: #ffffff;
        }}
        .badge-proceed {{
            background-color: #16a34a;
            color: #ffffff;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 12px;
        }}
        .image-box {{
            background-color: #1e293b;
            padding: 8px;
            border-radius: 6px;
            text-align: center;
        }}
        .image-box p {{
            font-size: 0.8em;
            color: #94a3b8;
            margin: 0 0 6px 0;
        }}
        .image-box img {{
            width: 100%;
            border-radius: 4px;
            display: block;
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
