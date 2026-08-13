import os
from PIL import Image, ImageDraw, ImageFont

scenes = ["context_scene_000_A", "context_scene_010_B", "context_scene_020_C"]
data_dir = "data/context_challenge"
out_path = "artifacts/learning_stage1/demo_dataset.png"

# We want 3 rows (scenes) and 3 columns (RGB, Mask 1, Mask 2)
# Load one image to get size
sample = Image.open(os.path.join(data_dir, scenes[0], "rgb.png"))
w, h = sample.size
padding = 40
top_padding = 60
left_padding = 100

total_w = w * 3 + padding * 4 + left_padding
total_h = h * 3 + padding * 4 + top_padding

canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

col_titles = ["RGB", "Task 1 Causal Mask", "Task 2 Causal Mask"]

# Try to load a font, or fallback to default
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

# Draw col titles
for j, title in enumerate(col_titles):
    x = left_padding + padding + j * (w + padding) + w // 2 - 50
    y = padding
    draw.text((x, y), title, fill=(0, 0, 0), font=font)

for i, scene in enumerate(scenes):
    # Draw row titles
    y_text = top_padding + padding + i * (h + padding) + h // 2
    draw.text((padding, y_text), f"Scene {i+1}", fill=(0, 0, 0), font=font)
    
    rgb = Image.open(os.path.join(data_dir, scene, "rgb.png")).convert('RGB')
    mask1 = Image.open(os.path.join(data_dir, scene, "task1_causal_mask.png")).convert('RGB')
    mask2 = Image.open(os.path.join(data_dir, scene, "task2_causal_mask.png")).convert('RGB')
    
    # Position
    y = top_padding + padding + i * (h + padding)
    x0 = left_padding + padding
    x1 = x0 + w + padding
    x2 = x1 + w + padding
    
    canvas.paste(rgb, (x0, y))
    canvas.paste(mask1, (x1, y))
    canvas.paste(mask2, (x2, y))

canvas.save(out_path)
print("Saved to", out_path)
