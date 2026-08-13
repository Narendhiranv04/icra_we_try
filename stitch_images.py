import os
from PIL import Image, ImageDraw, ImageFont

samples = [
    "compositional_sample_30_task_2_STOP",
    "compositional_sample_0_task_1_STOP"
]

data_dir = "/home/projects/long-horizon/infeasibiilty-latent/artifacts/learning_stage1/heatmaps"
out_path = "/home/projects/long-horizon/.gemini/antigravity-ide/brain/c9be9186-f7b7-4aec-acca-29276c0d5567/stitched_image.png"

sample_img = Image.open(os.path.join(data_dir, f"{samples[0]}_query.jpg"))
w, h = sample_img.size
padding = 40
top_padding = 60
left_padding = 120

total_w = w * 4 + padding * 5 + left_padding
total_h = h * 2 + padding * 3 + top_padding

canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

col_titles = ["Query RGB", "Ground Truth Mask", "Predicted Heatmap", "Overlay"]
row_labels = ["Sample 30\n(T2 STOP)", "Sample 0\n(T1 STOP)"]

try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

for j, title in enumerate(col_titles):
    x = left_padding + padding + j * (w + padding) + w // 2 - 50
    y = padding
    draw.text((x, y), title, fill=(0, 0, 0), font=font)

for i, sample_prefix in enumerate(samples):
    y_text = top_padding + padding + i * (h + padding) + h // 2
    draw.text((padding, y_text), row_labels[i], fill=(0, 0, 0), font=font)
    
    img_query = Image.open(os.path.join(data_dir, f"{sample_prefix}_query.jpg")).convert('RGB')
    img_gt = Image.open(os.path.join(data_dir, f"{sample_prefix}_gt_mask.jpg")).convert('RGB')
    img_heat = Image.open(os.path.join(data_dir, f"{sample_prefix}_heatmap.jpg")).convert('RGB')
    img_over = Image.open(os.path.join(data_dir, f"{sample_prefix}_overlay.jpg")).convert('RGB')
    
    if img_gt.size != (w, h):
        img_gt = img_gt.resize((w, h), Image.NEAREST)
        
    imgs = [img_query, img_gt, img_heat, img_over]
    
    for j, img in enumerate(imgs):
        x = left_padding + padding + j * (w + padding)
        y = top_padding + padding + i * (h + padding)
        canvas.paste(img, (x, y))

canvas.save(out_path)
print("Saved to", out_path)
