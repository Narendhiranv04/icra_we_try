import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

scenes = ["context_scene_000_A", "context_scene_010_B", "context_scene_020_C"]
data_dir = "data/context_challenge"

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle("Dataset Examples", fontsize=16)

col_titles = ["RGB", "Task 1 Mask", "Task 2 Mask"]

for i, scene in enumerate(scenes):
    rgb_path = os.path.join(data_dir, scene, "rgb.png")
    mask1_path = os.path.join(data_dir, scene, "task1_causal_mask.png")
    mask2_path = os.path.join(data_dir, scene, "task2_causal_mask.png")
    
    img_rgb = Image.open(rgb_path)
    img_m1 = Image.open(mask1_path)
    img_m2 = Image.open(mask2_path)
    
    axes[i, 0].imshow(img_rgb)
    axes[i, 1].imshow(img_m1, cmap='gray')
    axes[i, 2].imshow(img_m2, cmap='gray')
    
    for j in range(3):
        axes[i, j].axis('off')
        if i == 0:
            axes[i, j].set_title(col_titles[j])
        if j == 0:
            axes[i, j].text(-0.1, 0.5, f"Scene {i+1}", transform=axes[i, j].transAxes, 
                            fontsize=12, va='center', ha='right', rotation=90)

plt.tight_layout()
plt.savefig("demo_dataset.png", dpi=150)
print("Saved demo_dataset.png")
