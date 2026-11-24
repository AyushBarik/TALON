# /// script
# dependencies = ["torch", "transformers", "pillow", "numpy", "scipy"]
# ///
"""Token keep order of every case-study frame (Figure 2), by the recipe of fig/attn_map.py (Figure 3).

The [CLS] attention of the last ViT layer, averaged over heads and smoothed with a one-patch Gaussian, is pooled to
the 8x4 token grid of the figures; cells are ranked from most to least attended.
Model: facebook/dino-vits16 (illustration only; not the pruner's encoder).
Writes fig/case/keep_order.json: {"<frame index>": [cell, ...]} with cell = row*8 + col and row 0 = top of the image.
Run: uv run --no-project fig/case_keep_order.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import ViTImageProcessor, ViTModel

HERE = Path(__file__).parent
FRAMES = HERE / "case" / "frames"
OUT = HERE / "case" / "keep_order.json"
COLS, ROWS = 8, 4

proc = ViTImageProcessor.from_pretrained("facebook/dino-vits16")
model = ViTModel.from_pretrained("facebook/dino-vits16", add_pooling_layer=False, attn_implementation="eager").eval()

orders = {}
for path in sorted(FRAMES.glob("*.jpg")):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    ph, pw = H // 16, W // 16
    x = proc(img, do_resize=True, size={"height": ph * 16, "width": pw * 16}, return_tensors="pt").pixel_values
    with torch.no_grad():
        out = model(x, output_attentions=True, interpolate_pos_encoding=True)
    att = out.attentions[-1][0, :, 0, 1:].mean(0).reshape(ph, pw).numpy()  # [CLS] -> patches, mean over heads
    att = torch.tensor(gaussian_filter(att, sigma=1.0))
    g = F.adaptive_avg_pool2d(att[None, None], (ROWS, COLS))[0, 0].numpy()  # row 0 = top of the image
    orders[str(int(path.stem))] = np.argsort(-g.reshape(-1), kind="stable").tolist()

OUT.write_text(json.dumps(orders, indent=0) + "\n")
print("wrote", OUT, len(orders), "frames")
