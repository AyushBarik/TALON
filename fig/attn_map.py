# /// script
# dependencies = ["torch", "transformers", "pillow", "numpy", "matplotlib", "scipy"]
# ///
"""Attention-map picture for Figure 3 (overview, stage 3).

Standard recipe (DINO / PyTorch forum style): the [CLS] attention of the last ViT layer, averaged over heads,
smoothed with a one-patch Gaussian, bilinearly upsampled to the frame size, coloured with jet and alpha-blended over the frame.
Model: facebook/dino-vits16 (illustration only; not the pruner's encoder).
Also prints the 8x4 token keep order (cell index = row*8 + col, row 0 = bottom, as in the TikZ macros), ranked by
the same map pooled to 8x4, so the bars and the kept tokens in the figure match the heat map.
Run: uv run --no-project fig/attn_map.py
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import ViTImageProcessor, ViTModel

HERE = Path(__file__).parent
SRC = HERE / "frames" / "frame4.jpg"
OUT = HERE / "frames" / "frame4_attn.png"
ALPHA = 0.5  # weight of the heat map in the blend

img = Image.open(SRC).convert("RGB")
W, H = img.size
proc = ViTImageProcessor.from_pretrained("facebook/dino-vits16")
model = ViTModel.from_pretrained("facebook/dino-vits16", add_pooling_layer=False, attn_implementation="eager").eval()

# Run at the native frame size (patch 16 -> a 20x11 grid) with interpolated position embeddings.
ph, pw = H // 16, W // 16
x = proc(img, do_resize=True, size={"height": ph * 16, "width": pw * 16}, return_tensors="pt").pixel_values
with torch.no_grad():
    out = model(x, output_attentions=True, interpolate_pos_encoding=True)
att = out.attentions[-1][0, :, 0, 1:].mean(0).reshape(ph, pw).numpy()  # [CLS] -> patches, mean over heads
att = torch.tensor(gaussian_filter(att, sigma=1.0))

# Pixel-level heat map, min-max normalised.
up = F.interpolate(att[None, None], size=(H * 2, W * 2), mode="bilinear", align_corners=False)[0, 0].numpy()
up = (up - up.min()) / (up.max() - up.min())
heat = colormaps["jet"](up)[..., :3]
base = np.asarray(img.resize((W * 2, H * 2), Image.BICUBIC), dtype=float) / 255
blend = (1 - ALPHA) * base + ALPHA * heat
Image.fromarray((blend * 255).round().astype(np.uint8)).save(OUT)

# 8x4 token ranking from the same map (adaptive average pooling).
g = F.adaptive_avg_pool2d(torch.tensor(up)[None, None], (4, 8))[0, 0].numpy()  # row 0 = top of the image
g = g[::-1]  # row 0 = bottom, as TikZ y grows upward
order = np.argsort(-g.reshape(-1), kind="stable")
print("keep order:", ",".join(map(str, order)))
print("wrote", OUT)
