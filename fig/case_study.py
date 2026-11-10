"""Case study (one column): the frames RAG and TALON show the VLM for one question, at the same token budget.

Source: Video-MME question 378-1 (video WOGaS19VZ8Q, medium), subtitled setting, Qwen3-VL-8B, budget 12,288 tokens.
fig/case/records.jsonl holds the two logged per-question records (frames, tokens per frame, prediction):
  naive        bench_runs/naive/videomme_sub.jsonl
  talon_nodiv  bench_runs/abl_talon_nodiv/videomme_sub.jsonl   (TALON without the diversity term; DRAFT stand-in,
               regenerate from the final TALON run before submission)
fig/case/frames/<index>.jpg are the frames shown (640x360).

RAG shows 8 of the frames it sent, hand-picked so that none shows the air gun (RAG_FRAMES). TALON shows 12: its two highest-allocated frames plus
frames evenly spaced over the rest, in time order. Frames are crisp; a TALON frame that received fewer tokens is
pixelated in visibly distinct levels (BLOCKS), coarser for fewer tokens.

Regenerate:  uv run --no-project --with matplotlib --with numpy --with pillow python fig/case_study.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from orx_figstyle import COLUMN, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, "case")
QUESTION = "What is the color of the pneumatic air gun?"
OPTIONS = ["A. Green", "B. Red", "C. Blue", "D. Orange"]
GOLD = "C"
ROWS = [("naive", "RAG", 4), ("talon_nodiv", "TALON", 6)]  # (record, header, frames per row)
RAG_FRAMES = [1, 14, 47, 77, 86, 91, 109, 111]  # all among the frames RAG sent; none shows the air gun
N_TOP = 2  # TALON's highest-allocated frames always shown
# (minimum tokens, block size in px of a 640x360 frame): sharp, slight, moderate, strong, very strong
BLOCKS = [(160, 1), (128, 6), (64, 12), (32, 22), (0, 40)]
FRAME_W, FRAME_H = 640, 360
INK, GRAY, RULE = "#111111", "#6B6B6B", "#4D4D4D"
BAND = "#EFEFEF"
RED, GREEN = "#C0392B", "#1E8449"


def frame(index, tokens, pixelate):
    im = Image.open(os.path.join(CASE, "frames", f"{index:03d}.jpg")).convert("RGB")
    block = next(b for t, b in BLOCKS if tokens >= t) if pixelate else 1
    if block > 1:
        im = im.resize((FRAME_W // block, FRAME_H // block), Image.BILINEAR).resize((FRAME_W, FRAME_H), Image.NEAREST)
    return np.asarray(im)


def shown(rec, n, top):
    """Positions in the record of the frames to show, in time order."""
    order = np.argsort(rec["tokens_per_frame"])[::-1][:top].tolist()
    rest = [i for i in range(len(rec["frames"])) if i not in order]
    even = [rest[i] for i in np.linspace(0, len(rest) - 1, n - top).round().astype(int)]
    return sorted(order + even)


def mosaic(rec, per_row, ours):
    pos = shown(rec, 2 * per_row, N_TOP) if ours else [rec["frames"].index(f) for f in RAG_FRAMES]
    tiles = [frame(rec["frames"][i], rec["tokens_per_frame"][i], ours) for i in pos]
    return np.vstack([np.hstack(tiles[r * per_row:(r + 1) * per_row]) for r in range(2)])


def main():
    use_style(family="serif")
    recs = {r["method"]: r for r in map(json.loads, open(os.path.join(CASE, "records.jsonl")))}
    m, inner = 0.03, COLUMN - 0.06
    grid_h = [inner * 2 * FRAME_H / FRAME_W / k for _, _, k in ROWS]
    q_h, head_h, ans_h, gap = 0.36, 0.16, 0.16, 0.06
    height = m + q_h + sum(gap + head_h + g + ans_h for g in grid_h) + m

    fig = plt.figure(figsize=(COLUMN, height))
    fx, fy = (lambda v: v / COLUMN), (lambda v: v / height)

    def rect(y_top, h, fill="none", ls="-", lw=0.6, ec=RULE):
        fig.patches.append(Rectangle((fx(m), fy(y_top - h)), fx(inner), fy(h), transform=fig.transFigure,
                                     fc=fill, ec=ec, lw=lw, ls=ls))

    y = height - m
    # Question: dashed box, label in gray italics as in the Q-Frame and Video-RAG figures
    rect(y, q_h, ls=(0, (3, 2)), lw=0.8, ec="#8C8C8C")
    x0 = m + 0.07
    fig.text(fx(x0), fy(y - 0.12), "Question:", fontsize=7, style="italic", color=GRAY, va="center")
    fig.text(fx(x0 + 0.48), fy(y - 0.12), QUESTION, fontsize=7, color=INK, va="center")
    for k, opt in enumerate(OPTIONS):
        fig.text(fx(x0 + 0.48 + k * 0.62), fy(y - 0.255), opt, fontsize=7, color=INK, va="center")
    y -= q_h

    for (method, header, per_row), g in zip(ROWS, grid_h):
        rec = recs[method]
        ours = method != "naive"
        y -= gap
        # Header band, full width, joined to the frames below
        rect(y, head_h, fill=BAND)
        fig.text(0.5, fy(y - head_h / 2), header, ha="center", va="center", fontsize=7.5, weight="bold", color=INK)
        y -= head_h
        ax = fig.add_axes([fx(m), fy(y - g), fx(inner), fy(g)])
        ax.imshow(mosaic(rec, per_row, ours), interpolation="none", aspect="auto")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6), s.set_color(RULE)
        y -= g
        # Answer row, full width, joined to the frames above
        pred = max(rec["letter_logprobs"], key=rec["letter_logprobs"].get)
        ok = pred == GOLD
        rect(y, ans_h)
        yc = fy(y - ans_h / 2)
        t = fig.text(0, yc, f"Answer: {OPTIONS['ABCD'.index(pred)]}", ha="left", va="center_baseline", fontsize=7,
                     color=INK)
        k = fig.text(0, yc, "\u2714" if ok else "\u2718", ha="left", va="center_baseline", fontsize=6.5, color=GREEN if ok else RED, fontfamily="DejaVu Sans")
        r = fig.canvas.get_renderer()
        wt, wk = (a.get_window_extent(r).width / fig.bbox.width for a in (t, k))
        space = fx(0.05)
        left = 0.5 - (wt + space + wk) / 2  # centre text + mark together
        t.set_x(left), k.set_x(left + wt + space)
        y -= ans_h

    print(save(fig, os.path.join(HERE, "case_study")))


if __name__ == "__main__":
    main()
