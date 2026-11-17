"""Token value and the allocation it leads to: TWO separate half-column figures, set side by side in one column
(fig/value_half.pdf = Figure "value", fig/alloc_case.pdf = Figure "alloc"; no panel letters, each has its own caption).

(a) Normalized accuracy (0 = no visual tokens, 1 = full frame) against the tokens of each frame, with four curve families fitted to the measured means (as in
    value_curve.py; data fig/value_curve.csv, held-out LVBench calibration, Qwen3-VL-8B, 280 questions).
(b) The allocation of one question over the time of its video, in one plot with two y-axes: tokens of every candidate
    frame (left axis) and the transcript tokens shown (right axis, summed per frame interval). Data: fig/alloc_case.json, Video-MME
    question 378-1 (video WOGaS19VZ8Q, the case-study question), Qwen3-VL-8B, 12,288-token budget, 128 candidate
    frames, 247 transcript cues. It is a CPU replay of the allocation of the Table 1 run; the replay reproduces the
    logged frames, their tokens and the number of cues shown (195) exactly.
Every axis starts at its first tick and ends at its last tick.

Regenerate:  uv run --no-project --with matplotlib --with numpy --with scipy python fig/value_pair.py
"""
import csv
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.optimize import curve_fit

from logaxis import log_axis
from orx_figstyle import COLUMN, save, use_style
from value_curve import FAMILIES, ZORDER

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME, TEXT = "#4E79A7", "#F28E2B"  # the frame and text colours of the overview figure (fig/overview_palette.tex)


def main():
    use_style()
    plt.rcParams.update({"xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8})
    W1, W2, H = COLUMN * 0.5, COLUMN * 0.5, COLUMN * 0.58  # both figures have one size (the paper sets them at equal widths)
    fig, ax = plt.subplots(figsize=(W1, H), layout="constrained")
    fig2, fx = plt.subplots(figsize=(W2, H), layout="constrained")
    tx = fx.twinx()

    rows = list(csv.DictReader(open(os.path.join(HERE, "value_curve.csv"))))
    b = np.array([float(r["tokens_per_frame"]) for r in rows])
    # Normalized accuracy: the probability of the correct answer, rescaled so that 0 is the level without visual
    # tokens and 1 is the full frame (column p_gold_normalized).
    acc = np.array([float(r["p_gold_normalized"]) for r in rows])
    grid = np.exp(np.linspace(np.log(3.5), np.log(1000), 300))
    start = {"Rational": [0, 1, 30], "Sigmoid": [0, 1, 30], "Exponential": [0, 1, 50], "Log": [0, 0.2]}
    for name, (f, _, color, ls) in FAMILIES.items():
        p, _ = curve_fit(f, b, acc, p0=start[name], maxfev=20000)
        print(f"{name:12s} RMSE={np.sqrt(np.mean((f(b, *p) - acc) ** 2)):.3f}")
        ax.plot(grid, f(grid, *p), color=color, ls=ls, lw=1.0, label=name, zorder=ZORDER[name])
    ax.plot(b, acc, "o", color="black", ms=2.2, zorder=3)
    log_axis(ax, 2, 1000, label=(1,))  # labels at 2, 10, 100, 1k; short ticks at every other step (fig/logaxis.py)
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Tokens per frame"); ax.set_ylabel("Normalized accuracy", labelpad=1)
    # Legends sit above the axes, two rows in both figures, so that the two plot areas keep one size.
    fig.legend(loc="outside upper center", ncol=2, frameon=False, fontsize=7, labelspacing=0.2, handlelength=1.2,
               columnspacing=0.8, handletextpad=0.4, borderaxespad=0, borderpad=0)

    c = json.load(open(os.path.join(HERE, "alloc_case.json")))
    t = np.array(c["frame_times"]) / 60
    step = float(np.median(np.diff(t)))
    edges = np.concatenate([t, [t[-1] + step]])
    mid = np.array([(q["start"] + q["end"]) / 120 for q in c["cues"]])
    tok = np.array([q["tokens"] for q in c["cues"]], float)
    shown = np.array([q["shown"] for q in c["cues"]])
    kept, _ = np.histogram(mid[shown], edges, weights=tok[shown])
    # One plot, two y-axes: transcript bars behind (right axis), frame bars in front (left axis).
    tx.bar(t, kept, width=step, align="edge", color=TEXT, lw=0, label="Transcript")
    fx.bar(t, c["frame_tokens"], width=step, align="edge", color=FRAME, lw=0, label="Frames")
    fx.set_zorder(tx.get_zorder() + 1); fx.patch.set_visible(False)
    fx.set_ylim(0, 800); fx.set_yticks([0, 200, 400, 600, 800]); fx.set_ylabel("Frame tokens", labelpad=1)
    tx.set_ylim(0, 80); tx.set_yticks([0, 20, 40, 60, 80]); tx.set_ylabel("Transcript tokens", labelpad=2)
    tx.grid(False); tx.spines["right"].set_visible(True)
    fx.set_xlim(0, 9); fx.set_xticks([0, 3, 6, 9]); fx.set_xlabel("Video time (min)")
    h = fx.get_legend_handles_labels()[0] + tx.get_legend_handles_labels()[0]
    fig2.legend(h, ["Frames", "Transcript"], loc="outside upper center", ncol=1, frameon=False, fontsize=7,
                handlelength=0.9, handleheight=0.7, labelspacing=0.2, handletextpad=0.4, borderaxespad=0, borderpad=0)
    print(save(fig, os.path.join(HERE, "value_half")))
    print(save(fig2, os.path.join(HERE, "alloc_case")))
    print("max frame", max(c["frame_tokens"]), "max text bin", kept.max(), "kept", kept.sum())


if __name__ == "__main__":
    main()
