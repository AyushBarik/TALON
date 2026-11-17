"""Sensitivity to the two hand-set weights, as TWO separate half-column figures: accuracy against prefill compute,
one labelled dot per value (sensitivity.csv).

sens_beta.pdf   Subtitle weight beta on Video-MME with subtitles (150 questions, 12,288 tokens); TFLOPs per question.
sens_omega.pdf  Attention trust omega on MT-Video-Bench (200 conversations, 6,144 tokens per turn); TFLOPs per
    conversation.
Every axis starts at its first tick and ends at its last tick.

Regenerate:  uv run --no-project --with matplotlib --with numpy python sensitivity.py
"""
import csv
import os

import matplotlib.pyplot as plt

from methods import METHODS
from orx_figstyle import COLUMN, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = COLUMN * 0.5, COLUMN * 0.5
COLOR, MARK = METHODS["Lembas"]
# per panel: symbol, x ticks, y ticks, label offset in points (dx, dy, ha, va) per value
PANELS = {
    "beta": dict(sym=r"\beta", xt=[231, 232, 233, 234], yt=[65, 70, 75, 80, 85], off={
        "0": (0, 4, "center", "bottom"), "0.25": (4, -3, "left", "center"), "0.5": (4, 2, "left", "center"),
        "1": (0, 4, "center", "bottom"), "2": (-4, -2, "right", "center")}),
    "omega": dict(sym=r"\omega", xt=[260, 280, 300, 320], yt=[56, 57, 58, 59, 60], off={
        "0": (0, 4, "center", "bottom"), "0.25": (0, 4, "center", "bottom"), "0.5": (-4, 1, "right", "center"),
        "0.75": (0, -4, "center", "top"), "1": (4, -1, "left", "center")}),
}


def main():
    use_style()
    plt.rcParams.update({"xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.labelsize": 6.5})
    rows = list(csv.DictReader(open(os.path.join(HERE, "sensitivity.csv"))))
    for name, p in PANELS.items():
        fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
        for r in rows:
            if r["parameter"] != name:
                continue
            x, y = float(r["tflops"]), float(r["accuracy"])
            ax.plot([x], [y], color=COLOR, marker=MARK, ms=2.8, ls="none", zorder=3, clip_on=False)
            dx, dy, ha, va = p["off"][r["value"]]
            ax.annotate(rf"${p['sym']}={r['value']}$", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                        fontsize=5, zorder=4)
        ax.set_xlim(p["xt"][0], p["xt"][-1]); ax.set_xticks(p["xt"])
        ax.set_ylim(p["yt"][0], p["yt"][-1]); ax.set_yticks(p["yt"])
        ax.set_xlabel("TFLOPs"); ax.set_ylabel("Accuracy (%)", labelpad=1)
        print(save(fig, os.path.join(HERE, f"sens_{name}")))


if __name__ == "__main__":
    main()
