"""Supplementary sensitivity figures, one per column: attention weight omega (MT-Video-Bench), subtitle weight beta
(Video-MME w/ subtitles), relevance temperature T (Video-MME w/ subtitles, 300 questions; the T = 0.63 point is the
default run on the same questions, sens_default_T.json). Data: sens_weights.json (artifacts sensitivity/build_data.py), sens_sweeps.json
(artifacts supplement-evals/sens.py).
Regenerate: uv run --no-project --with matplotlib --with numpy python fig/sens_split.py"""
import json
from pathlib import Path

import orx_figstyle as fs

here = Path(__file__).parent
w = json.load(open(here / "sens_weights.json"))
s = json.load(open(here / "sens_sweeps.json"))
fs.use_style()
OFF = {
    "omega": {0.0: (0, 6, "center"), 0.25: (0, 6, "center"), 0.5: (0, 6, "center"), 0.75: (0, -11, "center"), 1.0: (0, 6, "center")},
    "beta": {0.0: (0, 6, "center"), 0.25: (6, -3, "left"), 0.5: (6, -3, "left"), 1.0: (0, 6, "center"), 2.0: (-6, -3, "right")},
}
spec = {"omega": (r"\omega", "blue", "TFLOPs per conversation", range(260, 321, 20), (260, 320), range(56, 61), (56, 60)),
        "beta": (r"\beta", "orange", "TFLOPs per question", range(231, 235), (231, 234), range(65, 86, 5), (65, 85))}
for key, (sym, col, xl, xt, xlim, yt, ylim) in spec.items():
    fig, ax = fs.figure(width=fs.COLUMN, ratio=0.62)
    c = fs.PALETTE[col]
    for r in w[key]:
        ax.plot([r["tflops"]], [r["acc"]], marker="o", ms=4.5, color=c, ls="none", clip_on=False, zorder=4)
        dx, dy, ha = OFF[key][r["value"]]
        ax.annotate(rf"${sym}={r['value']:g}$", (r["tflops"], r["acc"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="bottom" if dy > 0 else ("top" if dx == 0 else "center"), fontsize=7, zorder=5)
    ax.set_xlabel(xl); ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(list(xt)); ax.set_xlim(*xlim); ax.set_yticks(list(yt)); ax.set_ylim(*ylim)
    print(fs.save(fig, str(here / f"sens_{key}")))

dT = json.load(open(here / "sens_default_T.json"))  # default run (T = 0.63) on the same 300 questions
pts = [(0.3, s["T0.3"]["tflops"], s["T0.3"]["acc"]), (0.63, dT["tflops"], dT["acc"]),
       (1.0, s["T1"]["tflops"], s["T1"]["acc"]), (2.0, s["T2"]["tflops"], s["T2"]["acc"])]
OFF_T = {0.3: (0, 6, "center"), 0.63: (0, 6, "center"), 1.0: (6, -3, "left"), 2.0: (0, -11, "center")}
fig, ax = fs.figure(width=fs.COLUMN, ratio=0.62)
c = fs.PALETTE["green"]
for t, tf, a in pts:
    ax.plot([tf], [a], marker="o", ms=4.5, color=c, ls="none", clip_on=False, zorder=4)
    dx, dy, ha = OFF_T[t]
    ax.annotate(rf"$T={t:g}$", (tf, a), xytext=(dx, dy), textcoords="offset points", ha=ha,
                va="bottom" if dy > 0 else ("top" if dx == 0 else "center"), fontsize=7, zorder=5)
ax.set_xlabel("TFLOPs per question"); ax.set_ylabel("Accuracy (%)")
ax.set_xticks(range(226, 237, 2)); ax.set_xlim(226, 236)
ax.set_yticks(range(74, 81, 2)); ax.set_ylim(74, 80)
print(fs.save(fig, str(here / "sens_T")))
