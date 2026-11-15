"""Multi-turn prefill, two panels in ONE column (300 MT-Video-Bench conversations, 12,288 tokens per turn).

(a) New prefill tokens per turn: the tokens a turn needs (allocated frame tokens + question tokens) that the KV cache
    does not already hold, mean over the 300 conversations, for turns 1 to 5 (every conversation has at least 5 turns).
(b) Judged accuracy per turn (multiturn_turn_accuracy.csv): mean judge score of the turns that MT-Video-Bench scores,
    from the Table 3 runs (200 conversations, 6,144 tokens per turn; turns 1 to 5 have 27, 70, 78, 92 and 108 judged
    turns). NOTE: (a) is at 12,288 tokens on 300 conversations, (b) at 6,144 on 200. Top-K has no saved per-turn
    verdicts yet, so it is drawn only in (a).
Only Top-K and Lembas (and its ablation) are drawn (author decision). "w/o carried allocation" in (a) follows from its
definition: the whole prompt is prefilled again at every turn, so every token a turn needs is new.
Every axis starts at its first tick and ends at its last tick.

Regenerate:  uv run --no-project --with matplotlib --with numpy python fig/overlap_pair.py
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from methods import METHODS, NO_CARRY, NO_CARRY_STYLE, is_ours
from orx_figstyle import COLUMN, panel_labels, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = [0, 25, 50, 75, 100]
TURNS = [1, 2, 3, 4, 5]
SHOWN = ("Top-K", "Lembas (Ours)")


def cdf(ax, values, **style):
    v = np.sort(np.asarray(values, float))
    y = 100 * np.arange(1, len(v) + 1) / len(v)
    return ax.step(np.concatenate([[0], v, [100]]), np.concatenate([[0], y, [100]]), where="post", **style)[0]


def main():
    use_style()
    plt.rcParams.update({"xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.labelsize": 6.5})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COLUMN, COLUMN * 0.6), layout="constrained")
    turns = json.load(open(os.path.join(HERE, "overlap_turns.json")))
    new, reuse = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    rows = [dict(r, method="Lembas (Ours)") for r in turns] + list(csv.DictReader(open(os.path.join(HERE, "baseline_reuse_turns.csv"))))
    for r in rows:
        t, need, got = int(r["turn"]), int(r["need"]), int(r["reused"])
        new[r["method"]][t].append(need - got)
        if t >= 2:
            reuse[r["method"]].append(100 * got / need)
    handles = {}
    for method, (color, marker) in METHODS.items():
        if method in SHOWN and method in new:
            z = 3 if is_ours(method) else 2
            handles[method] = ax.plot(TURNS, [np.mean(new[method][t]) / 1000 for t in TURNS], color=color, marker=marker,
                                      ms=2.6, lw=1.0, zorder=z)[0]
            cdf(bx, reuse[method], color=color, lw=1.0, zorder=z)
    need = defaultdict(list)
    for r in turns:
        need[int(r["turn"])].append(int(r["need"]))
    handles[NO_CARRY] = ax.plot(TURNS, [np.mean(need[t]) / 1000 for t in TURNS], ms=2.6, lw=1.0, zorder=2.5, **NO_CARRY_STYLE)[0]
    acc = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(HERE, "multiturn_turn_accuracy.csv"))):
        acc[r["method"]][int(r["turn"])] = float(r["accuracy"])
    for method, style in [(NO_CARRY, NO_CARRY_STYLE)] + [(m, dict(color=c, marker=k)) for m, (c, k) in METHODS.items()]:
        if method in acc:
            bx.plot(TURNS, [acc[method][t] for t in TURNS], ms=2.6, lw=1.0, zorder=3 if is_ours(method) else 2.5, **style)
    ax.set_xlim(1, 5); ax.set_xticks(TURNS); ax.set_ylim(0, 15); ax.set_yticks([0, 5, 10, 15])
    ax.set_yticklabels(["0", "5k", "10k", "15k"])
    ax.set_xlabel("Turn"); ax.set_ylabel("Tokens prefilled", labelpad=1)
    bx.set_xlim(1, 5); bx.set_xticks(TURNS); bx.set_ylim(40, 70); bx.set_yticks([40, 50, 60, 70])
    bx.set_xlabel("Turn"); bx.set_ylabel("MTVB accuracy (%)", labelpad=1)
    order = [m for m in METHODS if m in handles] + [NO_CARRY]
    fig.legend([handles[m] for m in order], order, loc="outside upper center", ncol=3, frameon=False, fontsize=5.5,
               columnspacing=0.8, handlelength=1.6, handletextpad=0.4)
    panel_labels([ax, bx])
    print(save(fig, os.path.join(HERE, "overlap_pair")))


if __name__ == "__main__":
    main()
