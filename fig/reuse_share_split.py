"""The reuse pair as TWO separate half-column figures, each with its own legend (no panel letters), so the legend of
each figure lists exactly the lines it draws. Same data as reuse_share_pair.py: 300 MT-Video-Bench conversations,
12,288 tokens per turn, turns 1 to 5 (overlap_turns.json for Lembas, baseline_reuse_turns.csv for Top-K).

turn_seen.pdf     Share of the frames a turn retrieves that an earlier turn of the same conversation already
    retrieved, pooled over the conversations. Top-K only (author decision): every Top-K frame has the same token
    count, so the share of frame tokens is the share of frames.
turn_prefill.pdf  Tokens prefilled per turn (tokens needed that the KV cache does not hold), mean over the
    conversations. "Lembas w/o incremental allocation" follows from its definition: the whole prompt is prefilled
    again at every turn, so every token a turn needs is new.
Every axis starts at its first tick and ends at its last tick.

Regenerate:  uv run --no-project --with matplotlib --with numpy python reuse_share_split.py
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from methods import METHODS, is_ours
from orx_figstyle import COLUMN, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
TURNS = [1, 2, 3, 4, 5]
SHOWN = ("Top-K", "Lembas")
NO_INC = "Lembas w/o\nincremental allocation"
NO_INC_COLOR = "#0B2A4A"  # dark navy: a Lembas variant, same hue family, solid like every other line
RENAME = {"Top-K (w/ pruning)": "Top-K"}
W, H = COLUMN * 0.5, COLUMN * 0.5
LEG = dict(frameon=False, fontsize=5, handlelength=1.6, handletextpad=0.4, labelspacing=0.35, borderaxespad=0.2)


def main():
    use_style()
    plt.rcParams.update({"xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.labelsize": 6.5})
    rows = [dict(r, method="Lembas") for r in json.load(open(os.path.join(HERE, "overlap_turns.json")))]
    rows += list(csv.DictReader(open(os.path.join(HERE, "baseline_reuse_turns.csv"))))
    got, need, n, qtok = (defaultdict(lambda: defaultdict(int)) for _ in range(4))
    for r in rows:
        m, t = RENAME.get(r["method"], r["method"]), int(r["turn"])
        got[m][t] += int(r["reused"]); need[m][t] += int(r["need"]); n[m][t] += 1; qtok[m][t] += int(r["qtok"])
    shown = [(m, c, k) for m, (c, k) in METHODS.items() if m in SHOWN]

    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    for m, c, k in shown:
        if m != "Top-K":
            continue
        ax.plot(TURNS, [100 * got[m][t] / (need[m][t] - qtok[m][t]) for t in TURNS], color=c, marker=k, ms=2.6, lw=1.0,
                zorder=2, clip_on=False, label=m)
    ax.set_xlim(1, 5); ax.set_xticks(TURNS); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Turn"); ax.set_ylabel("Frames seen\nin earlier turns (%)", labelpad=1)
    ax.legend(loc="lower right", **LEG)
    print(save(fig, os.path.join(HERE, "turn_seen")))

    fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
    for m, c, k in shown:
        ax.plot(TURNS, [(need[m][t] - got[m][t]) / n[m][t] / 1000 for t in TURNS], color=c, marker=k, ms=2.6, lw=1.0,
                zorder=3 if is_ours(m) else 2, clip_on=False, label=m)
    m = "Lembas"
    ax.plot(TURNS, [need[m][t] / n[m][t] / 1000 for t in TURNS], color=NO_INC_COLOR, marker="v", ms=2.8, lw=1.0,
            zorder=2.5, clip_on=False, label=NO_INC)
    ax.set_xlim(1, 5); ax.set_xticks(TURNS); ax.set_ylim(0, 15); ax.set_yticks([0, 5, 10, 15])
    ax.set_yticklabels(["0", "5k", "10k", "15k"])
    ax.set_xlabel("Turn"); ax.set_ylabel("Tokens prefilled", labelpad=1)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), **LEG)
    print(save(fig, os.path.join(HERE, "turn_prefill")))


if __name__ == "__main__":
    main()
