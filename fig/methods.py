"""Method names and styles shared by every results figure, so a method looks the same everywhere.

Shape and hue both carry the method, so the figures survive greyscale and colorblind readers. Lembas comes last so it
is drawn on top.
"""
from orx_figstyle import BASELINE, PALETTE

METHODS = {
    "Uniform sampling": (PALETTE["black"], "x"),
    "Top-K": (PALETTE["cyan"], "P"),
    "Q-Frame": (PALETTE["orange"], "s"),
    "QuoTA": (PALETTE["green"], "D"),
    "FOCUS": (PALETTE["red"], "^"),
    "AdaAlloc": (PALETTE["purple"], "X"),
    "Lembas (Ours)": (PALETTE["blue"], "o"),
}


# Ablation of the multi-turn design, drawn like every other method (solid line, filled marker), only in the
# multi-turn panels. Grey, because every palette hue already names a method.
NO_CARRY = "w/o carried allocation"
NO_CARRY_STYLE = dict(color=BASELINE, marker="v")


def is_ours(method):
    return method.startswith("Lembas")
