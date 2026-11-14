"""Method names and styles shared by every results figure, so a method looks the same everywhere.

Shape and hue both carry the method, so the figures survive greyscale and colorblind readers. TALON comes last so it
is drawn on top.
"""
from orx_figstyle import BASELINE, PALETTE

METHODS = {
    "Uniform sampling": (PALETTE["black"], "x"),
    "Top-K": (BASELINE, "v"),  # top-K frames at native resolution, no pruner (multi-turn figure only)
    "Top-K (w/ pruning)": (PALETTE["cyan"], "P"),
    "Q-Frame": (PALETTE["orange"], "s"),
    "QuoTA": (PALETTE["green"], "D"),
    "FOCUS": (PALETTE["red"], "^"),
    "AdaAlloc": (PALETTE["purple"], "X"),
    "TALON (Ours)": (PALETTE["blue"], "o"),
}


def is_ours(method):
    return method.startswith("TALON")
