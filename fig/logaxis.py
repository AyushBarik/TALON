"""One convention for every base-10 log axis in the paper figures.

Labelled ticks follow the 1-2-5 series (..., 0.5, 1, 2, 5, 10, 20, 50, 100, ...), written as plain numbers. Every other
integer multiple of a power of ten (3, 4, 6, 7, 8, 9) gets a short unlabelled tick, so no position of the scale is
skipped. The axis starts and ends at a labelled tick, so both limits must be 1-2-5 values.
(Token counts per frame use base 2 instead: ticks at powers of two, no minor ticks. See value_pair.py.)
"""
import math

import matplotlib.ticker as mticker


def _series(lo, hi, mantissas):
    out = []
    for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 2):
        for m in mantissas:
            v = m * 10.0 ** e
            if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9):
                out.append(float(f"{v:.6g}"))
    return out


def log_axis(ax, lo, hi, which="x"):
    """Log scale from lo to hi (both 1-2-5 values) with 1-2-5 labels and complete minor ticks."""
    major = _series(lo, hi, (1, 2, 5))
    assert major and major[0] == float(f"{lo:.6g}") and major[-1] == float(f"{hi:.6g}"), "limits must be 1-2-5 values"
    minor = _series(lo, hi, (3, 4, 6, 7, 8, 9))
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis.set_major_locator(mticker.FixedLocator(major))
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(mticker.FixedLocator(minor))
    axis.set_minor_formatter(mticker.NullFormatter())
    (ax.set_xlim if which == "x" else ax.set_ylim)(lo, hi)
