"""Support / resistance level detection from swing pivots."""

from __future__ import annotations

import pandas as pd

from strategies import _ok

def support_resistance(df: pd.DataFrame, price: float, n_levels: int = 3, window: int = 5):
    """Cluster swing highs / lows into price levels.

    A pivot is a bar whose high (low) is the highest (lowest) within +-window bars.  Pivots
    closer than ~0.5 ATR are merged into one level; the level's strength is the number of
    touches.  Returns (supports, resistances) as lists of (level, touches), strongest first,
    ties broken by proximity to the current price."""
    if len(df) < 2 * window + 3:
        return [], []
    highs, lows = df["High"].to_numpy(), df["Low"].to_numpy()
    n = len(df)
    pivots = []
    for i in range(window, n - window):
        seg_h, seg_l = highs[i - window:i + window + 1], lows[i - window:i + window + 1]
        if highs[i] >= seg_h.max():
            pivots.append(highs[i])
        if lows[i] <= seg_l.min():
            pivots.append(lows[i])
    if not pivots:
        return [], []
    a = float(df["ATR"].iloc[-1]) if _ok(df["ATR"].iloc[-1]) else price * 0.01
    tol = max(0.5 * a, price * 0.002)
    pivots.sort()
    clusters: list[list[float]] = []  # [sum, count]
    for pv in pivots:
        if clusters and abs(pv - clusters[-1][0] / clusters[-1][1]) <= tol:
            clusters[-1][0] += pv
            clusters[-1][1] += 1
        else:
            clusters.append([pv, 1])
    levels = [(tot / cnt, int(cnt)) for tot, cnt in clusters]

    def pick(cands, key):
        # Nearest levels first, preferring ones confirmed by >= 2 touches; single-touch
        # pivots only fill in when there are not enough confirmed levels.
        strong = sorted([lv for lv in cands if lv[1] >= 2], key=key)
        weak = sorted([lv for lv in cands if lv[1] < 2], key=key)
        return (strong + weak)[:n_levels]

    sup = pick([lv for lv in levels if lv[0] < price], key=lambda lv: price - lv[0])
    res = pick([lv for lv in levels if lv[0] > price], key=lambda lv: lv[0] - price)
    return sorted(sup, key=lambda lv: -lv[0]), sorted(res, key=lambda lv: lv[0])
