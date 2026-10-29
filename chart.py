"""Native Tk canvas charting.

Everything is drawn as vector canvas items (lines, rectangles, text) rather than a
bitmap, so the chart stays crisp on Retina / HiDPI displays and resizes freely."""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

import numpy as np

from config import BG, BLUE, FG, GREEN, MUTED, ORANGE, PANEL, PURPLE, RED, YELLOW, LONG_TERM, SHORT_TERM
from levels import support_resistance

if TYPE_CHECKING:  # avoid a circular import at runtime
    from app import StockMonitorApp

GRID = "#3a3a50"
AXIS = "#44445c"


def blend(c1: str, c2: str, t: float) -> str:
    """Mix two '#rrggbb' colours; t=0 -> c1, t=1 -> c2.  Tk has no alpha, so a
    translucent look is faked by blending the colour toward the background."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


BB_FILL = blend(PANEL, "#d8c8ff", 0.10)   # ~10% lavender over the panel: barely-there shading
BB_EDGE = blend(PANEL, "#d8c8ff", 0.45)   # muted band outlines
FONT_S = ("Helvetica", 9)
FONT_T = ("Helvetica", 11, "bold")


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return []
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if (hi - lo) / step <= n:
            break
    start = np.ceil(lo / step) * step
    return [float(v) for v in np.arange(start, hi + step * 1e-6, step)]


def fmt_num(v: float) -> str:
    av = abs(v)
    if av < 1e-9:
        return "0"
    if av >= 1e9:
        return f"{v / 1e9:.1f}B"
    if av >= 1e6:
        return f"{v / 1e6:.1f}M"
    if av >= 1e4:
        return f"{v / 1e3:.0f}K"
    if av >= 1:
        return f"{v:,.2f}"
    return f"{v:.3f}"


# Panels the user can switch on / off in the View menu.  Price is always shown.
PANELS = [
    ("price", "Price", 3.0, True),
    ("volume", "Volume", 1.0, True),
    ("rsi", "RSI (14)", 1.0, True),
    ("macd", "MACD (12, 26, 9)", 1.0, True),
    ("stoch", "Stochastic (14, 3, 3)", 1.0, False),
    ("adx", "ADX / DMI (14)", 1.0, False),
    ("cci", "CCI (20)", 1.0, False),
    ("obv", "On-balance volume", 1.0, False),
]
OVERLAYS = [
    ("sma", "Moving averages (SMA fast / slow)", True),
    ("bb", "Bollinger bands", True),
    ("supertrend", "Supertrend line", False),
    ("psar", "Parabolic SAR dots", False),
    ("vwap", "VWAP", False),
    ("levels", "Strategy stop / target", True),
    ("sr", "Support / resistance levels", True),
]


class _Panel:
    """Pixel mapping for one horizontal band of the chart."""

    def __init__(self, cv: "ChartCanvas", x0, x1, y0, y1, n, vmin, vmax):
        self.cv, self.x0, self.x1, self.y0, self.y1, self.n = cv, x0, x1, y0, y1, n
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = (vmin if np.isfinite(vmin) else 0.0), (vmin if np.isfinite(vmin) else 0.0) + 1.0
        self.vmin, self.vmax = vmin, vmax
        self.bar_w = (x1 - x0) / max(n, 1)
        self._label_ys: dict[str, list[float]] = {"left": [], "right": []}

    def px(self, i: float) -> float:
        return self.x0 + (i + 0.5) * self.bar_w

    def py(self, v: float) -> float:
        return self.y1 - (v - self.vmin) / (self.vmax - self.vmin) * (self.y1 - self.y0)

    def inside(self, v: float) -> bool:
        return self.vmin <= v <= self.vmax

    # ---- primitives
    def frame(self, title: str | None = None, yticks: int = 5, fmt=fmt_num):
        c = self.cv
        c.create_rectangle(self.x0, self.y0, self.x1, self.y1, outline=AXIS, width=1)
        for t in nice_ticks(self.vmin, self.vmax, yticks):
            y = self.py(t)
            if self.y0 + 2 < y < self.y1 - 2:
                c.create_line(self.x0, y, self.x1, y, fill=GRID, width=1)
            c.create_text(self.x1 + 6, y, text=fmt(t), fill=MUTED, font=FONT_S, anchor="w")
        if title:
            c.create_text(self.x0 + 8, self.y0 + 6, text=title, fill=MUTED, font=FONT_S, anchor="nw")

    def line(self, series, color: str, width: float = 1.0, dash=None, smooth=False):
        vals = np.asarray(series, dtype=float)
        pts: list[float] = []
        for i, v in enumerate(vals):
            if np.isfinite(v):
                pts += [self.px(i), self.py(v)]
            elif pts:
                self._flush_line(pts, color, width, dash)
                pts = []
        self._flush_line(pts, color, width, dash)

    def _flush_line(self, pts, color, width, dash):
        if len(pts) >= 4:
            kw = {"dash": dash} if dash else {}
            self.cv.create_line(*pts, fill=color, width=width, **kw)

    def bars(self, series, colors, base: float = 0.0, frac: float = 0.7):
        vals = np.asarray(series, dtype=float)
        hw = max(0.5, self.bar_w * frac / 2)
        yb = self.py(max(min(base, self.vmax), self.vmin))
        for i, v in enumerate(vals):
            if not np.isfinite(v):
                continue
            x = self.px(i)
            self.cv.create_rectangle(x - hw, yb, x + hw, self.py(v), fill=colors[i], outline="")

    def hline(self, v: float, color: str, dash=None, label: str | None = None, side: str = "right", width: float = 1.0):
        if not self.inside(v):
            return
        y = self.py(v)
        kw = {"dash": dash} if dash else {}
        self.cv.create_line(self.x0, y, self.x1, y, fill=color, width=width, **kw)
        if label:
            used = self._label_ys[side]
            ty, anchor = y - 2, "s"          # default: just above the line
            if any(abs(ty - u) < 12 for u in used):
                ty, anchor = y + 2, "n"      # collides: put it just below instead
            if any(abs(ty - u) < 12 for u in used):
                return                       # still crowded: line only, no label
            used.append(ty)
            if side == "right":
                self.cv.create_text(self.x1 - 4, ty, text=label, fill=color, font=FONT_S, anchor=anchor + "e")
            else:
                self.cv.create_text(self.x0 + 4, ty, text=label, fill=color, font=FONT_S, anchor=anchor + "w")

    def price_tag(self, v: float, color: str):
        """Boxed value on the right axis (used for the live price)."""
        if not self.inside(v):
            return
        y = self.py(v)
        txt = f"{v:,.2f}"
        self.cv.create_rectangle(self.x1 + 3, y - 8, self.x1 + 72, y + 8, fill=color, outline="")
        self.cv.create_text(self.x1 + 6, y, text=txt, fill=BG, font=("Helvetica", 9, "bold"), anchor="w")


class ChartCanvas(tk.Canvas):
    LEFT, RIGHT, TOP, BOTTOM, GAP = 8, 80, 8, 28, 10

    def __init__(self, master, app: "StockMonitorApp"):
        super().__init__(master, bg=PANEL, highlightthickness=0)
        self.app = app
        self._job = None
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, _event=None):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(40, self.redraw)

    # ---- main entry ---------------------------------------------------------
    def redraw(self):
        self._job = None
        self.delete("all")
        app = self.app
        s = app.snapshot
        W, H = self.winfo_width(), self.winfo_height()
        if s is None or W < 50 or H < 50:
            self.create_text(W / 2, H / 2, text="Load a ticker to see the chart", fill=MUTED, font=FONT_T)
            return
        short = app.view.get() == "short"
        df = s.short_df if short else s.long_df
        cfg = SHORT_TERM if short else LONG_TERM
        sig = app.short_sig if short else app.long_sig
        panels = [key for key, _n, _w, _d in PANELS if key == "price" or app.panel_vars[key].get()]
        weights = {key: w for key, _n, w, _d in PANELS}

        x0, x1 = self.LEFT, W - self.RIGHT
        avail = H - self.TOP - self.BOTTOM - self.GAP * (len(panels) - 1)
        total_w = sum(weights[k] for k in panels)
        y = self.TOP
        n = len(df)
        idx = df.index
        self._x_axis(x0, x1, H - self.BOTTOM, n, idx, short)
        for key in panels:
            h = avail * weights[key] / total_w
            y0, y1 = y, y + h
            y = y1 + self.GAP
            getattr(self, f"_draw_{key}")(df, cfg, sig, s, x0, x1, y0, y1, short)

    # ---- axes -------------------------------------------------------------
    def _x_axis(self, x0, x1, ybase, n, idx, short):
        if n == 0:
            return
        k = max(2, int((x1 - x0) / 110))
        ticks = np.linspace(0, n - 1, k).round().astype(int)
        for j, i in enumerate(ticks):
            x = x0 + (i + 0.5) * (x1 - x0) / n
            ts = idx[i]
            lbl = ts.strftime("%m-%d %H:%M") if short else ts.strftime("%Y-%m-%d")
            self.create_line(x, self.TOP, x, ybase, fill=GRID, width=1)
            anchor = "nw" if j == 0 else "ne" if j == len(ticks) - 1 else "n"
            self.create_text(x, ybase + 6, text=lbl, fill=MUTED, font=FONT_S, anchor=anchor)

    @staticmethod
    def _range(*arrays, pad=0.04):
        vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays if a is not None])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0, 1.0
        lo, hi = float(vals.min()), float(vals.max())
        span = (hi - lo) or abs(hi) * 0.02 or 1.0
        return lo - span * pad, hi + span * pad

    # ---- panels -------------------------------------------------------------
    def _draw_price(self, df, cfg, sig, s, x0, x1, y0, y1, short):
        app = self.app
        ov = {k: app.overlay_vars[k].get() for k, _n, _d in OVERLAYS}
        candles = app.chart_style.get() == "candle"
        n = len(df)
        parts = [df["High"], df["Low"]] if candles else [df["Close"]]
        if ov["sma"]:
            parts += [df["SMA_fast"], df["SMA_slow"]]
        if ov["bb"]:
            parts += [df["BB_up"], df["BB_lo"]]
        if ov["supertrend"]:
            parts.append(df["ST"])
        if ov["vwap"]:
            parts.append(df["VWAP"])
        if ov["levels"] and sig is not None and sig.label != "NEUTRAL":
            parts.append(np.array([sig.stop, sig.target]))
        vmin, vmax = self._range(*parts)
        header = 42  # title + legend strip above the plot area
        p = _Panel(self, x0, x1, y0 + header, y1, n, vmin, vmax)
        p.frame(yticks=6, fmt=lambda v: f"{v:,.2f}")

        # Bollinger band fill + edges
        if ov["bb"]:
            up, lo = df["BB_up"].to_numpy(), df["BB_lo"].to_numpy()
            poly = []
            for i in range(n):
                if np.isfinite(up[i]) and np.isfinite(lo[i]):
                    poly.append((p.px(i), p.py(up[i]), p.py(lo[i])))
            if len(poly) >= 2:
                pts = [c for x, yu, _ in poly for c in (x, yu)] + [c for x, _, yl in reversed(poly) for c in (x, yl)]
                poly_id = self.create_polygon(*pts, fill=BB_FILL, outline="")
                self.tag_lower(poly_id)  # keep grid lines and everything else on top of it
            p.line(up, BB_EDGE, 1)
            p.line(lo, BB_EDGE, 1)

        if ov["sma"]:
            p.line(df["SMA_fast"], BLUE, 1.2)
            p.line(df["SMA_slow"], ORANGE, 1.2)
        if ov["vwap"]:
            p.line(df["VWAP"], YELLOW, 1, dash=(4, 3))
        if ov["supertrend"]:
            d = df["ST_dir"].to_numpy()
            st = df["ST"].to_numpy()
            p.line(np.where(d > 0, st, np.nan), GREEN, 1.4)
            p.line(np.where(d < 0, st, np.nan), RED, 1.4)
        if ov["psar"]:
            ps = df["PSAR"].to_numpy()
            cl = df["Close"].to_numpy()
            for i in range(n):
                if np.isfinite(ps[i]):
                    x, yy = p.px(i), p.py(ps[i])
                    self.create_oval(x - 1.5, yy - 1.5, x + 1.5, yy + 1.5, fill=GREEN if cl[i] > ps[i] else RED, outline="")

        # Price itself
        if candles:
            o, h, lo, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
            hw = max(1.0, p.bar_w * 0.35)
            for i in range(n):
                if not np.isfinite(c[i]):
                    continue
                col = GREEN if c[i] >= o[i] else RED
                x = p.px(i)
                self.create_line(x, p.py(h[i]), x, p.py(lo[i]), fill=col, width=1)
                top, bot = p.py(max(o[i], c[i])), p.py(min(o[i], c[i]))
                if bot - top < 1:
                    bot = top + 1
                self.create_rectangle(x - hw, top, x + hw, bot, fill=col, outline=col)
        else:
            p.line(df["Close"], FG, 1.5)

        # Support (solid) / resistance (dashed)
        if ov["sr"]:
            sup, res = support_resistance(df, s.price, n_levels=app.sr_count.get())
            for lv, cnt in sup:
                p.hline(lv, GREEN, label=f"S {lv:,.2f} ({cnt} touch{'es' if cnt > 1 else ''})", side="left", width=1.4)
            for lv, cnt in res:
                p.hline(lv, RED, dash=(6, 4), label=f"R {lv:,.2f} ({cnt} touch{'es' if cnt > 1 else ''})", side="left", width=1.4)
            if not res:
                self.create_text(x1 - 6, y0 + header + 6, text="no resistance above: at highs", fill=RED, font=FONT_S, anchor="ne")
            if not sup:
                self.create_text(x1 - 6, y1 - 6, text="no support below: at lows", fill=GREEN, font=FONT_S, anchor="se")

        # Strategy levels
        if ov["levels"] and sig is not None and sig.label != "NEUTRAL":
            p.hline(sig.stop, RED, dash=(2, 3), label="stop")
            p.hline(sig.target, GREEN, dash=(2, 3), label="target")

        # Live price marker
        col = GREEN if s.price >= s.prev_close else RED
        x, yy = p.px(n - 1), p.py(s.price)
        if p.inside(s.price):
            self.create_oval(x - 3.5, yy - 3.5, x + 3.5, yy + 3.5, fill=col, outline=BG)
            self.create_line(x0, yy, x1, yy, fill=col, width=1, dash=(1, 4))
        p.price_tag(s.price, col)

        # Title + legend
        label = sig.label if sig else "—"
        self.create_text(x0 + 2, y0 + 2, text=f"{s.ticker} · {cfg['label']} · {'candles' if candles else 'line'} · signal: {label}",
                         fill=FG, font=FONT_T, anchor="nw")
        lx = x0 + 2
        ly = y0 + 22
        legend = []
        if ov["sma"]:
            legend += [(f"SMA{cfg['fast']}", BLUE), (f"SMA{cfg['slow']}", ORANGE)]
        if ov["bb"]:
            legend.append(("Bollinger", BB_EDGE))
        if ov["vwap"]:
            legend.append(("VWAP", YELLOW))
        if ov["sr"]:
            legend += [("— support", GREEN), ("- - resistance", RED)]
        for text, colr in legend:
            t = self.create_text(lx, ly, text=text, fill=colr, font=FONT_S, anchor="nw")
            lx = self.bbox(t)[2] + 12

    def _simple(self, x0, x1, y0, y1, n, title, *ranges, pad=0.08):
        vmin, vmax = self._range(*ranges, pad=pad)
        p = _Panel(self, x0, x1, y0, y1, n, vmin, vmax)
        p.frame(title, yticks=3)
        return p

    def _draw_volume(self, df, cfg, sig, s, x0, x1, y0, y1, short):
        v = df["Volume"].to_numpy()
        p = _Panel(self, x0, x1, y0, y1, len(df), 0.0, float(np.nanmax(v)) * 1.08 if np.isfinite(np.nanmax(v)) else 1.0)
        p.frame("Volume", yticks=3)
        up = (df["Close"] >= df["Open"]).to_numpy()
        p.bars(v, [GREEN if u else RED for u in up])
        p.line(df["VOL_SMA"], YELLOW, 1)

    def _draw_rsi(self, df, cfg, sig, s, x0, x1, y0, y1, short):
        p = _Panel(self, x0, x1, y0, y1, len(df), 0.0, 100.0)
        p.frame("RSI (14)", yticks=3)
        p.hline(70, RED, dash=(2, 3))
        p.hline(30, GREEN, dash=(2, 3))
        p.line(df["RSI"], YELLOW, 1.2)

    def _draw_macd(self, df, cfg, sig, s, x0, x1, y0, y1, short):
        p = self._simple(x0, x1, y0, y1, len(df), "MACD (12, 26, 9)", df["MACD"], df["MACD_sig"], df["MACD_hist"], [0])
        h = df["MACD_hist"].to_numpy()
        p.bars(h, [GREEN if np.isfinite(v) and v >= 0 else RED for v in h])
        p.hline(0, AXIS)
        p.line(df["MACD"], BLUE, 1.2)
        p.line(df["MACD_sig"], ORANGE, 1.2)

    def _draw_stoch(self, df, cfg, sig, s, x0, x1, y0, y1, short):
        p = _Panel(self, x0, x1, y0, y1, len(df), 0.0, 100.0)
        p.frame("Stochastic %K / %D", yticks=3)
        p.hline(80, RED, dash=(2, 3))
        p.hline(20, GREEN, dash=(2, 3))
        p.line(df["STOCH_K"], BLUE, 1.2)
        p.line(df["STOCH_D"], ORANGE, 1.2)

    def _draw_adx(self, df, cfg, sig, s, x0, x1, y0, y1, short):
        p = self._simple(x0, x1, y0, y1, len(df), "ADX (white) · +DI (green) · -DI (red)",
                         df["ADX"], df["DI_plus"], df["DI_minus"], [0, 25])
        p.hline(20, AXIS, dash=(2, 3))
        p.line(df["DI_plus"], GREEN, 1)
        p.line(df["DI_minus"], RED, 1)
        p.line(df["ADX"], FG, 1.4)

    def _draw_cci(self, df, cfg, sig, s, x0, x1, y0, y1, short):
        p = self._simple(x0, x1, y0, y1, len(df), "CCI (20)", df["CCI"], [-120, 120])
        p.hline(100, RED, dash=(2, 3))
        p.hline(-100, GREEN, dash=(2, 3))
        p.hline(0, AXIS)
        p.line(df["CCI"], PURPLE, 1.2)

    def _draw_obv(self, df, cfg, sig, s, x0, x1, y0, y1, short):
        p = self._simple(x0, x1, y0, y1, len(df), "On-balance volume", df["OBV"], df["OBV_SMA"])
        p.line(df["OBV"], BLUE, 1.2)
        p.line(df["OBV_SMA"], ORANGE, 1)
