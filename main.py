#!/usr/bin/env python3
"""
Real-time stock monitor with configurable long / short strategy signals.

Data source : Yahoo Finance via `yfinance` (free, ~1 minute delayed for most exchanges).
GUI         : pure tkinter; charts are vector drawn on a Tk canvas (crisp on HiDPI).

Modules
  config.py       timeframes + colour palette
  indicators.py   indicator maths and `enrich`
  strategies.py   pluggable rule registry, presets, `evaluate`
  data.py         yfinance fetching -> `Snapshot`
  levels.py       support / resistance detection
  chart.py        Tk canvas chart (panels, overlays, candles / line)
  app.py          the tkinter application
  main.py         this entry point

Run:  python3 main.py
Educational tool, not financial advice.
"""

from __future__ import annotations

import tkinter as tk

from app import StockMonitorApp


def main():
    root = tk.Tk()
    StockMonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
