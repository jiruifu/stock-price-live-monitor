"""Market data access: fetch OHLCV bars + live quote from Yahoo Finance (yfinance) and
package them into a `Snapshot`.  Runs in a worker thread, so it must not touch Tk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from config import LONG_TERM, SHORT_TERM
from indicators import enrich
from strategies import Signal, evaluate

@dataclass
class Snapshot:
    ticker: str
    name: str
    price: float
    prev_close: float
    day_high: float
    day_low: float
    volume: float
    currency: str
    market_state: str
    fetched_at: datetime
    short_df: pd.DataFrame
    long_df: pd.DataFrame


def fetch_snapshot(ticker: str) -> Snapshot:
    t = yf.Ticker(ticker)

    long_raw = t.history(period=LONG_TERM["period"], interval=LONG_TERM["interval"], auto_adjust=True)
    if long_raw.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol.")
    short_raw = t.history(period=SHORT_TERM["period"], interval=SHORT_TERM["interval"], auto_adjust=True)
    if short_raw.empty:  # some tickers lack intraday data
        short_raw = long_raw.tail(60)

    long_df = enrich(long_raw, LONG_TERM["fast"], LONG_TERM["slow"], intraday=False)
    short_df = enrich(short_raw, SHORT_TERM["fast"], SHORT_TERM["slow"], intraday=True)

    # Live quote: fast_info is cheap; fall back to last bar.
    fi = {}
    try:
        fi = dict(t.fast_info)
    except Exception:
        pass

    def pick(key, fallback):
        v = fi.get(key)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else fallback

    last_bar = short_df.iloc[-1]
    price = pick("lastPrice", float(last_bar["Close"]))
    prev_close = pick("previousClose", float(long_df["Close"].iloc[-2]) if len(long_df) > 1 else price)
    name = ticker.upper()
    market_state = "UNKNOWN"
    try:
        info = t.get_info() if hasattr(t, "get_info") else t.info
        name = info.get("shortName") or info.get("longName") or name
        market_state = info.get("marketState", market_state)
    except Exception:
        pass

    # Patch live price into the last bar so indicators reflect it.
    for frame in (short_df, long_df):
        frame.iloc[-1, frame.columns.get_loc("Close")] = price

    long_df = enrich(long_df[["Open", "High", "Low", "Close", "Volume"]], LONG_TERM["fast"], LONG_TERM["slow"], intraday=False)
    short_df = enrich(short_df[["Open", "High", "Low", "Close", "Volume"]], SHORT_TERM["fast"], SHORT_TERM["slow"], intraday=True)

    return Snapshot(
        ticker=ticker.upper(),
        name=name,
        price=price,
        prev_close=prev_close,
        day_high=pick("dayHigh", float(long_df["High"].iloc[-1])),
        day_low=pick("dayLow", float(long_df["Low"].iloc[-1])),
        volume=pick("lastVolume", float(long_df["Volume"].iloc[-1])),
        currency=str(fi.get("currency") or "USD"),
        market_state=market_state,
        fetched_at=datetime.now(),
        short_df=short_df,
        long_df=long_df,
    )


def evaluate_snapshot(s: Snapshot, enabled: list[str]) -> tuple[Signal, Signal]:
    return (
        evaluate(s.short_df, SHORT_TERM["fast"], SHORT_TERM["slow"], enabled, intraday=True),
        evaluate(s.long_df, LONG_TERM["fast"], LONG_TERM["slow"], enabled, intraday=False),
    )
