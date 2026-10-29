"""Technical indicator maths (pure pandas / numpy) and `enrich`, which computes every
indicator any strategy or chart panel may need on an OHLCV frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    return mid, mid + k * std, mid - k * std


def stochastic(df: pd.DataFrame, n: int = 14, k_smooth: int = 3, d_smooth: int = 3):
    lo = df["Low"].rolling(n).min()
    hi = df["High"].rolling(n).max()
    raw = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    k = raw.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def adx(df: pd.DataFrame, n: int = 14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), plus_di, minus_di


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def vwap(df: pd.DataFrame, intraday: bool, n: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = tp * df["Volume"]
    if intraday:
        day = df.index.normalize() if hasattr(df.index, "normalize") else df.index
        return pv.groupby(day).cumsum() / df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv.rolling(n).sum() / df["Volume"].rolling(n).sum().replace(0, np.nan)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi = df["High"].rolling(n).max()
    lo = df["Low"].rolling(n).min()
    return -100 * (hi - df["Close"]) / (hi - lo).replace(0, np.nan)


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    high, low = df["High"].to_numpy(), df["Low"].to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    if n < 3:
        return pd.Series(out, index=df.index)
    bull = high[1] > high[0]
    sar = low[0] if bull else high[0]
    ep = high[0] if bull else low[0]
    af = step
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < sar:
                bull, sar, ep, af = False, ep, low[i], step
            else:
                if high[i] > ep:
                    ep, af = high[i], min(af + step, max_step)
        else:
            sar = max(sar, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > sar:
                bull, sar, ep, af = True, ep, high[i], step
            else:
                if low[i] < ep:
                    ep, af = low[i], min(af + step, max_step)
        out[i] = sar
    return pd.Series(out, index=df.index)


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    """Returns (supertrend line, direction) where direction is +1 bullish / -1 bearish."""
    hl2 = (df["High"] + df["Low"]) / 2
    a = atr(df, n)
    upper = (hl2 + mult * a).to_numpy()
    lower = (hl2 - mult * a).to_numpy()
    close = df["Close"].to_numpy()
    m = len(df)
    st = np.full(m, np.nan)
    d = np.zeros(m)
    fu, fl = upper.copy(), lower.copy()
    for i in range(1, m):
        if np.isnan(upper[i]) or np.isnan(upper[i - 1]):
            continue
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        if d[i - 1] <= 0:  # was bearish (or undefined)
            d[i] = 1 if close[i] > fu[i - 1] else -1
        else:
            d[i] = -1 if close[i] < fl[i - 1] else 1
        st[i] = fl[i] if d[i] == 1 else fu[i]
    return pd.Series(st, index=df.index), pd.Series(d, index=df.index)


def enrich(df: pd.DataFrame, fast: int, slow: int, intraday: bool = False) -> pd.DataFrame:
    """Compute every indicator any strategy may need. Cheap, so always compute all."""
    df = df.copy()
    c = df["Close"]
    df["SMA_fast"] = sma(c, fast)
    df["SMA_slow"] = sma(c, slow)
    df["SMA_200"] = sma(c, 200)
    df["EMA_fast"] = ema(c, fast)
    df["EMA_slow"] = ema(c, slow)
    df["RSI"] = rsi(c)
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd(c)
    df["ATR"] = atr(df)
    df["BB_mid"], df["BB_up"], df["BB_lo"] = bollinger(c)
    df["STOCH_K"], df["STOCH_D"] = stochastic(df)
    df["ADX"], df["DI_plus"], df["DI_minus"] = adx(df)
    df["OBV"] = obv(df)
    df["OBV_SMA"] = sma(df["OBV"], 20)
    df["VWAP"] = vwap(df, intraday)
    df["DC_up"] = df["High"].rolling(20).max().shift(1)   # shifted: breakout vs *prior* channel
    df["DC_lo"] = df["Low"].rolling(20).min().shift(1)
    df["CCI"] = cci(df)
    df["WILLR"] = williams_r(df)
    df["ROC"] = c.pct_change(10) * 100
    df["VOL_SMA"] = sma(df["Volume"], 20)
    kc_mid = ema(c, 20)
    df["KC_up"], df["KC_lo"] = kc_mid + 1.5 * df["ATR"], kc_mid - 1.5 * df["ATR"]
    # Ichimoku (9 / 26 / 52); cloud aligned to the *current* bar (i.e. shifted forward 26)
    tenkan = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2
    kijun = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2
    df["ICH_tenkan"], df["ICH_kijun"] = tenkan, kijun
    df["ICH_a"] = ((tenkan + kijun) / 2).shift(26)
    df["ICH_b"] = ((df["High"].rolling(52).max() + df["Low"].rolling(52).min()) / 2).shift(26)
    df["PSAR"] = parabolic_sar(df)
    df["ST"], df["ST_dir"] = supertrend(df)
    return df
