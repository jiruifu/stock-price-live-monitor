"""Pluggable strategy registry.

Each rule inspects an enriched frame and returns (vote, reason) with vote in {+1, 0, -1}.
Rules are registered with the @strategy decorator so the GUI can list them in the
Strategies menu.  `evaluate` combines the enabled rules into a LONG / SHORT / NEUTRAL call."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Each rule inspects an enriched frame and returns (vote, reason) with vote in {+1, 0, -1}.
# A rule is registered with a key, display name, category and description, so the GUI
# can expose it as a toggle in the Strategies menu.

Vote = tuple[int, str]


@dataclass
class Strategy:
    key: str
    name: str
    category: str
    description: str
    fn: "callable"
    default_on: bool = True


STRATEGIES: list[Strategy] = []


def strategy(key: str, name: str, category: str, description: str, default_on: bool = True):
    def deco(fn):
        STRATEGIES.append(Strategy(key, name, category, description, fn, default_on))
        return fn
    return deco


def _ok(*vals) -> bool:
    return all(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in vals)


# ---- Trend -------------------------------------------------------------- #
@strategy("sma_trend", "SMA trend (fast vs slow)", "Trend",
          "Bullish when the fast SMA is above the slow SMA, bearish when below.")
def s_sma_trend(df, last, prev, ctx) -> Vote:
    f, s = last["SMA_fast"], last["SMA_slow"]
    if not _ok(f, s):
        return 0, "[ ] SMA trend: not enough bars."
    if f > s:
        return 1, f"[+] Uptrend: SMA{ctx['fast']} ({f:.2f}) above SMA{ctx['slow']} ({s:.2f})."
    if f < s:
        return -1, f"[-] Downtrend: SMA{ctx['fast']} ({f:.2f}) below SMA{ctx['slow']} ({s:.2f})."
    return 0, "[ ] SMAs equal."


@strategy("price_vs_sma", "Price vs slow SMA", "Trend",
          "Bullish when price trades above the slow SMA, bearish below it.")
def s_price_vs_sma(df, last, prev, ctx) -> Vote:
    p, s = last["Close"], last["SMA_slow"]
    if not _ok(s):
        return 0, "[ ] Price vs SMA: not enough bars."
    if p > s:
        return 1, f"[+] Price {p:.2f} is above SMA{ctx['slow']} ({s:.2f})."
    return -1, f"[-] Price {p:.2f} is below SMA{ctx['slow']} ({s:.2f})."


@strategy("sma_cross", "SMA crossover (recent)", "Trend",
          "Golden / death cross of fast and slow SMAs within the last 5 bars.")
def s_sma_cross(df, last, prev, ctx) -> Vote:
    diff = (df["SMA_fast"] - df["SMA_slow"]).tail(6)
    if diff.isna().any():
        return 0, "[ ] SMA crossover: not enough bars."
    if (diff.iloc[:-1] <= 0).any() and diff.iloc[-1] > 0:
        return 1, f"[+] Golden cross: SMA{ctx['fast']} crossed above SMA{ctx['slow']} within the last 5 bars."
    if (diff.iloc[:-1] >= 0).any() and diff.iloc[-1] < 0:
        return -1, f"[-] Death cross: SMA{ctx['fast']} crossed below SMA{ctx['slow']} within the last 5 bars."
    return 0, "[ ] No recent SMA crossover."


@strategy("ema_cross", "EMA crossover", "Trend",
          "Fast EMA above slow EMA is bullish; extra weight if the cross happened in the last 3 bars.")
def s_ema_cross(df, last, prev, ctx) -> Vote:
    diff = (df["EMA_fast"] - df["EMA_slow"]).tail(4)
    if diff.isna().any():
        return 0, "[ ] EMA crossover: not enough bars."
    recent = (diff.iloc[:-1] <= 0).any() if diff.iloc[-1] > 0 else (diff.iloc[:-1] >= 0).any()
    if diff.iloc[-1] > 0:
        return 1, f"[+] EMA{ctx['fast']} above EMA{ctx['slow']}" + (" (fresh bullish cross)." if recent else ".")
    if diff.iloc[-1] < 0:
        return -1, f"[-] EMA{ctx['fast']} below EMA{ctx['slow']}" + (" (fresh bearish cross)." if recent else ".")
    return 0, "[ ] EMAs equal."


@strategy("sma200", "200-period SMA regime", "Trend",
          "Long-term regime filter: price above the 200 SMA is bullish, below is bearish.", default_on=False)
def s_sma200(df, last, prev, ctx) -> Vote:
    s = last["SMA_200"]
    if not _ok(s):
        return 0, f"[ ] SMA200: need 200 bars, have {len(df)}."
    if last["Close"] > s:
        return 1, f"[+] Price above SMA200 ({s:.2f}): bull regime."
    return -1, f"[-] Price below SMA200 ({s:.2f}): bear regime."


@strategy("adx", "ADX / DMI trend strength", "Trend",
          "When ADX > 20 the trend is tradeable; direction from +DI vs -DI.")
def s_adx(df, last, prev, ctx) -> Vote:
    a, dp, dm = last["ADX"], last["DI_plus"], last["DI_minus"]
    if not _ok(a, dp, dm):
        return 0, "[ ] ADX: not enough bars."
    if a < 20:
        return 0, f"[ ] ADX {a:.1f} < 20: no clear trend (range-bound)."
    if dp > dm:
        return 1, f"[+] ADX {a:.1f} with +DI ({dp:.1f}) > -DI ({dm:.1f}): strong uptrend."
    return -1, f"[-] ADX {a:.1f} with -DI ({dm:.1f}) > +DI ({dp:.1f}): strong downtrend."


@strategy("ichimoku", "Ichimoku cloud", "Trend",
          "Price above the cloud with Tenkan > Kijun is bullish; below the cloud with Tenkan < Kijun is bearish.")
def s_ichimoku(df, last, prev, ctx) -> Vote:
    a, b, t, k, p = last["ICH_a"], last["ICH_b"], last["ICH_tenkan"], last["ICH_kijun"], last["Close"]
    if not _ok(a, b, t, k):
        return 0, "[ ] Ichimoku: not enough bars (needs 78)."
    top, bot = max(a, b), min(a, b)
    if p > top and t > k:
        return 1, f"[+] Ichimoku: price above cloud ({bot:.2f}-{top:.2f}), Tenkan > Kijun."
    if p < bot and t < k:
        return -1, f"[-] Ichimoku: price below cloud ({bot:.2f}-{top:.2f}), Tenkan < Kijun."
    return 0, "[ ] Ichimoku: price inside cloud or mixed Tenkan/Kijun."


@strategy("psar", "Parabolic SAR", "Trend",
          "Bullish while price is above the SAR dots, bearish while below.")
def s_psar(df, last, prev, ctx) -> Vote:
    sar = last["PSAR"]
    if not _ok(sar):
        return 0, "[ ] PSAR: not enough bars."
    if last["Close"] > sar:
        return 1, f"[+] Price above Parabolic SAR ({sar:.2f})."
    return -1, f"[-] Price below Parabolic SAR ({sar:.2f})."


@strategy("supertrend", "Supertrend (10, 3)", "Trend",
          "ATR-based trailing trend line; direction flips on a close through the line.")
def s_supertrend(df, last, prev, ctx) -> Vote:
    d, line = last["ST_dir"], last["ST"]
    if not _ok(line) or d == 0:
        return 0, "[ ] Supertrend: not enough bars."
    if d > 0:
        return 1, f"[+] Supertrend bullish, support at {line:.2f}."
    return -1, f"[-] Supertrend bearish, resistance at {line:.2f}."


@strategy("structure", "Market structure (HH/HL vs LH/LL)", "Trend",
          "Compares the last two 10-bar swing highs and lows: higher highs + higher lows is bullish.")
def s_structure(df, last, prev, ctx) -> Vote:
    if len(df) < 30:
        return 0, "[ ] Structure: not enough bars."
    h, lo = df["High"], df["Low"]
    hh = h.tail(10).max() > h.iloc[-20:-10].max()
    hl = lo.tail(10).min() > lo.iloc[-20:-10].min()
    if hh and hl:
        return 1, "[+] Structure: higher high and higher low vs previous 10 bars."
    if not hh and not hl:
        return -1, "[-] Structure: lower high and lower low vs previous 10 bars."
    return 0, "[ ] Structure: mixed (no clean HH/HL or LH/LL)."


# ---- Momentum ----------------------------------------------------------- #
@strategy("rsi", "RSI (14) overbought / oversold", "Momentum",
          "Mean reversion: RSI < 30 is a long bias, RSI > 70 a short bias.")
def s_rsi(df, last, prev, ctx) -> Vote:
    r = last["RSI"]
    if not _ok(r):
        return 0, "[ ] RSI: not enough bars."
    if r < 30:
        return 1, f"[+] RSI {r:.1f} oversold (<30): mean-reversion long bias."
    if r > 70:
        return -1, f"[-] RSI {r:.1f} overbought (>70): mean-reversion short bias."
    return 0, f"[ ] RSI {r:.1f} neutral."


@strategy("rsi_trend", "RSI 50 midline", "Momentum",
          "Trend-following reading of RSI: above 50 bullish momentum, below 50 bearish.", default_on=False)
def s_rsi_trend(df, last, prev, ctx) -> Vote:
    r = last["RSI"]
    if not _ok(r):
        return 0, "[ ] RSI midline: not enough bars."
    if r > 55:
        return 1, f"[+] RSI {r:.1f} above 55: bullish momentum."
    if r < 45:
        return -1, f"[-] RSI {r:.1f} below 45: bearish momentum."
    return 0, f"[ ] RSI {r:.1f} near midline."


@strategy("macd", "MACD histogram", "Momentum",
          "Histogram positive and rising is bullish; negative and falling is bearish.")
def s_macd(df, last, prev, ctx) -> Vote:
    h, hp = last["MACD_hist"], prev["MACD_hist"]
    if not _ok(h, hp):
        return 0, "[ ] MACD: not enough bars."
    if h > 0 and h > hp:
        return 1, f"[+] MACD histogram positive and rising ({h:+.3f})."
    if h < 0 and h < hp:
        return -1, f"[-] MACD histogram negative and falling ({h:+.3f})."
    return 0, f"[ ] MACD histogram {h:+.3f}, momentum fading."


@strategy("macd_cross", "MACD signal-line cross", "Momentum",
          "MACD line crossing above its signal line within 3 bars is bullish, below is bearish.", default_on=False)
def s_macd_cross(df, last, prev, ctx) -> Vote:
    diff = (df["MACD"] - df["MACD_sig"]).tail(4)
    if diff.isna().any():
        return 0, "[ ] MACD cross: not enough bars."
    if diff.iloc[-1] > 0 and (diff.iloc[:-1] <= 0).any():
        return 1, "[+] MACD crossed above its signal line recently."
    if diff.iloc[-1] < 0 and (diff.iloc[:-1] >= 0).any():
        return -1, "[-] MACD crossed below its signal line recently."
    return 0, "[ ] No recent MACD / signal cross."


@strategy("stoch", "Stochastic (14, 3, 3)", "Momentum",
          "%K crossing above %D below 20 is bullish; crossing below %D above 80 is bearish.")
def s_stoch(df, last, prev, ctx) -> Vote:
    k, d, kp, dp = last["STOCH_K"], last["STOCH_D"], prev["STOCH_K"], prev["STOCH_D"]
    if not _ok(k, d, kp, dp):
        return 0, "[ ] Stochastic: not enough bars."
    if k < 25 and k > d and kp <= dp:
        return 1, f"[+] Stochastic %K ({k:.1f}) crossed above %D in oversold zone."
    if k > 75 and k < d and kp >= dp:
        return -1, f"[-] Stochastic %K ({k:.1f}) crossed below %D in overbought zone."
    if k < 20:
        return 1, f"[+] Stochastic %K {k:.1f} oversold."
    if k > 80:
        return -1, f"[-] Stochastic %K {k:.1f} overbought."
    return 0, f"[ ] Stochastic %K {k:.1f} neutral."


@strategy("cci", "CCI (20)", "Momentum",
          "Commodity Channel Index below -100 is oversold (long), above +100 overbought (short).")
def s_cci(df, last, prev, ctx) -> Vote:
    c = last["CCI"]
    if not _ok(c):
        return 0, "[ ] CCI: not enough bars."
    if c < -100:
        return 1, f"[+] CCI {c:.0f} < -100: oversold."
    if c > 100:
        return -1, f"[-] CCI {c:.0f} > +100: overbought."
    return 0, f"[ ] CCI {c:.0f} neutral."


@strategy("willr", "Williams %R (14)", "Momentum",
          "Below -80 is oversold (long bias); above -20 is overbought (short bias).", default_on=False)
def s_willr(df, last, prev, ctx) -> Vote:
    w = last["WILLR"]
    if not _ok(w):
        return 0, "[ ] Williams %R: not enough bars."
    if w < -80:
        return 1, f"[+] Williams %R {w:.0f} oversold."
    if w > -20:
        return -1, f"[-] Williams %R {w:.0f} overbought."
    return 0, f"[ ] Williams %R {w:.0f} neutral."


@strategy("roc", "Rate of change (10)", "Momentum",
          "Pure momentum: 10-bar return above +2% is bullish, below -2% is bearish.", default_on=False)
def s_roc(df, last, prev, ctx) -> Vote:
    r = last["ROC"]
    if not _ok(r):
        return 0, "[ ] ROC: not enough bars."
    thr = 1.0 if ctx["intraday"] else 2.0
    if r > thr:
        return 1, f"[+] 10-bar ROC {r:+.2f}% > +{thr}%: positive momentum."
    if r < -thr:
        return -1, f"[-] 10-bar ROC {r:+.2f}% < -{thr}%: negative momentum."
    return 0, f"[ ] 10-bar ROC {r:+.2f}% flat."


# ---- Volatility / breakout ---------------------------------------------- #
@strategy("bollinger", "Bollinger band reversion (20, 2)", "Volatility",
          "Close below the lower band is a long bias, above the upper band a short bias.")
def s_bollinger(df, last, prev, ctx) -> Vote:
    p, up, lo = last["Close"], last["BB_up"], last["BB_lo"]
    if not _ok(up, lo):
        return 0, "[ ] Bollinger: not enough bars."
    if p < lo:
        return 1, f"[+] Price below lower Bollinger band ({lo:.2f}): stretched to the downside."
    if p > up:
        return -1, f"[-] Price above upper Bollinger band ({up:.2f}): stretched to the upside."
    return 0, "[ ] Price inside Bollinger bands."


@strategy("bb_squeeze", "Bollinger / Keltner squeeze breakout", "Volatility",
          "When Bollinger bands sit inside the Keltner channel volatility is compressed; "
          "a breakout in the direction of the MACD histogram is traded.", default_on=False)
def s_bb_squeeze(df, last, prev, ctx) -> Vote:
    need = ["BB_up", "BB_lo", "KC_up", "KC_lo", "MACD_hist"]
    tail = df[need].tail(6)
    if tail.isna().any().any():
        return 0, "[ ] Squeeze: not enough bars."
    sq = (tail["BB_up"] < tail["KC_up"]) & (tail["BB_lo"] > tail["KC_lo"])
    if sq.iloc[-1]:
        return 0, "[ ] Squeeze on: volatility compressed, waiting for release."
    if sq.iloc[:-1].any():
        if last["MACD_hist"] > 0:
            return 1, "[+] Squeeze just released with positive momentum."
        if last["MACD_hist"] < 0:
            return -1, "[-] Squeeze just released with negative momentum."
    return 0, "[ ] No squeeze in the last 5 bars."


@strategy("donchian", "Donchian channel breakout (20)", "Volatility",
          "Close above the prior 20-bar high is a breakout long; below the prior 20-bar low a breakdown short.")
def s_donchian(df, last, prev, ctx) -> Vote:
    p, up, lo = last["Close"], last["DC_up"], last["DC_lo"]
    if not _ok(up, lo):
        return 0, "[ ] Donchian: not enough bars."
    if p > up:
        return 1, f"[+] Breakout: close {p:.2f} above 20-bar high {up:.2f}."
    if p < lo:
        return -1, f"[-] Breakdown: close {p:.2f} below 20-bar low {lo:.2f}."
    return 0, f"[ ] Inside Donchian channel {lo:.2f}-{up:.2f}."


@strategy("keltner", "Keltner channel (20, 1.5 ATR)", "Volatility",
          "Trend-following: close above the upper Keltner band is bullish, below the lower band bearish.",
          default_on=False)
def s_keltner(df, last, prev, ctx) -> Vote:
    p, up, lo = last["Close"], last["KC_up"], last["KC_lo"]
    if not _ok(up, lo):
        return 0, "[ ] Keltner: not enough bars."
    if p > up:
        return 1, f"[+] Close above upper Keltner band ({up:.2f})."
    if p < lo:
        return -1, f"[-] Close below lower Keltner band ({lo:.2f})."
    return 0, "[ ] Inside Keltner channel."


# ---- Volume ------------------------------------------------------------- #
@strategy("obv", "On-balance volume trend", "Volume",
          "OBV above its 20-bar average means volume is confirming an advance; below, a decline.")
def s_obv(df, last, prev, ctx) -> Vote:
    o, m = last["OBV"], last["OBV_SMA"]
    if not _ok(o, m):
        return 0, "[ ] OBV: not enough bars."
    if o > m:
        return 1, "[+] OBV above its 20-bar average: volume confirms buying."
    if o < m:
        return -1, "[-] OBV below its 20-bar average: volume confirms selling."
    return 0, "[ ] OBV flat."


@strategy("vwap", "VWAP", "Volume",
          "Price above VWAP (session VWAP intraday, 20-bar rolling VWAP on daily) is bullish.")
def s_vwap(df, last, prev, ctx) -> Vote:
    v = last["VWAP"]
    if not _ok(v):
        return 0, "[ ] VWAP: not available."
    label = "session VWAP" if ctx["intraday"] else "20-bar VWAP"
    if last["Close"] > v:
        return 1, f"[+] Price above {label} ({v:.2f})."
    return -1, f"[-] Price below {label} ({v:.2f})."


@strategy("vol_spike", "Volume spike confirmation", "Volume",
          "A bar with volume > 1.5x its 20-bar average votes in the direction of that bar.", default_on=False)
def s_vol_spike(df, last, prev, ctx) -> Vote:
    v, m = last["Volume"], last["VOL_SMA"]
    if not _ok(m) or m == 0:
        return 0, "[ ] Volume spike: not enough bars."
    ratio = v / m
    if ratio < 1.5:
        return 0, f"[ ] Volume {ratio:.1f}x average, no spike."
    if last["Close"] > last["Open"]:
        return 1, f"[+] Bullish bar on {ratio:.1f}x average volume."
    if last["Close"] < last["Open"]:
        return -1, f"[-] Bearish bar on {ratio:.1f}x average volume."
    return 0, f"[ ] Doji on {ratio:.1f}x volume."


# ---- Price action ------------------------------------------------------- #
@strategy("engulfing", "Engulfing candle", "Price action",
          "Bullish engulfing (after a down bar) votes long; bearish engulfing votes short.", default_on=False)
def s_engulfing(df, last, prev, ctx) -> Vote:
    o, c, po, pc = last["Open"], last["Close"], prev["Open"], prev["Close"]
    if c > o and pc < po and o <= pc and c >= po:
        return 1, "[+] Bullish engulfing candle."
    if c < o and pc > po and o >= pc and c <= po:
        return -1, "[-] Bearish engulfing candle."
    return 0, "[ ] No engulfing pattern."


@strategy("hammer", "Hammer / shooting star", "Price action",
          "Long lower wick after a decline is a hammer (long); long upper wick after a rise is a shooting star (short).",
          default_on=False)
def s_hammer(df, last, prev, ctx) -> Vote:
    o, h, lo, c = last["Open"], last["High"], last["Low"], last["Close"]
    body = abs(c - o)
    rng = h - lo
    if rng <= 0:
        return 0, "[ ] Flat bar."
    upper, lower = h - max(o, c), min(o, c) - lo
    down_before = df["Close"].iloc[-6:-1].is_monotonic_decreasing if len(df) >= 6 else False
    up_before = df["Close"].iloc[-6:-1].is_monotonic_increasing if len(df) >= 6 else False
    if lower > 2 * body and upper < body and (down_before or df["Close"].iloc[-1] < df["SMA_fast"].iloc[-1]):
        return 1, "[+] Hammer candle after weakness."
    if upper > 2 * body and lower < body and (up_before or df["Close"].iloc[-1] > df["SMA_fast"].iloc[-1]):
        return -1, "[-] Shooting-star candle after strength."
    return 0, "[ ] No hammer / shooting star."


@strategy("gap", "Gap vs previous close", "Price action",
          "Gap up of more than 1% votes long, gap down more than 1% votes short (daily bars only).",
          default_on=False)
def s_gap(df, last, prev, ctx) -> Vote:
    if ctx["intraday"]:
        return 0, "[ ] Gap rule only applies to daily bars."
    g = (last["Open"] - prev["Close"]) / prev["Close"] * 100
    if g > 1:
        return 1, f"[+] Gapped up {g:+.2f}% today."
    if g < -1:
        return -1, f"[-] Gapped down {g:+.2f}% today."
    return 0, f"[ ] Gap {g:+.2f}%, not significant."


STRATEGY_BY_KEY = {st.key: st for st in STRATEGIES}
CATEGORIES = list(dict.fromkeys(st.category for st in STRATEGIES))

PRESETS = {
    "Default mix": [st.key for st in STRATEGIES if st.default_on],
    "Trend following": ["sma_trend", "price_vs_sma", "ema_cross", "adx", "supertrend", "psar", "ichimoku", "structure", "sma200"],
    "Mean reversion": ["rsi", "stoch", "cci", "willr", "bollinger"],
    "Momentum": ["macd", "macd_cross", "rsi_trend", "roc", "adx", "ema_cross"],
    "Breakout": ["donchian", "keltner", "bb_squeeze", "vol_spike", "structure"],
    "Volume confirmed": ["obv", "vwap", "vol_spike", "sma_trend", "macd"],
    "Everything": [st.key for st in STRATEGIES],
}


@dataclass
class Signal:
    label: str  # LONG / SHORT / NEUTRAL
    score: int
    n_rules: int
    threshold: int
    reasons: list[str] = field(default_factory=list)
    entry: float = float("nan")
    stop: float = float("nan")
    target: float = float("nan")


def evaluate(df: pd.DataFrame, fast: int, slow: int, enabled: list[str], intraday: bool = False) -> Signal:
    """Run every enabled rule on the last bar and combine votes into a LONG/SHORT/NEUTRAL call.

    Threshold scales with the number of enabled rules: roughly a third of them must agree
    (minimum 2, or 1 when only a single rule is on)."""
    n = len(enabled)
    if n == 0:
        return Signal("NEUTRAL", 0, 0, 0, ["No strategies enabled. Pick some in the Strategies menu."])
    threshold = 1 if n == 1 else max(2, int(np.ceil(n / 3)))
    if len(df) < 3:
        return Signal("NEUTRAL", 0, n, threshold, [f"Not enough bars ({len(df)})."])

    last, prev = df.iloc[-1], df.iloc[-2]
    ctx = {"fast": fast, "slow": slow, "intraday": intraday}
    score, reasons = 0, []
    for key in enabled:
        st = STRATEGY_BY_KEY.get(key)
        if st is None:
            continue
        try:
            vote, why = st.fn(df, last, prev, ctx)
        except Exception as exc:  # noqa: BLE001 - a broken rule must not kill the app
            vote, why = 0, f"[ ] {st.name}: error ({exc})."
        score += int(vote)
        reasons.append(why)

    price = float(last["Close"])
    label = "LONG" if score >= threshold else "SHORT" if score <= -threshold else "NEUTRAL"
    a = float(last["ATR"]) if _ok(last["ATR"]) else price * 0.02
    sig = Signal(label, score, n, threshold, reasons, entry=price)
    if label == "LONG":
        sig.stop, sig.target = price - 2 * a, price + 3 * a
    elif label == "SHORT":
        sig.stop, sig.target = price + 2 * a, price - 3 * a
    return sig
