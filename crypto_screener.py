#!/usr/bin/env python3
"""
CRYPTO SCREENER — Détecteur d'opportunités swing (positions ≤ 2 semaines)
Scanne TOUT le marché spot Binance (paires USDT) et classe les actifs selon
un score composite multi-piliers fondé sur des concepts éprouvés :

  TENDANCE   : EMA 20/50/200, ADX/DMI (Wilder), Ichimoku, Supertrend
  MOMENTUM   : ROC 7/14/28j, force relative vs BTC (O'Neil / Jegadeesh-Titman),
               MACD, effet plus-haut-200j (George & Hwang)
  TIMING     : RSI, stochastique, breakout Donchian 20j (Turtle Traders),
               pullback sur EMA20, squeeze Bollinger (TTM), divergences RSI
  VOLUME     : OBV (Granville), MFI, ratio volume haussier/baissier (Wyckoff),
               volume relatif (RVOL)
  QUALITÉ    : liquidité, volatilité ATR, profondeur d'historique, résistances
  RÉGIME     : filtre de tendance BTC, breadth du marché, Fear & Greed Index

Gestion du risque : stops ATR / plus-bas swing, objectifs 2R/4R, taille de
position à risque fixe (règle du 1 %, Van Tharp), sortie temporelle 14 jours.

Usage :
    python3 crypto_screener.py                      # scan complet du marché
    python3 crypto_screener.py --top 20 --detail 8  # plus de résultats
    python3 crypto_screener.py --min-vol 10         # volume 24h ≥ 10 M$
    python3 crypto_screener.py SOL INJ FET          # analyse ciblée
    python3 crypto_screener.py --capital 5000 --risk 1.5
    python3 crypto_screener.py --json resultats.json

Aucune clé API requise (endpoints publics Binance).
⚠ Outil éducatif — ne constitue pas un conseil en investissement.
"""

import argparse
import json
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN, RED, YELLOW, CYAN, MAGENTA = Fore.GREEN, Fore.RED, Fore.YELLOW, Fore.CYAN, Fore.MAGENTA
    BOLD, DIM, RESET = Style.BRIGHT, Style.DIM, Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = MAGENTA = BOLD = DIM = RESET = ""

BASE_URL = "https://api.binance.com"

# Actifs exclus du scan : stablecoins, fiat, jetons "wrappés" (doublons)
STABLE_BASES = {
    "USDC", "TUSD", "BUSD", "DAI", "FDUSD", "USDP", "PYUSD", "GUSD",
    "USDE", "XUSD", "USD1", "AEUR", "EURI", "SUSD", "BFUSD", "USDS",
    "RLUSD", "USDR", "USDQ", "USTC",
}
FIAT_BASES = {"EUR", "GBP", "AUD", "TRY", "BRL", "JPY", "RUB", "UAH", "ARS", "ZAR", "COP", "CZK", "MXN", "PLN", "RON"}
WRAPPED_BASES = {"WBTC", "WBETH", "BETH"}

TIME_STOP_DAYS = 14  # sortie temporelle : horizon max des positions


# ═══════════════════════════════════════════════════════════════════════════
#  COUCHE HTTP (sessions par thread, retry sur rate-limit)
# ═══════════════════════════════════════════════════════════════════════════

_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        s = requests.Session()
        s.headers.update({"User-Agent": "crypto-screener/1.0"})
        _tls.s = s
    return _tls.s


def get_json(path: str, params: dict | None = None, retries: int = 3):
    url = BASE_URL + path
    for attempt in range(retries):
        try:
            r = _session().get(url, params=params, timeout=15)
            if r.status_code in (429, 418):  # rate limit / ban temporaire
                wait = int(r.headers.get("Retry-After", 10))
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    data = get_json("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not data or len(data) < 10:
        return None
    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "ct", "qv", "n", "tb", "tq", "ig"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df[["time", "open", "high", "low", "close", "volume"]]


def fetch_fear_greed():
    """Fear & Greed Index (alternative.me) — sentiment contrarien éprouvé."""
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=6)
        d = r.json()["data"][0]
        labels = {"Extreme Fear": "Peur extrême", "Fear": "Peur", "Neutral": "Neutre",
                  "Greed": "Avidité", "Extreme Greed": "Avidité extrême"}
        return int(d["value"]), labels.get(d["value_classification"], d["value_classification"])
    except Exception:
        return None, None


def get_universe(min_vol_usd: float):
    """Construit l'univers : paires USDT actives, hors stables/fiat/wrappés/
    jetons à levier, avec volume 24h suffisant."""
    info = get_json("/api/v3/exchangeInfo")
    tickers = get_json("/api/v3/ticker/24hr")
    if not info or not tickers:
        return None, None, 0

    tick_map = {t["symbol"]: t for t in tickers}
    all_bases = {s["baseAsset"] for s in info["symbols"]}

    def is_leveraged(base: str) -> bool:
        # BTCUP, ETHDOWN, XBULL… — uniquement si le préfixe est lui-même un actif coté
        for suf in ("UP", "DOWN", "BULL", "BEAR"):
            if base.endswith(suf) and len(base) > len(suf) and base[: -len(suf)] in all_bases:
                return True
        return False

    universe = []
    total_usdt = 0
    for s in info["symbols"]:
        if s["quoteAsset"] != "USDT" or s["status"] != "TRADING":
            continue
        if not s.get("isSpotTradingAllowed", True):
            continue
        base = s["baseAsset"]
        total_usdt += 1
        if base in STABLE_BASES or base in FIAT_BASES or base in WRAPPED_BASES or is_leveraged(base):
            continue
        t = tick_map.get(s["symbol"])
        if not t:
            continue
        qv = float(t.get("quoteVolume", 0))
        if qv < min_vol_usd:
            continue
        universe.append({
            "symbol": s["symbol"],
            "base": base,
            "vol24": qv,
            "change24": float(t.get("priceChangePercent", 0)),
        })
    return universe, tick_map, total_usdt


# ═══════════════════════════════════════════════════════════════════════════
#  INDICATEURS TECHNIQUES
# ═══════════════════════════════════════════════════════════════════════════

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def macd(s: pd.Series, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(s: pd.Series, period=20, nb=2.0):
    mid = s.rolling(period).mean()
    sd = s.rolling(period).std()
    return mid + nb * sd, mid, mid - nb * sd


def stochastic(df: pd.DataFrame, k=14, d=3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    pk = 100 * (df["close"] - lo) / (hi - lo)
    return pk, pk.rolling(d).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift()
    return pd.concat([df["high"] - df["low"],
                      (df["high"] - prev).abs(),
                      (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period=14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx_dmi(df: pd.DataFrame, period=14):
    """ADX + DI directionnels — force de tendance de Wilder."""
    h, l = df["high"], df["low"]
    up, down = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    atr_w = true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False).mean()
    return float(adx_.iloc[-1]), float(pdi.iloc[-1]), float(mdi.iloc[-1])


def mfi(df: pd.DataFrame, period=14) -> float:
    """Money Flow Index — RSI pondéré par les volumes."""
    typ = (df["high"] + df["low"] + df["close"]) / 3
    rmf = typ * df["volume"]
    pos = rmf.where(typ > typ.shift(), 0.0).rolling(period).sum()
    neg = rmf.where(typ < typ.shift(), 0.0).rolling(period).sum()
    val = (100 - 100 / (1 + pos / neg.replace(0, np.nan))).iloc[-1]
    return float(val) if np.isfinite(val) else 50.0


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (Granville) — accumulation/distribution."""
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def ichimoku_position(df: pd.DataFrame) -> int | None:
    """Position du prix vs nuage Ichimoku : 1 au-dessus, 0 dedans, -1 dessous."""
    if len(df) < 80:
        return None
    h, l = df["high"], df["low"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    a, b = span_a.iloc[-1], span_b.iloc[-1]
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    price = df["close"].iloc[-1]
    top, bot = max(a, b), min(a, b)
    return 1 if price > top else (-1 if price < bot else 0)


def supertrend_dir(df: pd.DataFrame, period=10, mult=3.0) -> int:
    """Direction Supertrend : 1 haussier, -1 baissier."""
    atr_v = atr(df, period).values
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr_v, hl2 - mult * atr_v
    n = len(c)
    direction = np.ones(n, dtype=int)
    fub, flb = ub.copy(), lb.copy()
    for i in range(period + 1, n):
        fub[i] = min(ub[i], fub[i - 1]) if c[i - 1] <= fub[i - 1] else ub[i]
        flb[i] = max(lb[i], flb[i - 1]) if c[i - 1] >= flb[i - 1] else lb[i]
        if c[i] > fub[i - 1]:
            direction[i] = 1
        elif c[i] < flb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return int(direction[-1])


def detect_divergence(close: np.ndarray, low: np.ndarray, high: np.ndarray,
                      rsi_arr: np.ndarray, lookback=60) -> str | None:
    """Divergence RSI/prix sur pivots récents : 'bull', 'bear' ou None."""
    n = min(lookback, len(close))
    if n < 20:
        return None
    lo, hi, rs = low[-n:], high[-n:], rsi_arr[-n:]

    def pivot_idx(arr, mode):
        out = []
        for i in range(3, n - 3):
            w = arr[i - 3:i + 4]
            if mode == "low" and arr[i] == w.min():
                out.append(i)
            elif mode == "high" and arr[i] == w.max():
                out.append(i)
        return out

    piv_lo = pivot_idx(lo, "low")
    if len(piv_lo) >= 2:
        i1, i2 = piv_lo[-2], piv_lo[-1]
        if i2 >= n - 15 and np.isfinite(rs[i1]) and np.isfinite(rs[i2]):
            if lo[i2] < lo[i1] * 0.999 and rs[i2] > rs[i1] + 2 and rs[i2] < 50:
                return "bull"
    piv_hi = pivot_idx(hi, "high")
    if len(piv_hi) >= 2:
        i1, i2 = piv_hi[-2], piv_hi[-1]
        if i2 >= n - 15 and np.isfinite(rs[i1]) and np.isfinite(rs[i2]):
            if hi[i2] > hi[i1] * 1.001 and rs[i2] < rs[i1] - 2 and rs[i2] > 50:
                return "bear"
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  CALCUL DES MÉTRIQUES (journalier)
# ═══════════════════════════════════════════════════════════════════════════

def _f(x, default=np.nan) -> float:
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def roc(c: pd.Series, n: int) -> float:
    if len(c) <= n:
        return np.nan
    return _f((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100)


def compute_metrics(df: pd.DataFrame, sym_info: dict, btc: dict) -> dict | None:
    if df is None or len(df) < 60:
        return None
    c, h, l, v, o = df["close"], df["high"], df["low"], df["volume"], df["open"]
    price = float(c.iloc[-1])
    if price <= 0:
        return None

    m = {"symbol": sym_info["symbol"], "base": sym_info["base"],
         "vol24": sym_info["vol24"], "change24": sym_info["change24"],
         "price": price, "hist_len": len(df)}

    # — Tendance —
    e20, e50 = ema(c, 20), ema(c, 50)
    m["ema20"], m["ema50"] = _f(e20.iloc[-1]), _f(e50.iloc[-1])
    m["ema50_up"] = bool(e50.iloc[-1] > e50.iloc[-6]) if len(df) > 6 else False
    m["ema200"] = _f(ema(c, 200).iloc[-1]) if len(df) >= 200 else np.nan
    m["adx"], m["pdi"], m["mdi"] = adx_dmi(df)
    m["ichimoku"] = ichimoku_position(df)
    m["supertrend"] = supertrend_dir(df)

    # — Momentum —
    m["roc7"], m["roc14"], m["roc28"] = roc(c, 7), roc(c, 14), roc(c, 28)
    m["rs7"] = _f(m["roc7"] - btc["roc7"])
    m["rs14"] = _f(m["roc14"] - btc["roc14"])
    macd_l, macd_s, hist = macd(c)
    m["macd_above"] = bool(macd_l.iloc[-1] > macd_s.iloc[-1])
    m["hist_rising"] = bool(hist.iloc[-1] > hist.iloc[-2])
    m["hist_cross_recent"] = bool(hist.iloc[-1] > 0 and (hist.iloc[-4:-1] <= 0).any())
    high200 = float(h.max())
    m["high200"] = high200
    m["prox_high"] = price / high200 if high200 > 0 else np.nan

    # — Timing / setups —
    rsi_s = rsi(c)
    m["rsi"] = _f(rsi_s.iloc[-1], 50)
    sk, sd = stochastic(df)
    m["stoch_k"], m["stoch_d"] = _f(sk.iloc[-1], 50), _f(sd.iloc[-1], 50)
    m["stoch_bull"] = m["stoch_k"] > m["stoch_d"]
    m["stoch_cross_up"] = bool(
        len(sk) > 4 and m["stoch_bull"] and sk.iloc[-2] <= sd.iloc[-2]
        and np.nanmin(sk.iloc[-4:-1]) < 30)

    bb_up, bb_mid, bb_lo = bollinger(c)
    bbw = ((bb_up - bb_lo) / bb_mid).replace([np.inf, -np.inf], np.nan)
    win = bbw.dropna().iloc[-120:]
    if len(win) >= 30:
        thresh = win.quantile(0.15)
        m["squeeze_on"] = bool(bbw.iloc[-1] <= thresh)
        m["squeeze_release"] = bool(
            bbw.iloc[-2] <= thresh and bbw.iloc[-1] > bbw.iloc[-2] * 1.05
            and price > m["ema20"])
    else:
        m["squeeze_on"] = m["squeeze_release"] = False

    prev20 = h.shift(1).rolling(20).max().iloc[-1]
    m["prev20high"] = _f(prev20, high200)
    m["donchian_break"] = bool(np.isfinite(prev20) and price > prev20)

    atr_s = atr(df)
    m["atr"] = _f(atr_s.iloc[-1])
    m["atr_pct"] = m["atr"] / price * 100 if m["atr"] > 0 else np.nan
    m["extended"] = (price - m["ema20"]) / m["atr"] if m["atr"] > 0 else 0
    m["dist_res_atr"] = ((m["prev20high"] - price) / m["atr"]
                         if (m["atr"] > 0 and not m["donchian_break"]) else 0)

    m["pullback"] = bool(
        m["ema20"] > m["ema50"] and m["ema50_up"]
        and price >= m["ema50"] * 0.985 and price <= m["ema20"] * 1.02
        and 35 <= m["rsi"] <= 55)

    m["divergence"] = detect_divergence(c.values, l.values, h.values, rsi_s.values)

    # — Volume —
    obv_s = obv(df)
    m["obv_above"] = bool(obv_s.iloc[-1] > ema(obv_s, 20).iloc[-1])
    m["obv_rising"] = bool(len(obv_s) > 15 and obv_s.iloc[-1] > obv_s.iloc[-15])
    m["mfi"] = mfi(df)
    green = c > o
    up_vol = float(v[green].iloc[-20:].sum()) if green.any() else 0.0
    dn_vol = float(v[~green].iloc[-20:].sum()) if (~green).any() else 0.0
    m["udvol"] = up_vol / dn_vol if dn_vol > 0 else 2.0
    vol_avg = v.iloc[-21:-1].mean()
    m["rvol"] = float(v.iloc[-1] / vol_avg) if vol_avg > 0 else 1.0
    m["green"] = bool(c.iloc[-1] >= o.iloc[-1])
    m["vol_confirm"] = bool(m["donchian_break"] and m["rvol"] >= 1.5)

    # — Risque —
    m["swing_low10"] = float(l.iloc[-10:].min())
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  SCORING MULTI-PILIERS
# ═══════════════════════════════════════════════════════════════════════════

def score_coin(m: dict) -> dict:
    tags, warns = [], []
    ok, ko = "✓", "✗"

    # ── Pilier 1 : TENDANCE (22 %) ──────────────────────────────────────────
    p, d = 0, []
    if m["price"] > m["ema20"]:
        p += 15; d.append(f"P>EMA20{ok}")
    else:
        d.append(f"P<EMA20{ko}")
    if m["ema20"] > m["ema50"]:
        p += 15; d.append(f"EMA20>50{ok}")
    else:
        d.append(f"EMA20<50{ko}")
    if np.isfinite(m["ema200"]):
        if m["ema50"] > m["ema200"]:
            p += 15; d.append(f"EMA50>200{ok}")
        else:
            d.append(f"EMA50<200{ko}")
    elif m["price"] > m["ema50"]:
        p += 8; d.append("EMA200 n/d")
    adx_v = m["adx"] if np.isfinite(m["adx"]) else 0
    if m["pdi"] > m["mdi"]:
        if adx_v >= 25:
            p += 20
        elif adx_v >= 20:
            p += 12
        elif adx_v >= 15:
            p += 5
        d.append(f"ADX {adx_v:.0f}{ok if adx_v >= 20 else '·'}")
    else:
        d.append(f"ADX {adx_v:.0f} baissier{ko}")
    if m["ichimoku"] == 1:
        p += 15; d.append(f"Ichimoku{ok}")
    elif m["ichimoku"] == 0:
        p += 7; d.append("Ichimoku·nuage")
    elif m["ichimoku"] == -1:
        d.append(f"Ichimoku{ko}")
    if m["supertrend"] == 1:
        p += 20; d.append(f"Supertrend{ok}")
    else:
        d.append(f"Supertrend{ko}")
    s_trend, d_trend = min(p, 100), "  ".join(d)

    # ── Pilier 2 : MOMENTUM (22 %) ──────────────────────────────────────────
    p, d = 0, []
    r7, r14, r28 = m["roc7"], m["roc14"], m["roc28"]
    if np.isfinite(r7) and r7 > 0:
        p += 10 + (5 if r7 > 5 else 0)
    if np.isfinite(r14) and r14 > 0:
        p += 10 + (5 if r14 > 10 else 0)
    d.append(f"ROC7 {r7:+.1f}%" if np.isfinite(r7) else "ROC7 n/d")
    d.append(f"ROC14 {r14:+.1f}%" if np.isfinite(r14) else "ROC14 n/d")
    if np.isfinite(m["rs14"]) and m["rs14"] > 0:
        p += 15 + (10 if m["rs14"] > 10 else 0)
        d.append(f"RS vs BTC {m['rs14']:+.1f}{ok}")
    else:
        d.append(f"RS vs BTC {m['rs14']:+.1f}{ko}" if np.isfinite(m["rs14"]) else "RS n/d")
    if m["macd_above"]:
        p += 10
    if m["hist_rising"]:
        p += 10
    d.append(f"MACD{ok if m['macd_above'] else ko}")
    prox = m["prox_high"]
    if np.isfinite(prox):
        if prox >= 0.95:
            p += 15
        elif prox >= 0.85:
            p += 10
        elif prox >= 0.70:
            p += 5
        d.append(f"Haut200j {prox * 100:.0f}%")
    if np.isfinite(r28) and r28 > 0:
        p += 10
    s_mom, d_mom = min(p, 100), "  ".join(d)

    # ── Pilier 3 : TIMING / SETUPS (20 %) ───────────────────────────────────
    p, d = 0, []
    r = m["rsi"]
    if 45 <= r <= 65:
        p += 20
    elif 35 <= r < 45:
        p += 12
    elif 65 < r <= 72:
        p += 8
    elif 72 < r <= 78:
        p += 4
    elif r < 35:
        p += 8
    if r > 78:
        warns.append(f"RSI {r:.0f} — surchauffe, risque de correction")
    d.append(f"RSI {r:.0f}")
    if m["stoch_bull"] and m["stoch_k"] < 80:
        p += 10
    if m["stoch_cross_up"]:
        p += 10; d.append(f"Stoch croisé bas{ok}")
    else:
        d.append(f"Stoch{ok if m['stoch_bull'] else ko}")

    setup_pts = 0
    if m["donchian_break"]:
        setup_pts += 25 + (10 if m["vol_confirm"] else 0)
        tags.append("Breakout20j" + ("+vol" if m["vol_confirm"] else ""))
    if m["pullback"]:
        setup_pts += 25
        tags.append("Pullback EMA20")
    if m["squeeze_release"]:
        setup_pts += 20
        tags.append("Sortie squeeze")
    elif m["squeeze_on"]:
        setup_pts += 10
        tags.append("Squeeze")
    if m["divergence"] == "bull":
        setup_pts += 15
        tags.append("Div.RSI+")
    elif m["divergence"] == "bear":
        setup_pts -= 10
        warns.append("Divergence RSI baissière détectée")
    p += max(0, min(setup_pts, 45))

    ext = m["extended"]
    if ext <= 1.0:
        p += 10
    elif ext <= 2.0:
        p += 5
    if ext > 2.5:
        warns.append(f"Prix étendu ({ext:.1f}×ATR au-dessus EMA20) — attendre un repli")
    if m["hist_cross_recent"]:
        p += 5
    if tags:
        d.append("Setups: " + "/".join(tags))
    s_tim, d_tim = min(p, 100), "  ".join(d)

    # ── Pilier 4 : VOLUME / ACCUMULATION (18 %) ─────────────────────────────
    p, d = 0, []
    if m["obv_above"]:
        p += 25
    if m["obv_rising"]:
        p += 15
    d.append(f"OBV{'↑' + ok if m['obv_above'] else '↓' + ko}")
    mf = m["mfi"]
    if 50 <= mf <= 80:
        p += 20
    elif 35 <= mf < 50:
        p += 10
    elif mf > 80:
        p += 8
    d.append(f"MFI {mf:.0f}")
    ud = m["udvol"]
    if ud >= 1.2:
        p += 20
    elif ud >= 1.0:
        p += 10
    d.append(f"Vol.h/b {ud:.2f}{ok if ud >= 1.2 else '·'}")
    rv = m["rvol"]
    if rv >= 1.3 and m["green"]:
        p += 20
    elif rv >= 1.0:
        p += 12
    elif rv >= 0.6:
        p += 6
    d.append(f"RVOL ×{rv:.1f}")
    if rv < 0.5:
        warns.append("Volume en assèchement (RVOL < 0.5)")
    s_vol, d_vol = min(p, 100), "  ".join(d)

    # ── Pilier 5 : QUALITÉ / RISQUE (18 %) ──────────────────────────────────
    p, d = 0, []
    qv = m["vol24"]
    if qv >= 50e6:
        p += 30
    elif qv >= 20e6:
        p += 24
    elif qv >= 10e6:
        p += 18
    elif qv >= 5e6:
        p += 12
    else:
        p += 5
    d.append(f"Liq {human_vol(qv)}")
    ap = m["atr_pct"]
    if np.isfinite(ap):
        if 2 <= ap <= 8:
            p += 30
        elif 1 <= ap < 2 or 8 < ap <= 12:
            p += 18
        elif ap < 1:
            p += 8
        else:
            p += 5
            warns.append(f"Volatilité extrême (ATR {ap:.1f}%/j)")
        d.append(f"ATR {ap:.1f}%/j")
    hl = m["hist_len"]
    if hl >= 200:
        p += 20
    elif hl >= 120:
        p += 14
    else:
        p += 8
        warns.append(f"Listing récent ({hl} j d'historique)")
    d.append(f"Hist {hl}j")
    if m["donchian_break"]:
        p += 20; d.append(f"Ciel dégagé{ok}")
    elif m["dist_res_atr"] >= 1.5:
        p += 12; d.append(f"Résist. à {m['dist_res_atr']:.1f}×ATR")
    else:
        p += 5; d.append("Résist. proche")
    s_qual, d_qual = min(p, 100), "  ".join(d)

    base = (0.22 * s_trend + 0.22 * s_mom + 0.20 * s_tim
            + 0.18 * s_vol + 0.18 * s_qual)

    m.update({
        "s_trend": s_trend, "s_mom": s_mom, "s_tim": s_tim,
        "s_vol": s_vol, "s_qual": s_qual, "base_score": base,
        "d_trend": d_trend, "d_mom": d_mom, "d_tim": d_tim,
        "d_vol": d_vol, "d_qual": d_qual,
        "tags": tags, "warns": warns,
    })
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  RÉGIME DE MARCHÉ & CONFLUENCE 4H
# ═══════════════════════════════════════════════════════════════════════════

def btc_reference() -> dict | None:
    df = fetch_klines("BTCUSDT", "1d", 365)
    if df is None:
        return None
    c = df["close"]
    return {
        "df": df, "price": float(c.iloc[-1]),
        "roc7": roc(c, 7), "roc14": roc(c, 14),
        "ema50": float(ema(c, 50).iloc[-1]),
        "ema200": float(ema(c, 200).iloc[-1]),
    }


def market_regime(btc: dict, breadth: float | None, fng: int | None):
    p, e50, e200 = btc["price"], btc["ema50"], btc["ema200"]
    if p > e50 and e50 > e200:
        label, factor, color = "RISK-ON (tendance BTC haussière)", 1.05, GREEN
    elif p > e200:
        label, factor, color = "NEUTRE (BTC au-dessus EMA200)", 1.00, YELLOW
    elif p > e50:
        label, factor, color = "REBOND SOUS RÉSISTANCE", 0.90, YELLOW
    else:
        label, factor, color = "RISK-OFF (BTC sous EMA200)", 0.80, RED
        if np.isfinite(btc["roc14"]) and btc["roc14"] < -12:
            label, factor = "RISK-OFF FORT (chute BTC)", 0.70
    if breadth is not None:
        if breadth >= 55:
            factor += 0.03
        elif breadth <= 30:
            factor -= 0.05
    if fng is not None:
        if fng >= 80:
            factor -= 0.03   # euphorie → prudence contrarienne
        elif fng <= 20:
            factor += 0.02   # peur extrême → opportunités contrariennes
    return label, color, max(0.65, min(1.10, factor))


def confluence_4h(symbol: str):
    """Validation multi-timeframe : la structure 4h confirme-t-elle le 1d ?"""
    df = fetch_klines(symbol, "4h", 210)
    if df is None or len(df) < 60:
        return 0, "4h: données insuffisantes"
    c = df["close"]
    bonus, parts = 0, []
    if ema(c, 20).iloc[-1] > ema(c, 50).iloc[-1]:
        bonus += 2; parts.append("EMA✓")
    else:
        bonus -= 2; parts.append("EMA✗")
    _, _, hist = macd(c)
    if hist.iloc[-1] > 0:
        bonus += 2; parts.append("MACD✓")
    else:
        bonus -= 2; parts.append("MACD✗")
    r = _f(rsi(c).iloc[-1], 50)
    if 40 <= r <= 70:
        bonus += 1; parts.append(f"RSI {r:.0f}✓")
    elif r > 75 or r < 35:
        bonus -= 1; parts.append(f"RSI {r:.0f}✗")
    else:
        parts.append(f"RSI {r:.0f}·")
    sk, sd = stochastic(df)
    if _f(sk.iloc[-1], 50) > _f(sd.iloc[-1], 50):
        bonus += 1; parts.append("Stoch✓")
    else:
        bonus -= 1; parts.append("Stoch✗")
    return bonus, "4h: " + " ".join(parts) + f"  →  {bonus:+d} pts"


# ═══════════════════════════════════════════════════════════════════════════
#  PLAN DE TRADE
# ═══════════════════════════════════════════════════════════════════════════

def build_plan(m: dict, capital: float, risk_pct: float) -> dict:
    price, atr_d = m["price"], m["atr"]
    stop = max(m["swing_low10"] * 0.995, price - 2 * atr_d)
    if stop >= price * 0.995:
        stop = price - 1.5 * atr_d
    tp1, tp2 = price + 2 * atr_d, price + 4 * atr_d
    risk_unit = price - stop
    rr1 = (tp1 - price) / risk_unit if risk_unit > 0 else 0
    rr2 = (tp2 - price) / risk_unit if risk_unit > 0 else 0
    risk_amount = capital * risk_pct / 100
    qty = risk_amount / risk_unit if risk_unit > 0 else 0
    notional = qty * price
    capped = notional > capital
    if capped:
        qty, notional = capital / price, capital
    return {
        "entry": price, "stop": stop, "tp1": tp1, "tp2": tp2,
        "stop_pct": (price - stop) / price * 100,
        "tp1_pct": (tp1 - price) / price * 100,
        "tp2_pct": (tp2 - price) / price * 100,
        "rr1": rr1, "rr2": rr2,
        "qty": qty, "notional": notional, "capped": capped,
        "risk_amount": risk_amount,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════

def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:,.4f}"
    if p >= 0.001:
        return f"{p:.6f}"
    return f"{p:.8f}"


def human_vol(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.1f}Md$"
    if v >= 1e6:
        return f"{v / 1e6:.0f}M$"
    return f"{v / 1e3:.0f}k$"


def rating(score: float):
    if score >= 72:
        return "ACHAT FORT", GREEN + BOLD
    if score >= 62:
        return "ACHAT", GREEN
    if score >= 52:
        return "SURVEILLER", YELLOW
    if score >= 42:
        return "NEUTRE", ""
    return "ÉVITER", RED


def bar10(score: float) -> str:
    n = max(0, min(10, round(score / 10)))
    return "█" * n + "·" * (10 - n)


def pct_color(x: float) -> str:
    if not np.isfinite(x):
        return "   n/d"
    col = GREEN if x >= 0 else RED
    return f"{col}{x:+7.1f}%{RESET}"


def print_regime(btc, regime_label, regime_color, factor, breadth, fng_val, fng_label):
    print(f"\n{BOLD}{CYAN}{'═' * 78}{RESET}")
    print(f"{BOLD}{CYAN}  RÉGIME DE MARCHÉ{RESET}")
    print(f"{CYAN}{'─' * 78}{RESET}")
    e50ok = "✓" if btc["price"] > btc["ema50"] else "✗"
    e200ok = "✓" if btc["price"] > btc["ema200"] else "✗"
    print(f"  BTC {fmt_price(btc['price'])}$  |  >EMA50 {e50ok}  >EMA200 {e200ok}"
          f"  |  ROC7 {btc['roc7']:+.1f}%  ROC14 {btc['roc14']:+.1f}%")
    extras = []
    if breadth is not None:
        extras.append(f"Breadth: {breadth:.0f}% des paires > EMA50(1d)")
    if fng_val is not None:
        extras.append(f"Fear & Greed: {fng_val} ({fng_label})")
    if extras:
        print("  " + "  |  ".join(extras))
    print(f"  Régime: {regime_color}{BOLD}{regime_label}{RESET}  →  facteur ×{factor:.2f} appliqué aux scores")
    print(f"{BOLD}{CYAN}{'═' * 78}{RESET}")


def print_table(results: list[dict]):
    print(f"\n{BOLD}  {'#':>2}  {'PAIRE':<13} {'PRIX':>12}  {'24H':>8}  {'7J':>8}"
          f"  {'VOL24H':>7}  {'SCORE':>5}  {'NOTE':<11} SETUPS{RESET}")
    print(f"  {'─' * 96}")
    for i, m in enumerate(results, 1):
        note, ncol = rating(m["final_score"])
        tags = ", ".join(m["tags"])[:30] if m["tags"] else "—"
        print(f"  {i:>2}  {m['symbol']:<13} {fmt_price(m['price']):>12}"
              f"  {pct_color(m['change24'])}  {pct_color(m['roc7'])}"
              f"  {human_vol(m['vol24']):>7}"
              f"  {BOLD}{m['final_score']:>5.1f}{RESET}"
              f"  {ncol}{note:<11}{RESET} {tags}")
    print(f"  {'─' * 96}")


def print_card(rank: int, m: dict, plan: dict, capital: float, risk_pct: float):
    note, ncol = rating(m["final_score"])
    tag_str = ("  [" + " · ".join(m["tags"]) + "]") if m["tags"] else ""
    print(f"\n{BOLD}{'═' * 78}{RESET}")
    print(f"{BOLD}  #{rank}  {CYAN}{m['symbol']}{RESET}{BOLD}{tag_str}{RESET}")
    print(f"{'─' * 78}")
    c24 = GREEN if m["change24"] >= 0 else RED
    r7c = GREEN if (np.isfinite(m["roc7"]) and m["roc7"] >= 0) else RED
    r7s = f"{m['roc7']:+.1f}%" if np.isfinite(m["roc7"]) else "n/d"
    print(f"  Prix {BOLD}{fmt_price(m['price'])} USDT{RESET}"
          f"   24h {c24}{m['change24']:+.1f}%{RESET}"
          f"   7j {r7c}{r7s}{RESET}"
          f"   Vol24h {human_vol(m['vol24'])}")
    fs = m["final_score"]
    extra = f"   (1d {m['base_score']:.1f} ×régime, 4h {m['bonus4h']:+d})" if "bonus4h" in m else ""
    print(f"  Score {BOLD}{fs:.1f}/100{RESET} [{bar10(fs)}]  →  {ncol}{BOLD}{note}{RESET}{extra}")
    print()
    print(f"  ├─ Tendance  {m['s_trend']:>3.0f} [{bar10(m['s_trend'])}]  {DIM}{m['d_trend']}{RESET}")
    print(f"  ├─ Momentum  {m['s_mom']:>3.0f} [{bar10(m['s_mom'])}]  {DIM}{m['d_mom']}{RESET}")
    print(f"  ├─ Timing    {m['s_tim']:>3.0f} [{bar10(m['s_tim'])}]  {DIM}{m['d_tim']}{RESET}")
    print(f"  ├─ Volume    {m['s_vol']:>3.0f} [{bar10(m['s_vol'])}]  {DIM}{m['d_vol']}{RESET}")
    print(f"  └─ Qualité   {m['s_qual']:>3.0f} [{bar10(m['s_qual'])}]  {DIM}{m['d_qual']}{RESET}")
    if m.get("conf4h_str"):
        print(f"     {DIM}{m['conf4h_str']}{RESET}")

    print(f"\n  {BOLD}PLAN DE TRADE (horizon max {TIME_STOP_DAYS} jours){RESET}")
    print(f"    Entrée    : {fmt_price(plan['entry'])} USDT")
    print(f"    Stop-loss : {RED}{fmt_price(plan['stop'])}{RESET}  (−{plan['stop_pct']:.1f}%)")
    print(f"    Objectif 1: {GREEN}{fmt_price(plan['tp1'])}{RESET}  (+{plan['tp1_pct']:.1f}%, R/R {plan['rr1']:.1f})"
          f"   Objectif 2: {GREEN}{fmt_price(plan['tp2'])}{RESET}  (+{plan['tp2_pct']:.1f}%, R/R {plan['rr2']:.1f})")
    cap_note = f"  {YELLOW}(plafonné au capital){RESET}" if plan["capped"] else ""
    print(f"    Taille    : {plan['qty']:.6g} {m['base']} ≈ {plan['notional']:.0f}$ "
          f"(capital {capital:.0f}$, risque {risk_pct:.1f}% = {plan['risk_amount']:.0f}$){cap_note}")
    if np.isfinite(m["prox_high"]):
        dist_h = (m["high200"] / m["price"] - 1) * 100
        print(f"    Plus haut 200j : {fmt_price(m['high200'])} ({'+' if dist_h >= 0 else ''}{dist_h:.1f}%)")
    print(f"    ⏰ Sortie temporelle : clôturer après {TIME_STOP_DAYS} j si ni stop ni objectif touché.")
    if m["warns"]:
        for w in m["warns"]:
            print(f"  {YELLOW}⚠ {w}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def scan_market(universe, btc, workers):
    total = len(universe)
    done = [0]
    lock = threading.Lock()
    young = [0]
    errors = [0]

    def is_stable_like(m: dict) -> bool:
        # Stablecoin non répertorié : prix ≈ 1$ et volatilité quasi nulle
        r14 = m["roc14"] if np.isfinite(m["roc14"]) else 0
        return (0.9 <= m["price"] <= 1.1 and np.isfinite(m["atr_pct"])
                and m["atr_pct"] < 0.6 and abs(r14) < 2.5)

    def worker(sym_info):
        try:
            df = fetch_klines(sym_info["symbol"], "1d", 365)
            if df is None:
                errors[0] += 1
                return None
            if len(df) < 60:
                young[0] += 1
                return None
            m = compute_metrics(df, sym_info, btc)
            if m and is_stable_like(m):
                return None
            return score_coin(m) if m else None
        except Exception:
            errors[0] += 1
            return None
        finally:
            with lock:
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == total:
                    print(f"\r  Analyse du marché : {done[0]}/{total} paires…",
                          end="", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = [r for r in ex.map(worker, universe) if r is not None]
    print()
    return results, young[0], errors[0]


def to_py(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def main():
    ap = argparse.ArgumentParser(
        description="Screener crypto swing (≤ 2 semaines) — marché Binance complet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemples :
  python3 crypto_screener.py                       scan complet, top 15
  python3 crypto_screener.py --top 25 --detail 8   top élargi
  python3 crypto_screener.py --min-vol 10          volume 24h ≥ 10 M$
  python3 crypto_screener.py SOL INJ FET           analyse ciblée
  python3 crypto_screener.py --json out.json       export JSON""")
    ap.add_argument("symbols", nargs="*", help="analyse ciblée de symboles (sinon scan complet)")
    ap.add_argument("--top", type=int, default=15, help="taille du classement (défaut 15)")
    ap.add_argument("--detail", type=int, default=5, help="fiches détaillées (défaut 5)")
    ap.add_argument("--min-vol", type=float, default=5.0, help="volume 24h minimal en M$ (défaut 5)")
    ap.add_argument("--capital", type=float, default=1000.0, help="capital pour la taille de position (défaut 1000$)")
    ap.add_argument("--risk", type=float, default=1.0, help="%% du capital risqué par trade (défaut 1%%)")
    ap.add_argument("--workers", type=int, default=10, help="requêtes parallèles (défaut 10)")
    ap.add_argument("--no-4h", action="store_true", help="sauter la confluence 4h (plus rapide)")
    ap.add_argument("--json", metavar="FICHIER", help="exporter les résultats en JSON")
    args = ap.parse_args()

    t0 = time.time()
    print(f"\n{BOLD}{MAGENTA}  ◆ CRYPTO SCREENER — opportunités swing ≤ {TIME_STOP_DAYS} jours ◆{RESET}")
    print(f"  {DIM}{datetime.now().strftime('%d/%m/%Y %H:%M')} — données Binance spot (USDT){RESET}\n")

    print("  Référence BTC…", end=" ", flush=True)
    btc = btc_reference()
    if btc is None:
        print(f"{RED}échec — API Binance injoignable.{RESET}")
        sys.exit(1)
    print("OK")
    fng_val, fng_label = fetch_fear_greed()

    targeted = bool(args.symbols)
    if targeted:
        universe = []
        for s in args.symbols:
            sym = s.upper()
            if not sym.endswith("USDT"):
                sym += "USDT"
            t = get_json("/api/v3/ticker/24hr", {"symbol": sym})
            if t is None:
                print(f"  {RED}✗ {sym} introuvable sur Binance.{RESET}")
                continue
            universe.append({"symbol": sym, "base": sym[:-4],
                             "vol24": float(t.get("quoteVolume", 0)),
                             "change24": float(t.get("priceChangePercent", 0))})
        if not universe:
            sys.exit(1)
    else:
        print("  Construction de l'univers…", end=" ", flush=True)
        universe, _, total_usdt = get_universe(args.min_vol * 1e6)
        if universe is None:
            print(f"{RED}échec — API Binance injoignable.{RESET}")
            sys.exit(1)
        print(f"{len(universe)} paires retenues "
              f"(sur {total_usdt} paires USDT actives, filtre vol ≥ {args.min_vol:g} M$)")

    results, young, errors = scan_market(universe, btc, args.workers)
    if not results:
        print(f"{RED}  Aucun actif analysable.{RESET}")
        sys.exit(1)

    # Breadth (santé du marché) — uniquement en mode scan complet
    breadth = None
    if not targeted and len(results) >= 30:
        above = sum(1 for m in results if m["price"] > m["ema50"])
        breadth = 100 * above / len(results)

    regime_label, regime_color, factor = market_regime(btc, breadth, fng_val)
    for m in results:
        m["final_score"] = max(0.0, min(100.0, m["base_score"] * factor))

    results.sort(key=lambda m: m["final_score"], reverse=True)
    top_n = results if targeted else results[:max(args.top, args.detail)]

    # Confluence 4h sur le top uniquement
    if not args.no_4h:
        print(f"  Confluence 4h sur le top {len(top_n)}…", end=" ", flush=True)
        with ThreadPoolExecutor(max_workers=min(8, args.workers)) as ex:
            confs = list(ex.map(lambda m: confluence_4h(m["symbol"]), top_n))
        for m, (bonus, s) in zip(top_n, confs):
            m["bonus4h"] = bonus
            m["conf4h_str"] = s
            m["final_score"] = max(0.0, min(100.0, m["final_score"] + bonus))
        top_n.sort(key=lambda m: m["final_score"], reverse=True)
        print("OK")

    print_regime(btc, regime_label, regime_color, factor, breadth, fng_val, fng_label)
    print_table(top_n if targeted else top_n[:args.top])

    n_cards = len(top_n) if targeted else min(args.detail, len(top_n))
    for i in range(n_cards):
        m = top_n[i]
        plan = build_plan(m, args.capital, args.risk)
        print_card(i + 1, m, plan, args.capital, args.risk)

    dur = time.time() - t0
    print(f"\n{BOLD}{'═' * 78}{RESET}")
    print(f"  {DIM}Analysées : {len(results)}  |  Listings récents ignorés : {young}"
          f"  |  Erreurs : {errors}  |  Durée : {dur:.0f}s{RESET}")
    print(f"  {DIM}⚠ Outil éducatif. Les performances passées ne préjugent pas des "
          f"performances futures.{RESET}")
    print(f"  {DIM}  Aucun signal ne dispense d'une gestion du risque stricte "
          f"(stop-loss systématique).{RESET}")
    print(f"{BOLD}{'═' * 78}{RESET}\n")

    if args.json:
        export = {
            "generated_at": datetime.now().isoformat(),
            "regime": {"label": regime_label, "factor": factor,
                       "breadth": breadth, "fear_greed": fng_val},
            "results": [
                {k: v for k, v in m.items() if k not in ("conf4h_str",)}
                for m in top_n
            ],
        }
        with open(args.json, "w") as f:
            json.dump(export, f, indent=2, ensure_ascii=False, default=to_py)
        print(f"  Résultats exportés → {args.json}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}  Interrompu.{RESET}")
        sys.exit(130)
