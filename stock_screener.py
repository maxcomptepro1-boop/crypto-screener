#!/usr/bin/env python3
"""
STOCK SCREENER — Détecteur d'opportunités swing actions (positions ≤ 2 semaines)
Scanne un univers d'actions (S&P 500 par défaut) via Yahoo Finance et classe
chaque titre selon un score composite multi-piliers fondé sur des concepts
éprouvés, TECHNIQUES *et* FONDAMENTAUX :

  TENDANCE     : EMA 20/50/200, ADX/DMI (Wilder), Ichimoku, Supertrend,
                 Trend Template de Minervini
  MOMENTUM     : ROC 1/3/6 mois, force relative vs S&P 500 (RS rating O'Neil),
                 MACD, effet plus-haut-52-semaines (George & Hwang)
  TIMING       : RSI, stochastique, breakout Donchian (Turtle Traders),
                 pullback sur EMA20, squeeze Bollinger (TTM), divergences RSI
  VOLUME       : OBV (Granville), MFI, ratio volume haussier/baissier (Wyckoff),
                 volume relatif (RVOL), accumulation/distribution
  QUALITÉ      : liquidité ($ volume), volatilité ATR, bêta, distance résistance
  FONDAMENTAL  : valorisation (PER, PEG, P/S, P/B, EV/EBITDA), croissance
                 (CA & BPA), rentabilité (ROE, marges), santé financière
                 (dette/fonds propres, current ratio), qualité type Piotroski,
                 sentiment analystes, surprise de résultats
  RÉGIME       : tendance S&P 500 vs EMA200, breadth du marché, indice VIX

Gestion du risque : stops ATR / plus-bas swing, objectifs 2R/4R, taille de
position à risque fixe (règle du 1 %, Van Tharp), sortie temporelle 14 jours.

Usage :
    python3 stock_screener.py                       # scan S&P 500, top 15
    python3 stock_screener.py --top 25 --detail 8   # plus de résultats
    python3 stock_screener.py --universe nasdaq100  # autre univers
    python3 stock_screener.py AAPL NVDA MSFT        # analyse ciblée
    python3 stock_screener.py --capital 5000 --risk 1.5
    python3 stock_screener.py --no-fundamentals     # technique seul (rapide)
    python3 stock_screener.py --json resultats.json

Source : Yahoo Finance via yfinance (aucune clé API requise).
⚠ Outil éducatif — ne constitue pas un conseil en investissement.
"""

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

try:
    import yfinance as yf
except ImportError:
    print("Erreur : yfinance manquant. Installez-le :  pip install yfinance")
    sys.exit(1)

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN, RED, YELLOW, CYAN, MAGENTA = Fore.GREEN, Fore.RED, Fore.YELLOW, Fore.CYAN, Fore.MAGENTA
    BOLD, DIM, RESET = Style.BRIGHT, Style.DIM, Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = MAGENTA = BOLD = DIM = RESET = ""

TIME_STOP_DAYS = 14  # sortie temporelle : horizon max des positions
BENCHMARK = "^GSPC"  # S&P 500, référence de force relative et de régime

# Liste de secours si la récupération en ligne du S&P 500 échoue (méga/grandes
# capitalisations très liquides — suffisant pour un scan utile hors-ligne).
FALLBACK_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA",
    "BRK-B", "LLY", "JPM", "V", "XOM", "UNH", "MA", "COST", "HD", "PG", "JNJ",
    "WMT", "NFLX", "BAC", "CRM", "ORCL", "MRK", "ABBV", "CVX", "KO", "AMD",
    "PEP", "ADBE", "TMO", "LIN", "ACN", "MCD", "CSCO", "WFC", "ABT", "GE",
    "DHR", "IBM", "NOW", "TXN", "QCOM", "INTU", "CAT", "PM", "VZ", "ISRG",
    "AMGN", "CMCSA", "SPGI", "PFE", "UNP", "DIS", "GS", "RTX", "HON", "AMAT",
    "LOW", "NEE", "T", "BKNG", "ELV", "PGR", "SYK", "BLK", "COP", "TJX",
    "VRTX", "C", "MS", "BSX", "LMT", "ADP", "MDT", "PLD", "REGN", "ETN",
    "CB", "MU", "BX", "FI", "SCHW", "MMC", "DE", "ADI", "CI", "LRCX", "SBUX",
    "BMY", "KLAC", "GILD", "UPS", "SO", "MO", "DUK", "INTC", "PANW", "SHW",
]

NASDAQ100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA",
    "COST", "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "INTU", "QCOM",
    "TXN", "AMGN", "CMCSA", "AMAT", "ISRG", "HON", "BKNG", "VRTX", "ADP",
    "PANW", "MU", "ADI", "REGN", "LRCX", "GILD", "KLAC", "SBUX", "MELI",
    "SNPS", "CDNS", "MAR", "CRWD", "ORLY", "CTAS", "ASML", "PYPL", "ABNB",
    "FTNT", "MRVL", "DASH", "ADSK", "WDAY", "NXPI", "ROP", "PCAR", "MNST",
    "CPRT", "TEAM", "CHTR", "PAYX", "AEP", "ODFL", "FAST", "KDP", "ROST",
    "DDOG", "EA", "VRSK", "CTSH", "XEL", "LULU", "GEHC", "EXC", "KHC", "CCEP",
    "IDXX", "BKR", "TTWO", "CSGP", "ON", "MCHP", "ZS", "ANSS", "DXCM", "CDW",
    "BIIB", "GFS", "WBD", "ILMN", "MDB", "ARM", "MRNA", "DLTR", "WBA",
]

DOW30 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "JPM", "V", "UNH", "HD", "PG", "JNJ",
    "CVX", "KO", "MRK", "CSCO", "MCD", "CRM", "AXP", "GS", "CAT", "DIS",
    "IBM", "VZ", "AMGN", "HON", "BA", "NKE", "TRV", "WMT", "MMM", "DOW",
]


# ═══════════════════════════════════════════════════════════════════════════
#  UNIVERS
# ═══════════════════════════════════════════════════════════════════════════

def fetch_sp500() -> list[str]:
    """Constituants du S&P 500 depuis Wikipedia, avec liste de secours."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        syms = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False)
        out = [s.strip().upper() for s in syms if s.strip()]
        return out if len(out) > 400 else FALLBACK_SP500
    except Exception:
        return FALLBACK_SP500


def get_universe(name: str) -> tuple[list[str], str]:
    name = name.lower()
    if name in ("sp500", "s&p500", "spx"):
        return fetch_sp500(), "S&P 500"
    if name in ("nasdaq100", "ndx", "nasdaq"):
        return NASDAQ100, "Nasdaq 100"
    if name in ("dow", "dow30", "djia"):
        return DOW30, "Dow Jones 30"
    return fetch_sp500(), "S&P 500"


# ═══════════════════════════════════════════════════════════════════════════
#  TÉLÉCHARGEMENT DES DONNÉES (OHLCV journalier + fondamentaux)
# ═══════════════════════════════════════════════════════════════════════════

def download_history(tickers: list[str], period="1y") -> dict[str, pd.DataFrame]:
    """Télécharge l'historique journalier en un appel groupé (rapide, robuste)."""
    data = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    out: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t].dropna(how="all").reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            if _valid_ohlcv(df):
                out[t] = df[["open", "high", "low", "close", "volume"]].astype(float)
    else:  # un seul ticker → colonnes simples
        df = data.dropna(how="all").reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        if _valid_ohlcv(df):
            out[tickers[0]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return out


def _valid_ohlcv(df: pd.DataFrame) -> bool:
    need = {"open", "high", "low", "close", "volume"}
    return need.issubset(df.columns) and len(df.dropna(subset=["close"])) >= 60


def fetch_fundamentals(ticker: str) -> dict:
    """Récupère les ratios fondamentaux d'un titre (best effort)."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    keys = (
        "trailingPE", "forwardPE", "pegRatio", "priceToSalesTrailing12Months",
        "priceToBook", "enterpriseToEbitda", "returnOnEquity", "profitMargins",
        "operatingMargins", "grossMargins", "revenueGrowth", "earningsGrowth",
        "earningsQuarterlyGrowth", "debtToEquity", "currentRatio", "quickRatio",
        "freeCashflow", "operatingCashflow", "totalRevenue", "recommendationMean",
        "numberOfAnalystOpinions", "targetMeanPrice", "currentPrice", "beta",
        "marketCap", "sector", "shortName", "dividendYield", "returnOnAssets",
    )
    return {k: info.get(k) for k in keys}


def vix_level() -> float | None:
    try:
        df = yf.download("^VIX", period="1mo", interval="1d",
                         auto_adjust=True, progress=False)
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return None


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
    typ = (df["high"] + df["low"] + df["close"]) / 3
    rmf = typ * df["volume"]
    pos = rmf.where(typ > typ.shift(), 0.0).rolling(period).sum()
    neg = rmf.where(typ < typ.shift(), 0.0).rolling(period).sum()
    val = (100 - 100 / (1 + pos / neg.replace(0, np.nan))).iloc[-1]
    return float(val) if np.isfinite(val) else 50.0


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def ichimoku_position(df: pd.DataFrame) -> int | None:
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


def detect_divergence(close, low, high, rsi_arr, lookback=60) -> str | None:
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
#  CALCUL DES MÉTRIQUES TECHNIQUES
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


def compute_metrics(ticker: str, df: pd.DataFrame, bench: dict) -> dict | None:
    if df is None or len(df) < 60:
        return None
    c, h, l, v, o = df["close"], df["high"], df["low"], df["volume"], df["open"]
    price = float(c.iloc[-1])
    if price <= 0:
        return None

    dollar_vol = float((c * v).iloc[-20:].mean())  # liquidité = $ échangés/jour
    m = {"symbol": ticker, "price": price, "hist_len": len(df),
         "dollar_vol": dollar_vol,
         "change24": roc(c, 1), "vol24": dollar_vol}

    # — Tendance —
    e20, e50 = ema(c, 20), ema(c, 50)
    m["ema20"], m["ema50"] = _f(e20.iloc[-1]), _f(e50.iloc[-1])
    m["ema50_up"] = bool(e50.iloc[-1] > e50.iloc[-6]) if len(df) > 6 else False
    m["ema200"] = _f(ema(c, 200).iloc[-1]) if len(df) >= 200 else np.nan
    m["ema200_up"] = bool(ema(c, 200).iloc[-1] > ema(c, 200).iloc[-21]) if len(df) >= 221 else False
    m["adx"], m["pdi"], m["mdi"] = adx_dmi(df)
    m["ichimoku"] = ichimoku_position(df)
    m["supertrend"] = supertrend_dir(df)

    # Trend Template de Minervini (8 conditions de tendance de fond)
    low52, high52 = float(l.iloc[-252:].min()), float(h.iloc[-252:].max())
    mt = 0
    if np.isfinite(m["ema200"]):
        if price > m["ema50"] > m["ema200"]:
            mt += 2
        if m["ema200_up"]:
            mt += 1
        if price >= low52 * 1.30:
            mt += 1
        if price >= high52 * 0.75:
            mt += 1
        if m["ema20"] > m["ema50"]:
            mt += 1
    m["minervini"] = mt  # 0..6
    m["low52"], m["high52"] = low52, high52

    # — Momentum —
    m["roc21"], m["roc63"], m["roc126"] = roc(c, 21), roc(c, 63), roc(c, 126)
    m["rs21"] = _f(m["roc21"] - bench["roc21"])
    m["rs63"] = _f(m["roc63"] - bench["roc63"])
    macd_l, macd_s, hist = macd(c)
    m["macd_above"] = bool(macd_l.iloc[-1] > macd_s.iloc[-1])
    m["hist_rising"] = bool(hist.iloc[-1] > hist.iloc[-2])
    m["hist_cross_recent"] = bool(hist.iloc[-1] > 0 and (hist.iloc[-4:-1] <= 0).any())
    m["prox_high"] = price / high52 if high52 > 0 else np.nan

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
    m["prev20high"] = _f(prev20, high52)
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
#  SCORING DES PILIERS TECHNIQUES
# ═══════════════════════════════════════════════════════════════════════════

def score_technical(m: dict) -> dict:
    tags, warns = [], []
    ok, ko = "✓", "✗"

    # ── TENDANCE ──────────────────────────────────────────────────────────────
    p, d = 0, []
    if m["price"] > m["ema20"]:
        p += 12; d.append(f"P>EMA20{ok}")
    else:
        d.append(f"P<EMA20{ko}")
    if m["ema20"] > m["ema50"]:
        p += 12; d.append(f"EMA20>50{ok}")
    else:
        d.append(f"EMA20<50{ko}")
    if np.isfinite(m["ema200"]):
        if m["ema50"] > m["ema200"]:
            p += 13; d.append(f"EMA50>200{ok}")
        else:
            d.append(f"EMA50<200{ko}")
    elif m["price"] > m["ema50"]:
        p += 7
    adx_v = m["adx"] if np.isfinite(m["adx"]) else 0
    if m["pdi"] > m["mdi"]:
        if adx_v >= 25:
            p += 18
        elif adx_v >= 20:
            p += 11
        elif adx_v >= 15:
            p += 5
        d.append(f"ADX {adx_v:.0f}{ok if adx_v >= 20 else '·'}")
    else:
        d.append(f"ADX {adx_v:.0f} baissier{ko}")
    if m["ichimoku"] == 1:
        p += 13; d.append(f"Ichimoku{ok}")
    elif m["ichimoku"] == 0:
        p += 6; d.append("Ichimoku·nuage")
    elif m["ichimoku"] == -1:
        d.append(f"Ichimoku{ko}")
    if m["supertrend"] == 1:
        p += 15; d.append(f"Supertrend{ok}")
    else:
        d.append(f"Supertrend{ko}")
    mt = m["minervini"]
    p += int(round(mt / 6 * 17))
    d.append(f"Minervini {mt}/6{ok if mt >= 4 else '·'}")
    s_trend, d_trend = min(p, 100), "  ".join(d)

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    p, d = 0, []
    r21, r63, r126 = m["roc21"], m["roc63"], m["roc126"]
    if np.isfinite(r21) and r21 > 0:
        p += 10 + (5 if r21 > 8 else 0)
    if np.isfinite(r63) and r63 > 0:
        p += 10 + (5 if r63 > 15 else 0)
    d.append(f"1M {r21:+.1f}%" if np.isfinite(r21) else "1M n/d")
    d.append(f"3M {r63:+.1f}%" if np.isfinite(r63) else "3M n/d")
    if np.isfinite(m["rs63"]) and m["rs63"] > 0:
        p += 15 + (10 if m["rs63"] > 10 else 0)
        d.append(f"RS vs S&P {m['rs63']:+.1f}{ok}")
    else:
        d.append(f"RS vs S&P {m['rs63']:+.1f}{ko}" if np.isfinite(m["rs63"]) else "RS n/d")
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
        d.append(f"Haut52s {prox * 100:.0f}%")
    if np.isfinite(r126) and r126 > 0:
        p += 10
    s_mom, d_mom = min(p, 100), "  ".join(d)

    # ── TIMING / SETUPS ───────────────────────────────────────────────────────
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

    # ── VOLUME / ACCUMULATION ─────────────────────────────────────────────────
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

    # ── QUALITÉ / RISQUE ──────────────────────────────────────────────────────
    p, d = 0, []
    dv = m["dollar_vol"]
    if dv >= 200e6:
        p += 30
    elif dv >= 50e6:
        p += 24
    elif dv >= 20e6:
        p += 18
    elif dv >= 5e6:
        p += 12
    else:
        p += 5
        warns.append("Liquidité faible (< 5 M$/j échangés)")
    d.append(f"Liq {human_vol(dv)}/j")
    ap = m["atr_pct"]
    if np.isfinite(ap):
        if 1.5 <= ap <= 5:
            p += 30
        elif 1 <= ap < 1.5 or 5 < ap <= 8:
            p += 18
        elif ap < 1:
            p += 8
        else:
            p += 5
            warns.append(f"Volatilité élevée (ATR {ap:.1f}%/j)")
        d.append(f"ATR {ap:.1f}%/j")
    hl = m["hist_len"]
    if hl >= 200:
        p += 20
    elif hl >= 120:
        p += 14
    else:
        p += 8
        warns.append(f"Cotation récente ({hl} j d'historique)")
    d.append(f"Hist {hl}j")
    if m["donchian_break"]:
        p += 20; d.append(f"Ciel dégagé{ok}")
    elif m["dist_res_atr"] >= 1.5:
        p += 12; d.append(f"Résist. à {m['dist_res_atr']:.1f}×ATR")
    else:
        p += 5; d.append("Résist. proche")
    s_qual, d_qual = min(p, 100), "  ".join(d)

    m.update({
        "s_trend": s_trend, "s_mom": s_mom, "s_tim": s_tim,
        "s_vol": s_vol, "s_qual": s_qual,
        "d_trend": d_trend, "d_mom": d_mom, "d_tim": d_tim,
        "d_vol": d_vol, "d_qual": d_qual,
        "tags": tags, "warns": warns,
    })
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  SCORING DU PILIER FONDAMENTAL
# ═══════════════════════════════════════════════════════════════════════════

def score_fundamental(m: dict, fund: dict) -> None:
    """Note /100 la santé fondamentale ; alimente m['s_fund'] et m['d_fund']."""
    if not fund:
        m["s_fund"], m["d_fund"], m["has_fund"] = None, "fondamentaux indisponibles", False
        return

    def g(k):
        return _f(fund.get(k))

    p, d = 0.0, []
    warns = m["warns"]

    # — Valorisation (30 pts) — moins c'est cher, mieux c'est —
    vp = 0
    pe = g("trailingPE")
    if np.isfinite(pe) and pe > 0:
        if pe < 15:
            vp += 10
        elif pe < 25:
            vp += 7
        elif pe < 35:
            vp += 4
        elif pe > 60:
            warns.append(f"Valorisation tendue (PER {pe:.0f})")
        d.append(f"PER {pe:.0f}")
    peg = g("pegRatio")
    if np.isfinite(peg) and peg > 0:
        if peg < 1:
            vp += 10
        elif peg < 1.5:
            vp += 7
        elif peg < 2.5:
            vp += 3
        d.append(f"PEG {peg:.2f}")
    ps = g("priceToSalesTrailing12Months")
    if np.isfinite(ps) and ps > 0:
        if ps < 3:
            vp += 5
        elif ps < 8:
            vp += 2
        d.append(f"P/S {ps:.1f}")
    ev = g("enterpriseToEbitda")
    if np.isfinite(ev) and ev > 0:
        if ev < 12:
            vp += 5
        elif ev < 20:
            vp += 2
        d.append(f"EV/EBITDA {ev:.0f}")
    p += min(vp, 30)

    # — Croissance (25 pts) —
    gp = 0
    rg = g("revenueGrowth")
    if np.isfinite(rg):
        if rg > 0.20:
            gp += 13
        elif rg > 0.10:
            gp += 9
        elif rg > 0.03:
            gp += 5
        elif rg < 0:
            warns.append(f"Chiffre d'affaires en recul ({rg * 100:.0f}%)")
        d.append(f"Croiss.CA {rg * 100:+.0f}%")
    eg = g("earningsGrowth") if np.isfinite(g("earningsGrowth")) else g("earningsQuarterlyGrowth")
    if np.isfinite(eg):
        if eg > 0.20:
            gp += 12
        elif eg > 0.08:
            gp += 8
        elif eg > 0:
            gp += 4
        elif eg < 0:
            warns.append(f"Bénéfices en recul ({eg * 100:.0f}%)")
        d.append(f"Croiss.BPA {eg * 100:+.0f}%")
    p += min(gp, 25)

    # — Rentabilité (25 pts) —
    pp = 0
    roe = g("returnOnEquity")
    if np.isfinite(roe):
        if roe > 0.20:
            pp += 10
        elif roe > 0.12:
            pp += 7
        elif roe > 0.05:
            pp += 3
        d.append(f"ROE {roe * 100:.0f}%")
    pm = g("profitMargins")
    if np.isfinite(pm):
        if pm > 0.20:
            pp += 8
        elif pm > 0.10:
            pp += 5
        elif pm > 0:
            pp += 2
        else:
            warns.append("Société non rentable (marge nette < 0)")
        d.append(f"Marge {pm * 100:.0f}%")
    om = g("operatingMargins")
    if np.isfinite(om) and om > 0.15:
        pp += 4
    fcf = g("freeCashflow")
    if np.isfinite(fcf) and fcf > 0:
        pp += 3; d.append("FCF+")
    p += min(pp, 25)

    # — Santé financière (20 pts) —
    hp = 0
    de = g("debtToEquity")
    if np.isfinite(de):
        if de < 50:
            hp += 10
        elif de < 100:
            hp += 6
        elif de < 200:
            hp += 2
        else:
            warns.append(f"Endettement élevé (dette/FP {de:.0f}%)")
        d.append(f"Dette/FP {de:.0f}%")
    cr = g("currentRatio")
    if np.isfinite(cr):
        if cr > 2:
            hp += 6
        elif cr > 1.2:
            hp += 4
        elif cr < 1:
            warns.append(f"Liquidité court terme faible (current ratio {cr:.1f})")
        d.append(f"Ratio liq. {cr:.1f}")
    roa = g("returnOnAssets")
    if np.isfinite(roa) and roa > 0.05:
        hp += 4
    p += min(hp, 20)

    score = min(p, 100)

    # — Sentiment analystes : modulateur léger (±, hors 100 de base) —
    rec = g("recommendationMean")  # 1 = strong buy … 5 = strong sell
    tgt, cur = g("targetMeanPrice"), m["price"]
    extra = []
    if np.isfinite(rec):
        if rec <= 2.0:
            score = min(100, score + 4)
        elif rec >= 3.5:
            score = max(0, score - 4)
        extra.append(f"Avis {rec:.1f}/5")
    if np.isfinite(tgt) and cur > 0:
        upside = (tgt / cur - 1) * 100
        m["analyst_upside"] = upside
        extra.append(f"Cible {upside:+.0f}%")
        if upside > 15:
            score = min(100, score + 3)
    if extra:
        d.append(" · ".join(extra))

    m["s_fund"] = score
    m["d_fund"] = "  ".join(d) if d else "données partielles"
    m["has_fund"] = True
    m["sector"] = fund.get("sector") or "—"


# ═══════════════════════════════════════════════════════════════════════════
#  LENTILLES DES GRANDS INVESTISSEURS
#  Chaque « légende » est une grille de lecture éprouvée. Quand plusieurs
#  s'alignent sur un même titre, la conviction monte (consensus de méthodes).
# ═══════════════════════════════════════════════════════════════════════════

def investor_lenses(m: dict, fund: dict | None) -> None:
    """Évalue le titre selon les doctrines de grands investisseurs.
    Alimente m['lenses'] (badges validés) et m['lens_notes'] (explications)."""
    passed, notes = [], []
    f = fund or {}

    def g(k):
        return _f(f.get(k))

    pe = g("trailingPE")
    pb = g("priceToBook")
    peg = g("pegRatio")
    ev = g("enterpriseToEbitda")
    roe = g("returnOnEquity")
    roa = g("returnOnAssets")
    pm = g("profitMargins")
    gm = g("grossMargins")
    rg = g("revenueGrowth")
    eg = g("earningsGrowth") if np.isfinite(g("earningsGrowth")) else g("earningsQuarterlyGrowth")
    de = g("debtToEquity")
    cr = g("currentRatio")
    fcf = g("freeCashflow")
    up = m.get("analyst_upside", np.nan)

    # — Warren Buffett / Charlie Munger : qualité durable (douve économique) —
    if (np.isfinite(roe) and roe > 0.15 and np.isfinite(pm) and pm > 0.12
            and (not np.isfinite(de) or de < 120) and (not np.isfinite(fcf) or fcf > 0)
            and (not np.isfinite(pe) or pe < 35)):
        passed.append("Buffett")
        notes.append("Buffett : rentabilité élevée + douve + dette maîtrisée")

    # — Benjamin Graham : valeur défensive & marge de sécurité —
    graham_pe_pb = (pe * pb) if (np.isfinite(pe) and np.isfinite(pb)) else np.inf
    if (np.isfinite(pe) and pe < 18 and np.isfinite(pb) and pb < 2.5
            and graham_pe_pb < 30 and (not np.isfinite(cr) or cr > 1.3)
            and (not np.isfinite(de) or de < 120)):
        passed.append("Graham")
        notes.append("Graham : décote (PER×P/B faible) + bilan solide")

    # — Peter Lynch : croissance à prix raisonnable (GARP), le PEG roi —
    if np.isfinite(peg) and 0 < peg < 1.3 and (not np.isfinite(eg) or eg > 0.08):
        passed.append("Lynch")
        notes.append(f"Lynch (GARP) : PEG {peg:.2f} < 1.3, croissance payée à bon prix")

    # — Joel Greenblatt (Magic Formula) : rendement élevé + capital rentable —
    earn_yield = (1 / ev) if (np.isfinite(ev) and ev > 0) else np.nan
    if (np.isfinite(earn_yield) and earn_yield > 0.08
            and ((np.isfinite(roe) and roe > 0.20) or (np.isfinite(roa) and roa > 0.10))):
        passed.append("Greenblatt")
        notes.append("Greenblatt (Magic Formula) : rendement bénéficiaire + ROIC élevés")

    # — William O'Neil (CAN SLIM) : leader, proche du plus-haut, en croissance —
    if (np.isfinite(m["rs63"]) and m["rs63"] > 0 and np.isfinite(m["prox_high"])
            and m["prox_high"] >= 0.90 and (not np.isfinite(eg) or eg > 0.15)):
        passed.append("O'Neil")
        notes.append("O'Neil (CAN SLIM) : leader proche du plus-haut, force relative +")

    # — Mark Minervini (SEPA / Trend Template) : tendance de fond confirmée —
    if m["minervini"] >= 5:
        passed.append("Minervini")
        notes.append(f"Minervini : Trend Template {m['minervini']}/6")

    # — Philip Fisher : croissance de qualité (marges + chiffre d'affaires) —
    if np.isfinite(rg) and rg > 0.15 and np.isfinite(gm) and gm > 0.40:
        passed.append("Fisher")
        notes.append("Fisher : forte croissance + marges brutes élevées")

    # — Templeton / Dreman : contrarien, acheter au pessimisme maximal —
    if (m["rsi"] < 42 and np.isfinite(pe) and pe < 18
            and (not np.isfinite(up) or up > 12)):
        passed.append("Templeton")
        notes.append("Templeton (contrarien) : survendu + valorisation basse")

    # — Soros / Druckenmiller : tendance puissante portée par le momentum —
    if (m["supertrend"] == 1 and np.isfinite(m["adx"]) and m["adx"] >= 25
            and np.isfinite(m["rs63"]) and m["rs63"] > 5):
        passed.append("Druckenmiller")
        notes.append("Soros/Druckenmiller : tendance forte (ADX≥25) + surperformance")

    # — Piotroski : robustesse financière (proxy F-Score) —
    fscore = 0
    if np.isfinite(roa) and roa > 0:
        fscore += 1
    if np.isfinite(fcf) and fcf > 0:
        fscore += 1
    if np.isfinite(pm) and pm > 0:
        fscore += 1
    if np.isfinite(de) and de < 80:
        fscore += 1
    if np.isfinite(cr) and cr > 1:
        fscore += 1
    m["piotroski"] = fscore  # 0..5
    if fscore >= 4:
        passed.append("Piotroski")
        notes.append(f"Piotroski : robustesse financière {fscore}/5")

    m["lenses"] = passed
    m["lens_notes"] = notes


# ═══════════════════════════════════════════════════════════════════════════
#  AGRÉGATION DU SCORE
# ═══════════════════════════════════════════════════════════════════════════

def combine_score(m: dict, use_fund: bool) -> None:
    tech = (0.26 * m["s_trend"] + 0.26 * m["s_mom"] + 0.20 * m["s_tim"]
            + 0.14 * m["s_vol"] + 0.14 * m["s_qual"])
    if use_fund and m.get("has_fund") and m["s_fund"] is not None:
        # 75 % technique (timing du swing) + 25 % fondamental (qualité du sous-jacent)
        base = 0.75 * tech + 0.25 * m["s_fund"]
    else:
        base = tech
    # Consensus des légendes : chaque doctrine validée ajoute de la conviction.
    n_lens = len(m.get("lenses", []))
    m["lens_bonus"] = min(n_lens * 2.0, 12.0)
    m["base_score"] = min(100.0, base + m["lens_bonus"])


# ═══════════════════════════════════════════════════════════════════════════
#  RÉGIME DE MARCHÉ
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_reference(period="1y") -> dict | None:
    try:
        df = yf.download(BENCHMARK, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        df = df.dropna()
        if len(df) < 200:
            return None
        c = df["Close"].squeeze()
        return {
            "price": float(c.iloc[-1]),
            "roc21": roc(c, 21), "roc63": roc(c, 63),
            "ema50": float(ema(c, 50).iloc[-1]),
            "ema200": float(ema(c, 200).iloc[-1]),
        }
    except Exception:
        return None


def market_regime(bench: dict, breadth: float | None, vix: float | None):
    p, e50, e200 = bench["price"], bench["ema50"], bench["ema200"]
    if p > e50 and e50 > e200:
        label, factor, color = "RISK-ON (S&P 500 haussier)", 1.05, GREEN
    elif p > e200:
        label, factor, color = "NEUTRE (S&P > EMA200)", 1.00, YELLOW
    elif p > e50:
        label, factor, color = "REBOND SOUS RÉSISTANCE", 0.90, YELLOW
    else:
        label, factor, color = "RISK-OFF (S&P sous EMA200)", 0.80, RED
        if np.isfinite(bench["roc63"]) and bench["roc63"] < -10:
            label, factor = "RISK-OFF FORT (marché en baisse)", 0.70
    if breadth is not None:
        if breadth >= 55:
            factor += 0.03
        elif breadth <= 30:
            factor -= 0.05
    if vix is not None:
        if vix >= 30:
            factor -= 0.05   # stress de marché → prudence
        elif vix >= 22:
            factor -= 0.02
        elif vix <= 14:
            factor += 0.02   # complaisance/calme → vent porteur
    return label, color, max(0.65, min(1.10, factor))


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
    shares = int(qty)  # actions = quantités entières
    notional = shares * price
    capped = notional > capital
    if capped:
        shares = int(capital / price)
        notional = shares * price
    return {
        "entry": price, "stop": stop, "tp1": tp1, "tp2": tp2,
        "stop_pct": (price - stop) / price * 100,
        "tp1_pct": (tp1 - price) / price * 100,
        "tp2_pct": (tp2 - price) / price * 100,
        "rr1": rr1, "rr2": rr2,
        "shares": shares, "notional": notional, "capped": capped,
        "risk_amount": risk_amount,
        "deadline": (datetime.now() + timedelta(days=TIME_STOP_DAYS)).strftime("%d/%m/%Y"),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════

def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:,.2f}"
    return f"{p:.4f}"


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


def print_regime(bench, label, color, factor, breadth, vix):
    print(f"\n{BOLD}{CYAN}{'═' * 78}{RESET}")
    print(f"{BOLD}{CYAN}  RÉGIME DE MARCHÉ{RESET}")
    print(f"{CYAN}{'─' * 78}{RESET}")
    e50ok = "✓" if bench["price"] > bench["ema50"] else "✗"
    e200ok = "✓" if bench["price"] > bench["ema200"] else "✗"
    print(f"  S&P 500 {fmt_price(bench['price'])}  |  >EMA50 {e50ok}  >EMA200 {e200ok}"
          f"  |  1M {bench['roc21']:+.1f}%  3M {bench['roc63']:+.1f}%")
    extras = []
    if breadth is not None:
        extras.append(f"Breadth: {breadth:.0f}% des titres > EMA50")
    if vix is not None:
        extras.append(f"VIX: {vix:.1f}")
    if extras:
        print("  " + "  |  ".join(extras))
    print(f"  Régime: {color}{BOLD}{label}{RESET}  →  facteur ×{factor:.2f} appliqué aux scores")
    print(f"{BOLD}{CYAN}{'═' * 78}{RESET}")


def print_table(results: list[dict], use_fund: bool):
    fcol = f"  {'FOND':>5}" if use_fund else ""
    print(f"\n{BOLD}  {'#':>2}  {'TITRE':<8} {'PRIX':>10}  {'1J':>8}  {'1M':>8}"
          f"  {'LIQ/J':>7}  {'SCORE':>5}{fcol}  {'NOTE':<11} SETUPS{RESET}")
    print(f"  {'─' * 100}")
    for i, m in enumerate(results, 1):
        note, ncol = rating(m["final_score"])
        nlen = len(m.get("lenses", []))
        lic = f"{GREEN}◆{nlen}{RESET} " if nlen else ""
        tags = lic + (", ".join(m["tags"])[:24] if m["tags"] else "—")
        fund = ""
        if use_fund:
            fv = m.get("s_fund")
            fund = f"  {fv:>5.0f}" if fv is not None else f"  {'n/d':>5}"
        print(f"  {i:>2}  {m['symbol']:<8} {fmt_price(m['price']):>10}"
              f"  {pct_color(m['change24'])}  {pct_color(m['roc21'])}"
              f"  {human_vol(m['dollar_vol']):>7}"
              f"  {BOLD}{m['final_score']:>5.1f}{RESET}{fund}"
              f"  {ncol}{note:<11}{RESET} {tags}")
    print(f"  {'─' * 100}")


def print_card(rank: int, m: dict, plan: dict, capital: float, risk_pct: float, use_fund: bool):
    note, ncol = rating(m["final_score"])
    tag_str = ("  [" + " · ".join(m["tags"]) + "]") if m["tags"] else ""
    sect = f"  {DIM}{m.get('sector', '')}{RESET}" if m.get("sector") else ""
    print(f"\n{BOLD}{'═' * 78}{RESET}")
    print(f"{BOLD}  #{rank}  {CYAN}{m['symbol']}{RESET}{BOLD}{tag_str}{RESET}{sect}")
    print(f"{'─' * 78}")
    c1 = GREEN if (np.isfinite(m["change24"]) and m["change24"] >= 0) else RED
    r1c = GREEN if (np.isfinite(m["roc21"]) and m["roc21"] >= 0) else RED
    r1s = f"{m['roc21']:+.1f}%" if np.isfinite(m["roc21"]) else "n/d"
    c1s = f"{m['change24']:+.1f}%" if np.isfinite(m["change24"]) else "n/d"
    print(f"  Prix {BOLD}{fmt_price(m['price'])} $US{RESET}"
          f"   1j {c1}{c1s}{RESET}"
          f"   1M {r1c}{r1s}{RESET}"
          f"   Liq {human_vol(m['dollar_vol'])}/j")
    fs = m["final_score"]
    extra = f"   (base {m['base_score']:.1f} ×régime)"
    print(f"  Score {BOLD}{fs:.1f}/100{RESET} [{bar10(fs)}]  →  {ncol}{BOLD}{note}{RESET}{extra}")
    print()
    print(f"  ├─ Tendance   {m['s_trend']:>3.0f} [{bar10(m['s_trend'])}]  {DIM}{m['d_trend']}{RESET}")
    print(f"  ├─ Momentum   {m['s_mom']:>3.0f} [{bar10(m['s_mom'])}]  {DIM}{m['d_mom']}{RESET}")
    print(f"  ├─ Timing     {m['s_tim']:>3.0f} [{bar10(m['s_tim'])}]  {DIM}{m['d_tim']}{RESET}")
    print(f"  ├─ Volume     {m['s_vol']:>3.0f} [{bar10(m['s_vol'])}]  {DIM}{m['d_vol']}{RESET}")
    char = "├" if use_fund else "└"
    print(f"  {char}─ Qualité    {m['s_qual']:>3.0f} [{bar10(m['s_qual'])}]  {DIM}{m['d_qual']}{RESET}")
    if use_fund:
        if m.get("s_fund") is not None:
            print(f"  └─ Fondamental{m['s_fund']:>3.0f} [{bar10(m['s_fund'])}]  {DIM}{m['d_fund']}{RESET}")
        else:
            print(f"  └─ Fondamental n/d  {DIM}{m.get('d_fund', '')}{RESET}")

    lenses = m.get("lenses", [])
    if lenses:
        badges = " ".join(f"{GREEN}◆ {x}{RESET}" for x in lenses)
        print(f"\n  {BOLD}LÉGENDES VALIDÉES ({len(lenses)}, +{m.get('lens_bonus', 0):.0f} pts){RESET}  {badges}")
        for n in m.get("lens_notes", []):
            print(f"    {DIM}• {n}{RESET}")

    print(f"\n  {BOLD}PLAN DE TRADE (horizon max {TIME_STOP_DAYS} jours){RESET}")
    print(f"    Entrée    : {fmt_price(plan['entry'])} $US")
    print(f"    Stop-loss : {RED}{fmt_price(plan['stop'])}{RESET}  (−{plan['stop_pct']:.1f}%)")
    print(f"    Objectif 1: {GREEN}{fmt_price(plan['tp1'])}{RESET}  (+{plan['tp1_pct']:.1f}%, R/R {plan['rr1']:.1f})"
          f"   Objectif 2: {GREEN}{fmt_price(plan['tp2'])}{RESET}  (+{plan['tp2_pct']:.1f}%, R/R {plan['rr2']:.1f})")
    cap_note = f"  {YELLOW}(plafonné au capital){RESET}" if plan["capped"] else ""
    print(f"    Taille    : {plan['shares']} actions ≈ {plan['notional']:.0f}$ "
          f"(capital {capital:.0f}$, risque {risk_pct:.1f}% = {plan['risk_amount']:.0f}$){cap_note}")
    dist_h = (m["high52"] / m["price"] - 1) * 100
    print(f"    Plus haut 52s : {fmt_price(m['high52'])} ({'+' if dist_h >= 0 else ''}{dist_h:.1f}%)")
    print(f"    ⏰ Date limite de vente : {plan['deadline']} — si ni stop ni objectif "
          f"touché d'ici là, tout revendre ce jour-là.")
    if m["warns"]:
        for w in m["warns"]:
            print(f"  {YELLOW}⚠ {w}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

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
        description="Screener actions swing (≤ 2 semaines) — technique + fondamental",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemples :
  python3 stock_screener.py                       scan S&P 500, top 15
  python3 stock_screener.py --universe nasdaq100  univers Nasdaq 100
  python3 stock_screener.py --top 25 --detail 8   top élargi
  python3 stock_screener.py AAPL NVDA MSFT        analyse ciblée
  python3 stock_screener.py --no-fundamentals     technique seul (rapide)
  python3 stock_screener.py --json out.json       export JSON""")
    ap.add_argument("symbols", nargs="*", help="analyse ciblée de tickers (sinon scan de l'univers)")
    ap.add_argument("--universe", default="sp500", help="sp500 (défaut), nasdaq100, dow30")
    ap.add_argument("--top", type=int, default=15, help="taille du classement (défaut 15)")
    ap.add_argument("--detail", type=int, default=5, help="fiches détaillées (défaut 5)")
    ap.add_argument("--min-liq", type=float, default=5.0, help="liquidité min en M$/jour (défaut 5)")
    ap.add_argument("--capital", type=float, default=1000.0, help="capital pour la taille de position (défaut 1000$)")
    ap.add_argument("--risk", type=float, default=1.0, help="%% du capital risqué par trade (défaut 1%%)")
    ap.add_argument("--no-fundamentals", action="store_true", help="ignorer le pilier fondamental (plus rapide)")
    ap.add_argument("--fund-workers", type=int, default=8, help="requêtes fondamentaux parallèles (défaut 8)")
    ap.add_argument("--json", metavar="FICHIER", help="exporter les résultats en JSON")
    args = ap.parse_args()

    use_fund = not args.no_fundamentals
    t0 = time.time()
    print(f"\n{BOLD}{MAGENTA}  ◆ STOCK SCREENER — opportunités swing ≤ {TIME_STOP_DAYS} jours ◆{RESET}")
    print(f"  {DIM}{datetime.now().strftime('%d/%m/%Y %H:%M')} — données Yahoo Finance{RESET}\n")

    print("  Référence S&P 500…", end=" ", flush=True)
    bench = benchmark_reference()
    if bench is None:
        print(f"{RED}échec — Yahoo Finance injoignable.{RESET}")
        sys.exit(1)
    print("OK")
    vix = vix_level()

    targeted = bool(args.symbols)
    if targeted:
        tickers = [s.upper().replace(".", "-") for s in args.symbols]
        uni_name = "ciblé"
    else:
        tickers, uni_name = get_universe(args.universe)
        print(f"  Univers : {uni_name} ({len(tickers)} titres)")

    print("  Téléchargement des historiques…", end=" ", flush=True)
    hist = download_history(tickers)
    print(f"{len(hist)} titres récupérés")
    if not hist:
        print(f"{RED}  Aucune donnée disponible.{RESET}")
        sys.exit(1)

    # Métriques techniques + filtre de liquidité
    results = []
    min_liq = args.min_liq * 1e6
    for t, df in hist.items():
        try:
            m = compute_metrics(t, df, bench)
        except Exception:
            m = None
        if m is None:
            continue
        if not targeted and m["dollar_vol"] < min_liq:
            continue
        score_technical(m)
        results.append(m)

    if not results:
        print(f"{RED}  Aucun titre analysable après filtre de liquidité.{RESET}")
        sys.exit(1)

    # Breadth
    breadth = None
    if not targeted and len(results) >= 30:
        above = sum(1 for m in results if m["price"] > m["ema50"])
        breadth = 100 * above / len(results)

    # Pré-classement technique pour limiter les appels fondamentaux au haut du panier
    for m in results:
        combine_score(m, use_fund=False)
    results.sort(key=lambda m: m["base_score"], reverse=True)

    fund_pool = results if targeted else results[:max(args.top, args.detail) + 15]
    if use_fund:
        print(f"  Fondamentaux sur le top {len(fund_pool)}…", end=" ", flush=True)
        with ThreadPoolExecutor(max_workers=args.fund_workers) as ex:
            funds = list(ex.map(lambda m: fetch_fundamentals(m["symbol"]), fund_pool))
        for m, f in zip(fund_pool, funds):
            score_fundamental(m, f)
            investor_lenses(m, f)
        print("OK")
    for m in results:
        if "s_fund" not in m:
            m["s_fund"], m["d_fund"], m["has_fund"] = None, "", False
        if "lenses" not in m:
            investor_lenses(m, None)

    # Régime + score final
    regime_label, regime_color, factor = market_regime(bench, breadth, vix)
    for m in results:
        combine_score(m, use_fund=use_fund)
        m["final_score"] = max(0.0, min(100.0, m["base_score"] * factor))

    results.sort(key=lambda m: m["final_score"], reverse=True)
    top_n = results if targeted else results[:max(args.top, args.detail)]

    print_regime(bench, regime_label, regime_color, factor, breadth, vix)
    print_table(top_n if targeted else top_n[:args.top], use_fund)

    n_cards = len(top_n) if targeted else min(args.detail, len(top_n))
    for i in range(n_cards):
        m = top_n[i]
        plan = build_plan(m, args.capital, args.risk)
        print_card(i + 1, m, plan, args.capital, args.risk, use_fund)

    dur = time.time() - t0
    print(f"\n{BOLD}{'═' * 78}{RESET}")
    print(f"  {DIM}Analysés : {len(results)}  |  Univers : {uni_name}"
          f"  |  Durée : {dur:.0f}s{RESET}")
    print(f"  {DIM}⚠ Outil éducatif. Les performances passées ne préjugent pas des "
          f"performances futures.{RESET}")
    print(f"  {DIM}  Aucun signal ne dispense d'une gestion du risque stricte "
          f"(stop-loss systématique).{RESET}")
    print(f"{BOLD}{'═' * 78}{RESET}\n")

    if args.json:
        export = {
            "generated_at": datetime.now().isoformat(),
            "universe": uni_name,
            "regime": {"label": regime_label, "factor": factor,
                       "breadth": breadth, "vix": vix},
            "results": top_n,
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
