#!/usr/bin/env python3
"""
STOCK INVESTOR — Sélecteur d'actions QUALITÉ + CROISSANCE pour le LONG TERME
(horizon 6 mois à plusieurs années, philosophie Buffett / Lynch / Fisher).

À l'inverse du swing (stock_screener.py), cet outil ne cherche pas un point
d'entrée à 2 semaines : il identifie d'EXCELLENTES SOCIÉTÉS à garder dans la
durée, et fournit pour chacune la thèse d'investissement + les signaux qui
devraient faire reconsidérer la position (pas de stop serré).

  QUALITÉ      (30 %) : ROE, ROA, marges brute/op./nette, FCF, douve
  CROISSANCE   (25 %) : croissance du chiffre d'affaires et des bénéfices
  SANTÉ        (15 %) : dette/fonds propres, liquidité, trésorerie vs dette
  VALORISATION (20 %) : PER, PEG, P/B, P/S, EV/EBITDA, rendement du FCF — on
                        ne cherche pas « bon marché », mais « pas absurde »
  TENDANCE LT  (10 %) : prix > moyenne 200 j, parcours 6-12 mois (éviter de
                        rattraper un couteau qui tombe)
  DOCTRINES           : Buffett, Lynch, Fisher, Greenblatt, Piotroski
                        (bonus de conviction quand elles s'alignent)

Sortie : classement, thèse « pourquoi la détenir », allocation suggérée d'un
portefeuille, signaux de vente, et date de prochaine revue trimestrielle.

Usage :
    python3 stock_investor.py                       # S&P 500, top 20
    python3 stock_investor.py --universe nasdaq100
    python3 stock_investor.py AAPL MSFT GOOGL       # analyse ciblée
    python3 stock_investor.py --portfolio 12 --capital 10000
    python3 stock_investor.py --json out.json

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

REVIEW_DAYS = 91  # revue trimestrielle (≈ 3 mois)

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
#  UNIVERS & DONNÉES
# ═══════════════════════════════════════════════════════════════════════════

def fetch_sp500() -> list[str]:
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
    if name in ("nasdaq100", "ndx", "nasdaq"):
        return NASDAQ100, "Nasdaq 100"
    if name in ("dow", "dow30", "djia"):
        return DOW30, "Dow Jones 30"
    return fetch_sp500(), "S&P 500"


def download_history(tickers: list[str], period="2y") -> dict[str, pd.DataFrame]:
    data = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    out: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t].dropna(how="all").reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            if "close" in df.columns and len(df.dropna(subset=["close"])) >= 60:
                out[t] = df[["close"]].astype(float)
    else:
        df = data.dropna(how="all").reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        if "close" in df.columns and len(df.dropna(subset=["close"])) >= 60:
            out[tickers[0]] = df[["close"]].astype(float)
    return out


def fetch_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    keys = (
        "trailingPE", "forwardPE", "pegRatio", "trailingPegRatio",
        "priceToSalesTrailing12Months", "priceToBook", "enterpriseToEbitda",
        "enterpriseToRevenue", "returnOnEquity", "returnOnAssets",
        "profitMargins", "operatingMargins", "grossMargins", "ebitdaMargins",
        "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
        "revenueQuarterlyGrowth", "debtToEquity", "currentRatio", "quickRatio",
        "freeCashflow", "operatingCashflow", "totalCash", "totalDebt",
        "totalRevenue", "marketCap", "recommendationMean", "numberOfAnalystOpinions",
        "targetMeanPrice", "currentPrice", "beta", "dividendYield", "payoutRatio",
        "fiveYearAvgDividendYield", "sector", "industry", "shortName",
    )
    return {k: info.get(k) for k in keys}


# ═══════════════════════════════════════════════════════════════════════════
#  TENDANCE LONG TERME (technique léger)
# ═══════════════════════════════════════════════════════════════════════════

def _f(x, default=np.nan) -> float:
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def roc(c: pd.Series, n: int) -> float:
    if len(c) <= n:
        return np.nan
    return _f((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100)


def trend_metrics(df: pd.DataFrame) -> dict:
    c = df["close"]
    price = float(c.iloc[-1])
    sma200 = _f(sma(c, 200).iloc[-1]) if len(c) >= 200 else np.nan
    sma200_prev = _f(sma(c, 200).iloc[-22]) if len(c) >= 222 else np.nan
    high52 = float(c.iloc[-252:].max())
    return {
        "price": price,
        "above_200": (np.isfinite(sma200) and price > sma200),
        "sma200_up": (np.isfinite(sma200) and np.isfinite(sma200_prev) and sma200 > sma200_prev),
        "perf6m": roc(c, 126), "perf12m": roc(c, 252),
        "prox_high": price / high52 if high52 > 0 else np.nan,
        "high52": high52,
    }


def score_trend_lt(tr: dict) -> tuple[float, str]:
    p, d = 0, []
    if tr["above_200"]:
        p += 35; d.append("Au-dessus moyenne 200j✓")
    else:
        d.append("Sous moyenne 200j✗")
    if tr["sma200_up"]:
        p += 20; d.append("Moyenne 200j en hausse✓")
    p6, p12 = tr["perf6m"], tr["perf12m"]
    if np.isfinite(p12):
        if p12 > 0:
            p += 20
        d.append(f"12 mois {p12:+.0f}%")
    if np.isfinite(p6) and p6 > 0:
        p += 15
    prox = tr["prox_high"]
    if np.isfinite(prox):
        if prox >= 0.80:
            p += 10
        elif prox < 0.55:
            d.append("loin du plus-haut")
    return min(p, 100), "  ".join(d)


# ═══════════════════════════════════════════════════════════════════════════
#  PILIERS FONDAMENTAUX
# ═══════════════════════════════════════════════════════════════════════════

def score_quality(f: dict, warns: list) -> tuple[float, str]:
    g = lambda k: _f(f.get(k))
    p, d = 0.0, []
    roe = g("returnOnEquity")
    if np.isfinite(roe):
        if roe > 0.25:
            p += 26
        elif roe > 0.15:
            p += 20
        elif roe > 0.10:
            p += 12
        elif roe > 0:
            p += 5
        else:
            warns.append("Rentabilité des fonds propres négative")
        d.append(f"ROE {roe * 100:.0f}%")
    roa = g("returnOnAssets")
    if np.isfinite(roa):
        if roa > 0.12:
            p += 12
        elif roa > 0.06:
            p += 8
        elif roa > 0:
            p += 3
        d.append(f"ROA {roa * 100:.0f}%")
    gm = g("grossMargins")
    if np.isfinite(gm):
        if gm > 0.50:
            p += 14
        elif gm > 0.35:
            p += 9
        elif gm > 0.20:
            p += 4
        d.append(f"Marge brute {gm * 100:.0f}%")
    om = g("operatingMargins")
    if np.isfinite(om):
        if om > 0.25:
            p += 14
        elif om > 0.15:
            p += 9
        elif om > 0.05:
            p += 4
        elif om < 0:
            warns.append("Marge opérationnelle négative")
        d.append(f"Marge op. {om * 100:.0f}%")
    pm = g("profitMargins")
    if np.isfinite(pm):
        if pm > 0.20:
            p += 14
        elif pm > 0.10:
            p += 9
        elif pm > 0:
            p += 4
        d.append(f"Marge nette {pm * 100:.0f}%")
    fcf = g("freeCashflow")
    if np.isfinite(fcf):
        if fcf > 0:
            p += 20; d.append("FCF positif✓")
        else:
            warns.append("Flux de trésorerie disponible négatif")
            d.append("FCF négatif✗")
    return min(p, 100), "  ".join(d)


def score_growth(f: dict, warns: list) -> tuple[float, str]:
    g = lambda k: _f(f.get(k))
    p, d = 0.0, []
    rg = g("revenueGrowth")
    if np.isfinite(rg):
        if rg > 0.25:
            p += 40
        elif rg > 0.15:
            p += 32
        elif rg > 0.08:
            p += 22
        elif rg > 0.03:
            p += 12
        elif rg < 0:
            warns.append(f"Chiffre d'affaires en recul ({rg * 100:.0f}%)")
        d.append(f"Croiss. CA {rg * 100:+.0f}%")
    eg = g("earningsGrowth") if np.isfinite(g("earningsGrowth")) else g("earningsQuarterlyGrowth")
    if np.isfinite(eg):
        if eg > 0.25:
            p += 40
        elif eg > 0.12:
            p += 32
        elif eg > 0.05:
            p += 20
        elif eg > 0:
            p += 10
        elif eg < 0:
            warns.append(f"Bénéfices en recul ({eg * 100:.0f}%)")
        d.append(f"Croiss. bénéfices {eg * 100:+.0f}%")
    rqg = g("revenueQuarterlyGrowth")
    if np.isfinite(rqg) and rqg > 0.10:
        p += 20
        d.append(f"Trimestre {rqg * 100:+.0f}%")
    return min(p, 100), "  ".join(d)


def score_health(f: dict, warns: list) -> tuple[float, str]:
    g = lambda k: _f(f.get(k))
    p, d = 0.0, []
    de = g("debtToEquity")
    if np.isfinite(de):
        if de < 30:
            p += 35
        elif de < 70:
            p += 26
        elif de < 130:
            p += 14
        elif de < 200:
            p += 6
        else:
            warns.append(f"Endettement élevé (dette/FP {de:.0f}%)")
        d.append(f"Dette/FP {de:.0f}%")
    cr = g("currentRatio")
    if np.isfinite(cr):
        if cr > 2:
            p += 25
        elif cr > 1.5:
            p += 18
        elif cr > 1:
            p += 10
        else:
            warns.append(f"Liquidité court terme faible (ratio {cr:.1f})")
        d.append(f"Ratio liq. {cr:.1f}")
    cash, debt = g("totalCash"), g("totalDebt")
    if np.isfinite(cash) and np.isfinite(debt):
        if cash >= debt:
            p += 25; d.append("Trésorerie > dette✓")
        elif debt > 0 and cash / debt > 0.4:
            p += 12
    qr = g("quickRatio")
    if np.isfinite(qr) and qr > 1:
        p += 15
    return min(p, 100), "  ".join(d)


def score_valuation(f: dict, warns: list) -> tuple[float, str]:
    """Pour la qualité+croissance, on ne réclame pas « bon marché » : on
    récompense le raisonnable et on pénalise l'excès manifeste."""
    g = lambda k: _f(f.get(k))
    p, d = 0.0, []
    pe = g("trailingPE")
    if np.isfinite(pe) and pe > 0:
        if pe < 15:
            p += 24
        elif pe < 25:
            p += 19
        elif pe < 35:
            p += 12
        elif pe < 50:
            p += 6
        else:
            warns.append(f"Valorisation tendue (PER {pe:.0f})")
        d.append(f"PER {pe:.0f}")
    peg = g("trailingPegRatio") if np.isfinite(g("trailingPegRatio")) else g("pegRatio")
    if np.isfinite(peg) and peg > 0:
        if peg < 1:
            p += 26
        elif peg < 1.5:
            p += 19
        elif peg < 2.5:
            p += 9
        d.append(f"PEG {peg:.2f}")
    pb = g("priceToBook")
    if np.isfinite(pb) and pb > 0:
        if pb < 3:
            p += 12
        elif pb < 8:
            p += 6
        d.append(f"P/B {pb:.1f}")
    ev = g("enterpriseToEbitda")
    if np.isfinite(ev) and ev > 0:
        if ev < 12:
            p += 12
        elif ev < 20:
            p += 7
        elif ev < 30:
            p += 3
        d.append(f"EV/EBITDA {ev:.0f}")
    fcf, mc = g("freeCashflow"), g("marketCap")
    if np.isfinite(fcf) and np.isfinite(mc) and mc > 0 and fcf > 0:
        fcf_yield = fcf / mc * 100
        if fcf_yield > 6:
            p += 26
        elif fcf_yield > 4:
            p += 18
        elif fcf_yield > 2:
            p += 9
        d.append(f"Rendt FCF {fcf_yield:.1f}%")
    return min(p, 100), "  ".join(d)


# ═══════════════════════════════════════════════════════════════════════════
#  DOCTRINES DES GRANDS INVESTISSEURS (orientées long terme)
# ═══════════════════════════════════════════════════════════════════════════

def investor_lenses(m: dict, f: dict) -> None:
    g = lambda k: _f(f.get(k))
    passed, notes = [], []
    roe, roa = g("returnOnEquity"), g("returnOnAssets")
    pm, gm = g("profitMargins"), g("grossMargins")
    rg = g("revenueGrowth")
    eg = g("earningsGrowth") if np.isfinite(g("earningsGrowth")) else g("earningsQuarterlyGrowth")
    de, cr = g("debtToEquity"), g("currentRatio")
    fcf = g("freeCashflow")
    pe, pb = g("trailingPE"), g("priceToBook")
    peg = g("trailingPegRatio") if np.isfinite(g("trailingPegRatio")) else g("pegRatio")
    ev = g("enterpriseToEbitda")

    if (np.isfinite(roe) and roe > 0.15 and np.isfinite(pm) and pm > 0.12
            and (not np.isfinite(de) or de < 130) and (not np.isfinite(fcf) or fcf > 0)):
        passed.append("Buffett")
        notes.append("Buffett : société de qualité, rentable, peu endettée (à garder)")
    if np.isfinite(peg) and 0 < peg < 1.5 and (not np.isfinite(eg) or eg > 0.08):
        passed.append("Lynch")
        notes.append(f"Lynch : croissance payée à bon prix (PEG {peg:.2f})")
    if np.isfinite(rg) and rg > 0.12 and np.isfinite(gm) and gm > 0.40:
        passed.append("Fisher")
        notes.append("Fisher : croissance durable + marges brutes élevées")
    earn_yield = (1 / ev) if (np.isfinite(ev) and ev > 0) else np.nan
    if (np.isfinite(earn_yield) and earn_yield > 0.07
            and ((np.isfinite(roe) and roe > 0.20) or (np.isfinite(roa) and roa > 0.10))):
        passed.append("Greenblatt")
        notes.append("Greenblatt (Magic Formula) : capital très rentable à bon rendement")
    if (np.isfinite(pe) and pe < 18 and np.isfinite(pb) and pb < 2.5
            and (not np.isfinite(cr) or cr > 1.3) and (not np.isfinite(de) or de < 120)):
        passed.append("Graham")
        notes.append("Graham : décote + bilan solide (marge de sécurité)")
    fscore = 0
    for cond in (np.isfinite(roa) and roa > 0, np.isfinite(fcf) and fcf > 0,
                 np.isfinite(pm) and pm > 0, np.isfinite(de) and de < 80,
                 np.isfinite(cr) and cr > 1):
        fscore += int(bool(cond))
    if fscore >= 4:
        passed.append("Piotroski")
        notes.append(f"Piotroski : robustesse financière {fscore}/5")
    m["lenses"], m["lens_notes"] = passed, notes


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSE COMPLÈTE D'UN TITRE
# ═══════════════════════════════════════════════════════════════════════════

def analyse(ticker: str, df: pd.DataFrame, f: dict) -> dict:
    warns = []
    tr = trend_metrics(df)
    m = {"symbol": ticker, "price": tr["price"], "high52": tr["high52"],
         "perf12m": tr["perf12m"], "warns": warns,
         "sector": f.get("sector") or "—", "name": f.get("shortName") or ticker}

    m["s_qual"], m["d_qual"] = score_quality(f, warns)
    m["s_grow"], m["d_grow"] = score_growth(f, warns)
    m["s_health"], m["d_health"] = score_health(f, warns)
    m["s_val"], m["d_val"] = score_valuation(f, warns)
    m["s_trend"], m["d_trend"] = score_trend_lt(tr)

    base = (0.30 * m["s_qual"] + 0.25 * m["s_grow"] + 0.15 * m["s_health"]
            + 0.20 * m["s_val"] + 0.10 * m["s_trend"])
    investor_lenses(m, f)
    m["lens_bonus"] = min(len(m["lenses"]) * 2.0, 10.0)
    m["score"] = min(100.0, base + m["lens_bonus"])

    # — Données pour la thèse & les signaux de vente —
    g = lambda k: _f(f.get(k))
    m["roe"], m["rg"], m["pe"] = g("returnOnEquity"), g("revenueGrowth"), g("trailingPE")
    m["div"] = g("dividendYield")
    tgt, cur = g("targetMeanPrice"), tr["price"]
    m["upside"] = (tgt / cur - 1) * 100 if (np.isfinite(tgt) and cur > 0) else np.nan
    return m


def thesis_and_sell_signals(m: dict) -> tuple[str, list[str]]:
    """Résume pourquoi détenir le titre + les conditions de revente."""
    bits = []
    if np.isfinite(m["roe"]) and m["roe"] > 0.15:
        bits.append(f"très rentable (ROE {m['roe'] * 100:.0f}%)")
    if np.isfinite(m["rg"]) and m["rg"] > 0.08:
        bits.append(f"en croissance ({m['rg'] * 100:+.0f}% de CA)")
    if m["lenses"]:
        bits.append("validée par " + ", ".join(m["lenses"]))
    thesis = "Société " + ", ".join(bits) + "." if bits else "Profil correct, données partielles."

    sells = []
    if np.isfinite(m["roe"]):
        sells.append(f"Si la rentabilité (ROE) tombe durablement sous 12 % (aujourd'hui {m['roe'] * 100:.0f}%).")
    sells.append("Si le chiffre d'affaires se met à reculer plusieurs trimestres de suite.")
    if np.isfinite(m["pe"]) and m["pe"] > 0:
        ceiling = max(45, round(m["pe"] * 1.6 / 5) * 5)
        sells.append(f"Si la valorisation s'emballe (PER au-dessus de ~{ceiling:.0f}, contre {m['pe']:.0f} aujourd'hui).")
    sells.append("Si le cours casse nettement et durablement sa moyenne 200 jours (la tendance de fond se retourne).")
    sells.append("Si tu trouves une société clairement meilleure pour la même somme.")
    return thesis, sells


# ═══════════════════════════════════════════════════════════════════════════
#  AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════

def fmt(p: float) -> str:
    return f"{p:,.2f}" if p >= 1 else f"{p:.4f}"


def rating(score: float):
    if score >= 75:
        return "À DÉTENIR (CONVICTION FORTE)", GREEN + BOLD
    if score >= 65:
        return "À DÉTENIR", GREEN
    if score >= 55:
        return "INTÉRESSANTE — À CREUSER", YELLOW
    if score >= 45:
        return "MOYENNE", ""
    return "À ÉVITER", RED


def bar10(score: float) -> str:
    n = max(0, min(10, round(score / 10)))
    return "█" * n + "·" * (10 - n)


def print_table(results: list[dict]):
    print(f"\n{BOLD}  {'#':>2}  {'TITRE':<8} {'PRIX':>10}  {'SCORE':>5}  {'QUAL':>4} {'CROIS':>5} "
          f"{'VALO':>4}  {'12M':>7}  {'NOTE':<28} LÉGENDES{RESET}")
    print(f"  {'─' * 112}")
    for i, m in enumerate(results, 1):
        note, ncol = rating(m["score"])
        p12 = f"{m['perf12m']:+.0f}%" if np.isfinite(m["perf12m"]) else "n/d"
        lens = ", ".join(m["lenses"]) if m["lenses"] else "—"
        print(f"  {i:>2}  {m['symbol']:<8} {fmt(m['price']):>10}"
              f"  {BOLD}{m['score']:>5.1f}{RESET}"
              f"  {m['s_qual']:>4.0f} {m['s_grow']:>5.0f} {m['s_val']:>4.0f}"
              f"  {p12:>7}  {ncol}{note:<28}{RESET} {GREEN}{lens}{RESET}")
    print(f"  {'─' * 112}")


def print_card(rank: int, m: dict, alloc: float | None, capital: float):
    note, ncol = rating(m["score"])
    print(f"\n{BOLD}{'═' * 78}{RESET}")
    print(f"{BOLD}  #{rank}  {CYAN}{m['symbol']}{RESET}{BOLD}  {m['name']}{RESET}  {DIM}{m['sector']}{RESET}")
    print(f"{'─' * 78}")
    p12 = f"{m['perf12m']:+.1f}%" if np.isfinite(m["perf12m"]) else "n/d"
    up = f"   Potentiel analystes {m['upside']:+.0f}%" if np.isfinite(m["upside"]) else ""
    print(f"  Prix {BOLD}{fmt(m['price'])} $US{RESET}   Sur 12 mois {p12}{up}")
    print(f"  Score {BOLD}{m['score']:.1f}/100{RESET} [{bar10(m['score'])}]  →  {ncol}{BOLD}{note}{RESET}")
    print()
    print(f"  ├─ Qualité      {m['s_qual']:>3.0f} [{bar10(m['s_qual'])}]  {DIM}{m['d_qual']}{RESET}")
    print(f"  ├─ Croissance   {m['s_grow']:>3.0f} [{bar10(m['s_grow'])}]  {DIM}{m['d_grow']}{RESET}")
    print(f"  ├─ Santé fin.   {m['s_health']:>3.0f} [{bar10(m['s_health'])}]  {DIM}{m['d_health']}{RESET}")
    print(f"  ├─ Valorisation {m['s_val']:>3.0f} [{bar10(m['s_val'])}]  {DIM}{m['d_val']}{RESET}")
    print(f"  └─ Tendance LT  {m['s_trend']:>3.0f} [{bar10(m['s_trend'])}]  {DIM}{m['d_trend']}{RESET}")

    if m["lenses"]:
        badges = " ".join(f"{GREEN}◆ {x}{RESET}" for x in m["lenses"])
        print(f"\n  {BOLD}DOCTRINES VALIDÉES ({len(m['lenses'])}, +{m['lens_bonus']:.0f} pts){RESET}  {badges}")
        for n in m["lens_notes"]:
            print(f"    {DIM}• {n}{RESET}")

    thesis, sells = thesis_and_sell_signals(m)
    print(f"\n  {BOLD}POURQUOI LA DÉTENIR{RESET}")
    print(f"    {thesis}")
    if alloc is not None:
        print(f"    Allocation suggérée : {alloc:.0f}$ ({alloc / capital * 100:.0f}% du portefeuille)")
    print(f"\n  {BOLD}🔔 REVOIR LA THÈSE (signaux de vente){RESET}")
    for s in sells:
        print(f"    • {s}")
    review = (datetime.now() + timedelta(days=REVIEW_DAYS)).strftime("%d/%m/%Y")
    print(f"  {BOLD}📅 Prochaine revue : {review}{RESET} (trimestrielle — pas de stop, on suit la qualité)")
    if m["warns"]:
        for w in m["warns"]:
            print(f"  {YELLOW}⚠ {w}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE
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
        description="Sélecteur d'actions qualité + croissance pour le long terme",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemples :
  python3 stock_investor.py                       S&P 500, top 20
  python3 stock_investor.py --universe nasdaq100  univers Nasdaq 100
  python3 stock_investor.py AAPL MSFT GOOGL       analyse ciblée
  python3 stock_investor.py --portfolio 12 --capital 10000
  python3 stock_investor.py --json out.json""")
    ap.add_argument("symbols", nargs="*", help="tickers ciblés (sinon scan de l'univers)")
    ap.add_argument("--universe", default="sp500", help="sp500 (défaut), nasdaq100, dow30")
    ap.add_argument("--top", type=int, default=20, help="taille du classement (défaut 20)")
    ap.add_argument("--detail", type=int, default=6, help="fiches détaillées (défaut 6)")
    ap.add_argument("--portfolio", type=int, default=0, help="construire un portefeuille de N lignes (allocation suggérée)")
    ap.add_argument("--capital", type=float, default=10000.0, help="capital total du portefeuille (défaut 10000$)")
    ap.add_argument("--workers", type=int, default=8, help="requêtes fondamentaux parallèles (défaut 8)")
    ap.add_argument("--json", metavar="FICHIER", help="exporter les résultats en JSON")
    args = ap.parse_args()

    t0 = time.time()
    print(f"\n{BOLD}{MAGENTA}  ◆ STOCK INVESTOR — qualité + croissance, horizon LONG TERME ◆{RESET}")
    print(f"  {DIM}{datetime.now().strftime('%d/%m/%Y %H:%M')} — données Yahoo Finance{RESET}\n")

    targeted = bool(args.symbols)
    if targeted:
        tickers, uni_name = [s.upper().replace(".", "-") for s in args.symbols], "ciblé"
    else:
        tickers, uni_name = get_universe(args.universe)
        print(f"  Univers : {uni_name} ({len(tickers)} titres)")

    print("  Téléchargement des cours…", end=" ", flush=True)
    hist = download_history(tickers)
    print(f"{len(hist)} titres")
    if not hist:
        print(f"{RED}  Aucune donnée disponible.{RESET}")
        sys.exit(1)

    ordered = [t for t in tickers if t in hist]
    print(f"  Récupération des fondamentaux ({len(ordered)})…", end=" ", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        funds = list(ex.map(fetch_fundamentals, ordered))
    print("OK")

    results = []
    for t, f in zip(ordered, funds):
        try:
            results.append(analyse(t, hist[t], f))
        except Exception:
            continue

    results.sort(key=lambda m: m["score"], reverse=True)
    top_n = results if targeted else results[:args.top]

    print_table(top_n)

    # Construction d'un portefeuille équipondéré (conviction) si demandé
    alloc_map = {}
    if args.portfolio > 0 and not targeted:
        picks = results[:args.portfolio]
        per_line = args.capital / len(picks) if picks else 0
        for m in picks:
            alloc_map[m["symbol"]] = per_line
        print(f"\n{BOLD}{CYAN}  PORTEFEUILLE SUGGÉRÉ — {len(picks)} lignes, "
              f"{per_line:.0f}$ chacune (capital {args.capital:.0f}$){RESET}")
        sectors = {}
        for m in picks:
            sectors[m["sector"]] = sectors.get(m["sector"], 0) + 1
        rep = ", ".join(f"{k} ×{v}" for k, v in sorted(sectors.items(), key=lambda x: -x[1]))
        print(f"  {DIM}Répartition sectorielle : {rep}{RESET}")
        warn_concentr = [k for k, v in sectors.items() if v > max(2, len(picks) // 3)]
        if warn_concentr:
            print(f"  {YELLOW}⚠ Concentration sur : {', '.join(warn_concentr)} — pense à diversifier.{RESET}")

    n_cards = len(top_n) if targeted else min(args.detail, len(top_n))
    for i in range(n_cards):
        m = top_n[i]
        print_card(i + 1, m, alloc_map.get(m["symbol"]), args.capital)

    dur = time.time() - t0
    print(f"\n{BOLD}{'═' * 78}{RESET}")
    print(f"  {DIM}Analysés : {len(results)}  |  Univers : {uni_name}  |  Durée : {dur:.0f}s{RESET}")
    print(f"  {DIM}⚠ Outil éducatif. Pas un conseil en investissement. Fais tes propres "
          f"recherches avant d'engager de l'argent.{RESET}")
    print(f"{BOLD}{'═' * 78}{RESET}\n")

    if args.json:
        export = {"generated_at": datetime.now().isoformat(), "universe": uni_name,
                  "results": top_n}
        with open(args.json, "w") as fp:
            json.dump(export, fp, indent=2, ensure_ascii=False, default=to_py)
        print(f"  Résultats exportés → {args.json}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}  Interrompu.{RESET}")
        sys.exit(130)
