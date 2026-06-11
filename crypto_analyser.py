#!/usr/bin/env python3
"""
Analyseur crypto — Signaux d'investissement court terme (day/swing)
Indicateurs : RSI, MACD, Bollinger Bands, EMA 20/50/200, Stochastique, Volume
Source de données : API publique Binance (aucune clé requise)

Usage :
    python crypto_analyser.py BTC
    python crypto_analyser.py ETH --interval 4h
    python crypto_analyser.py SOL BTC ETH        # multi-actifs
"""

import argparse
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    WHITE  = Fore.WHITE
    BOLD   = Style.BRIGHT
    DIM    = Style.DIM
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = WHITE = BOLD = DIM = RESET = ""


# ─── Récupération des données ──────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"

    url = "https://api.binance.com/api/v3/klines"
    resp = requests.get(url, params={"symbol": sym, "interval": interval, "limit": limit}, timeout=10)

    if resp.status_code == 400:
        raise ValueError(f"Symbole inconnu sur Binance : {sym}")
    resp.raise_for_status()

    df = pd.DataFrame(resp.json(), columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df[["time", "open", "high", "low", "close", "volume"]]


# ─── Indicateurs techniques ────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast  = series.ewm(span=fast,   adjust=False).mean()
    ema_slow  = series.ewm(span=slow,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line, macd_line - sig_line


def bollinger(series: pd.Series, period=20, nb_std=2):
    sma   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return sma + nb_std * sigma, sma, sma - nb_std * sigma


def stochastic(df: pd.DataFrame, k=14, d=3):
    low_min  = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    pct_k    = 100 * (df["close"] - low_min) / (high_max - low_min)
    return pct_k, pct_k.rolling(d).mean()


def atr(df: pd.DataFrame, period=14) -> pd.Series:
    prev  = df["close"].shift()
    tr    = pd.concat([df["high"] - df["low"],
                       (df["high"] - prev).abs(),
                       (df["low"]  - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─── Moteur d'analyse ──────────────────────────────────────────────────────────

def analyse(symbol: str, interval: str = "1h") -> dict:
    df    = fetch_ohlcv(symbol, interval)
    close = df["close"]
    price = close.iloc[-1]
    change_pct = (price - close.iloc[-2]) / close.iloc[-2] * 100

    # Calcul des indicateurs
    rsi_s          = rsi(close)
    macd_l, sig_l, hist = macd(close)
    bb_up, bb_mid, bb_lo = bollinger(close)
    ema20  = close.ewm(span=20,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    stoch_k, stoch_d = stochastic(df)
    atr_s  = atr(df)

    # Valeurs courantes
    rsi_v    = rsi_s.iloc[-1]
    macd_v   = macd_l.iloc[-1]
    sig_v    = sig_l.iloc[-1]
    hist_v   = hist.iloc[-1]
    hist_p   = hist.iloc[-2]
    bb_up_v  = bb_up.iloc[-1]
    bb_lo_v  = bb_lo.iloc[-1]
    bb_mid_v = bb_mid.iloc[-1]
    ema20_v  = ema20.iloc[-1]
    ema50_v  = ema50.iloc[-1]
    ema200_v = ema200.iloc[-1]
    sk_v     = stoch_k.iloc[-1]
    sd_v     = stoch_d.iloc[-1]
    atr_v    = atr_s.iloc[-1]
    vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]

    signals = []
    score   = 0

    # ── RSI (max ±2) ──────────────────────────────────────────────────────────
    if rsi_v < 30:
        s = +2; msg = "Survente extrême — signal d'achat fort"
    elif rsi_v < 45:
        s = +1; msg = "Zone basse — légèrement haussier"
    elif rsi_v > 70:
        s = -2; msg = "Surachat extrême — signal de vente fort"
    elif rsi_v > 55:
        s = -1; msg = "Zone haute — légèrement baissier"
    else:
        s =  0; msg = "Zone neutre (45–55)"
    score += s
    signals.append(("RSI(14)", f"{rsi_v:.1f}", s, msg))

    # ── MACD (max ±2) ─────────────────────────────────────────────────────────
    if macd_v > sig_v:
        s = +2 if hist_v > hist_p else +1
        msg = "Croisement haussier — momentum accéléré" if s == 2 else "Au-dessus signal, momentum ralentit"
    elif macd_v < sig_v:
        s = -2 if hist_v < hist_p else -1
        msg = "Croisement baissier — momentum baissier accéléré" if s == -2 else "Sous signal, momentum baissier"
    else:
        s = 0; msg = "Croisement en cours — neutre"
    score += s
    signals.append(("MACD(12/26)", f"{macd_v:.5f}", s, msg))

    # ── Bollinger Bands (max ±2) ──────────────────────────────────────────────
    bb_pct = (price - bb_lo_v) / (bb_up_v - bb_lo_v) * 100 if bb_up_v != bb_lo_v else 50
    if price <= bb_lo_v:
        s = +2; msg = "Prix sous bande basse — survente"
    elif bb_pct < 25:
        s = +1; msg = "Proche bande basse — zone d'achat"
    elif price >= bb_up_v:
        s = -2; msg = "Prix sur bande haute — surachat"
    elif bb_pct > 75:
        s = -1; msg = "Proche bande haute — zone de vente"
    else:
        s =  0; msg = f"Dans les bandes ({bb_pct:.0f}%)"
    score += s
    signals.append(("Bollinger(20)", f"{bb_pct:.0f}%  bandes", s, msg))

    # ── EMA 20/50 (max ±2) ───────────────────────────────────────────────────
    if ema20_v > ema50_v and price > ema20_v:
        s = +2; msg = "Tendance haussière — prix au-dessus EMA20"
    elif ema20_v > ema50_v:
        s = +1; msg = "Tendance haussière mais prix sous EMA20"
    elif ema20_v < ema50_v and price < ema20_v:
        s = -2; msg = "Tendance baissière — prix sous EMA20"
    elif ema20_v < ema50_v:
        s = -1; msg = "Tendance baissière mais prix sur EMA20"
    else:
        s =  0; msg = "EMAs proches — tendance indécise"
    score += s
    signals.append(("EMA 20/50", f"{ema20_v:.4f} / {ema50_v:.4f}", s, msg))

    # ── Stochastique (max ±1) ─────────────────────────────────────────────────
    if sk_v < 20 and sd_v < 20:
        s = +1; msg = "Zone de survente"
    elif sk_v > 80 and sd_v > 80:
        s = -1; msg = "Zone de surachat"
    else:
        s =  0; msg = "Zone neutre"
    score += s
    signals.append(("Stoch.(14)", f"K={sk_v:.1f}  D={sd_v:.1f}", s, msg))

    # ── Volume (max ±1) ───────────────────────────────────────────────────────
    if vol_ratio > 1.5 and change_pct > 0:
        s = +1; msg = f"Volume x{vol_ratio:.1f} moy. avec hausse — confirmation"
    elif vol_ratio > 1.5 and change_pct < 0:
        s = -1; msg = f"Volume x{vol_ratio:.1f} moy. avec baisse — pression vendeuse"
    elif vol_ratio < 0.5:
        s =  0; msg = f"Volume x{vol_ratio:.1f} moy. — conviction faible"
    else:
        s =  0; msg = f"Volume x{vol_ratio:.1f} moy. — normal"
    score += s
    signals.append(("Volume", f"x{vol_ratio:.2f} moy.", s, msg))

    # ── Recommandation ────────────────────────────────────────────────────────
    if score >= 6:
        rec = "ACHETER MAINTENANT"; cat = "buy_strong"
    elif score >= 3:
        rec = "SIGNAL D'ACHAT";     cat = "buy"
    elif score >= 1:
        rec = "SURVEILLER (biais haussier)"; cat = "watch_up"
    elif score <= -6:
        rec = "ÉVITER / VENDRE";    cat = "sell_strong"
    elif score <= -3:
        rec = "SIGNAL DE VENTE";    cat = "sell"
    elif score <= -1:
        rec = "PRUDENCE (biais baissier)"; cat = "watch_down"
    else:
        rec = "ATTENDRE (signal neutre)"; cat = "neutral"

    # ── Niveaux de gestion du risque ─────────────────────────────────────────
    stop_loss = price - 1.5 * atr_v
    target_1  = price + 2.0 * atr_v
    target_2  = price + 4.0 * atr_v
    rr_ratio  = (target_1 - price) / (price - stop_loss) if price > stop_loss else 0

    return {
        "symbol":     symbol.upper() + ("USDT" if not symbol.upper().endswith("USDT") else ""),
        "interval":   interval,
        "price":      price,
        "change_pct": change_pct,
        "score":      score,
        "rec":        rec,
        "cat":        cat,
        "signals":    signals,
        "ema200":     ema200_v,
        "atr":        atr_v,
        "stop_loss":  stop_loss,
        "target_1":   target_1,
        "target_2":   target_2,
        "rr_ratio":   rr_ratio,
    }


# ─── Affichage ─────────────────────────────────────────────────────────────────

def rec_color(cat: str) -> str:
    if "buy" in cat:   return GREEN
    if "sell" in cat:  return RED
    return YELLOW


def score_bar(score: int) -> str:
    total = 20
    # score ∈ [-10, +10] → position ∈ [0, 20]
    filled = max(0, min(total, int((score + 10) / 20 * total)))
    return "█" * filled + "░" * (total - filled)


def fmt_price(p: float) -> str:
    if p >= 1000:  return f"{p:,.2f}"
    if p >= 1:     return f"{p:,.4f}"
    return f"{p:.8f}"


def print_report(r: dict):
    sym    = r["symbol"]
    price  = r["price"]
    chg    = r["change_pct"]
    score  = r["score"]
    cat    = r["cat"]
    rc     = rec_color(cat)
    chg_c  = GREEN if chg >= 0 else RED
    sign   = "+" if chg >= 0 else ""
    now    = datetime.now().strftime("%d/%m/%Y %H:%M")

    W = 62
    print(f"\n{'═'*W}")
    print(f"{BOLD}{CYAN}  ▶ ANALYSE CRYPTO — {sym}  [{r['interval'].upper()}]  {now}{RESET}")
    print(f"{'═'*W}")

    print(f"  Prix      : {BOLD}{fmt_price(price)} USDT{RESET}   "
          f"{chg_c}{sign}{chg:.2f}%{RESET}")

    trend = "HAUSSIÈRE ▲" if price > r["ema200"] else "BAISSIÈRE ▼"
    tc    = GREEN if price > r["ema200"] else RED
    print(f"  Tendance  : {tc}{trend}{RESET}  (EMA200 = {fmt_price(r['ema200'])})")

    print(f"\n  Score     : {BOLD}{score:+d} / 10{RESET}")
    print(f"  [{score_bar(score)}]  −10 ←→ +10")
    print(f"\n  ▶ {BOLD}{rc}{r['rec']}{RESET}")

    # Tableau des signaux
    print(f"\n{'─'*W}")
    print(f"  {'INDICATEUR':<16}  {'VALEUR':<22}  SIG  ANALYSE")
    print(f"{'─'*W}")
    for name, val, sig, msg in r["signals"]:
        if sig > 0:
            sig_disp = f"{GREEN}▲ +{sig}{RESET}"
        elif sig < 0:
            sig_disp = f"{RED}▼ {sig}{RESET}"
        else:
            sig_disp = f"{YELLOW}● {sig}{RESET}"
        # colorama strip trick for alignment: pad before adding colors
        val_str = val[:22].ljust(22)
        print(f"  {name:<16}  {val_str}  {sig_disp}   {msg}")

    # Niveaux de risque
    sl_pct = (price - r["stop_loss"]) / price * 100
    t1_pct = (r["target_1"] - price)  / price * 100
    t2_pct = (r["target_2"] - price)  / price * 100

    print(f"\n{'─'*W}")
    print(f"  GESTION DU RISQUE  (ATR={fmt_price(r['atr'])})")
    print(f"  Stop-loss   : {RED}{fmt_price(r['stop_loss'])} USDT{RESET}  "
          f"(-{sl_pct:.2f}%)")
    print(f"  Objectif 1  : {GREEN}{fmt_price(r['target_1'])} USDT{RESET}  "
          f"(+{t1_pct:.2f}%)")
    print(f"  Objectif 2  : {GREEN}{fmt_price(r['target_2'])} USDT{RESET}  "
          f"(+{t2_pct:.2f}%)")
    print(f"  Ratio R/R   : {r['rr_ratio']:.1f}:1  "
          + (f"{GREEN}(favorable){RESET}" if r["rr_ratio"] >= 2 else f"{YELLOW}(à vérifier){RESET}"))

    print(f"\n{'═'*W}")
    print(f"  {DIM}⚠  Outil éducatif uniquement — pas un conseil financier.{RESET}")
    print(f"{'═'*W}\n")


def print_summary(results: list[dict]):
    """Tableau comparatif pour analyse multi-actifs."""
    if len(results) <= 1:
        return
    print(f"\n{'═'*70}")
    print(f"{BOLD}{CYAN}  COMPARATIF RAPIDE{RESET}")
    print(f"{'═'*70}")
    print(f"  {'ACTIF':<12}  {'PRIX':>12}  {'VAR':>8}  {'SCORE':>6}  RECOMMANDATION")
    print(f"{'─'*70}")
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        rc   = rec_color(r["cat"])
        chgc = GREEN if r["change_pct"] >= 0 else RED
        sign = "+" if r["change_pct"] >= 0 else ""
        print(f"  {r['symbol']:<12}  {fmt_price(r['price']):>12}  "
              f"{chgc}{sign}{r['change_pct']:.2f}%{RESET}  "
              f"{BOLD}{r['score']:>+5}{RESET}  "
              f"{rc}{r['rec']}{RESET}")
    print(f"{'═'*70}\n")


# ─── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyseur crypto — signaux d'investissement court terme",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python crypto_analyser.py BTC
  python crypto_analyser.py ETH --interval 4h
  python crypto_analyser.py BTC ETH SOL BNB
        """
    )
    parser.add_argument("symbols", nargs="+",
                        help="Symbole(s) crypto (ex: BTC ETH SOL)")
    parser.add_argument("--interval", "-i",
                        default="1h",
                        choices=["15m", "1h", "4h", "1d"],
                        help="Intervalle de bougie (défaut : 1h)")
    args = parser.parse_args()

    results = []
    for sym in args.symbols:
        print(f"Analyse de {sym.upper()}...", end=" ", flush=True)
        try:
            r = analyse(sym, args.interval)
            print("OK")
            results.append(r)
        except ValueError as e:
            print(f"ERREUR — {e}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"ERREUR réseau — {e}", file=sys.stderr)

    for r in results:
        print_report(r)

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
