#!/usr/bin/env python3
"""
STOCK BACKTEST — Mesure HONNÊTE de la performance du moteur tendance/momentum
qui sous-tend stock_investor.py, rejouée sur l'historique réel des cours.

⚠ HONNÊTETÉ MÉTHODOLOGIQUE (à lire) :
  • On backteste UNIQUEMENT la partie « prix » de la stratégie (tendance long
    terme + momentum 12-1), car c'est la seule qu'on peut rejouer sans tricher.
    Les fondamentaux (qualité, valorisation) ne sont PAS disponibles en version
    historique gratuitement → les inclure créerait un biais de « regard vers le
    futur » (look-ahead) qui gonflerait artificiellement les résultats.
  • Biais de survivance : on utilise les sociétés cotées AUJOURD'HUI. Les
    faillites/retraits passés ne sont pas dans l'univers → résultats un peu
    optimistes. On le signale, on ne le cache pas.
  • Conclusion : ces chiffres mesurent la VALEUR DU MOTEUR TECHNIQUE, pas une
    promesse de gain futur. Les performances passées ne préjugent de rien.

Stratégie testée :
  • Rééquilibrage mensuel. À chaque fin de mois, on garde les N actions :
      – dont le cours est au-dessus de sa moyenne 200 jours (tendance de fond),
      – les mieux classées en momentum 12-1 (perf sur 12 mois hors dernier mois,
        l'anomalie la mieux documentée académiquement),
    équipondérées, conservées un mois.
  • Comparaison à l'achat-conservation du S&P 500 (SPY).

Usage :
    python3 stock_backtest.py                       # Nasdaq 100, 6 ans, top 15
    python3 stock_backtest.py --universe sp500 --years 8 --top 20
    python3 stock_backtest.py --hold 15             # 15 lignes en portefeuille

Source : Yahoo Finance via yfinance.
"""

import argparse
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

try:
    import yfinance as yf
except ImportError:
    print("Erreur : yfinance manquant.  pip install yfinance")
    sys.exit(1)

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN, RED, YELLOW, CYAN, MAGENTA = Fore.GREEN, Fore.RED, Fore.YELLOW, Fore.CYAN, Fore.MAGENTA
    BOLD, DIM, RESET = Style.BRIGHT, Style.DIM, Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = MAGENTA = BOLD = DIM = RESET = ""

# On réutilise les univers de stock_investor s'il est présent ; sinon liste interne.
try:
    import stock_investor as si
    HAVE_SI = True
except Exception:
    HAVE_SI = False

FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "COST",
    "NFLX", "AMD", "PEP", "ADBE", "CSCO", "INTU", "QCOM", "TXN", "AMGN", "HON",
    "BKNG", "VRTX", "ADP", "MU", "ADI", "REGN", "LRCX", "GILD", "KLAC", "SNPS",
    "CDNS", "MAR", "ORLY", "CTAS", "MNST", "PCAR", "PAYX", "ROST", "ODFL", "FAST",
]


def get_universe(name: str) -> tuple[list[str], str]:
    if HAVE_SI:
        try:
            return si.get_universe(name)
        except Exception:
            pass
    return FALLBACK, "Échantillon (40 valeurs)"


def download_closes(tickers: list[str], years: int, chunk=120) -> pd.DataFrame:
    period = f"{years + 1}y"
    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        try:
            data = yf.download(batch, period=period, interval="1d",
                               group_by="ticker", auto_adjust=True,
                               threads=True, progress=False)
        except Exception:
            continue
        if data is None or not len(data):
            continue
        if isinstance(data.columns, pd.MultiIndex):
            for t in batch:
                if t in data.columns.get_level_values(0):
                    s = data[t]["Close"].rename(t)
                    frames.append(s)
        else:
            frames.append(data["Close"].rename(batch[0]))
        print(f"\r  Cours : {min(i + chunk, len(tickers))}/{len(tickers)}…", end="", flush=True)
    print()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((equity - peak) / peak).min() * 100)


def stats(monthly_ret: pd.Series, label: str) -> dict:
    r = monthly_ret.dropna()
    if len(r) < 6:
        return {}
    eq = (1 + r).cumprod()
    n = len(r)
    cagr = eq.iloc[-1] ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = cagr / vol if vol > 0 else 0
    return {"label": label, "cagr": cagr * 100, "vol": vol * 100,
            "sharpe": sharpe, "maxdd": max_drawdown(eq),
            "total": (eq.iloc[-1] - 1) * 100, "months": n, "equity": eq}


def main():
    ap = argparse.ArgumentParser(
        description="Backtest honnête du moteur tendance/momentum de stock_investor",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default="nasdaq100", help="nasdaq100 (défaut), sp500, dow30, sp1500")
    ap.add_argument("--years", type=int, default=6, help="profondeur du backtest (défaut 6 ans)")
    ap.add_argument("--top", "--hold", dest="top", type=int, default=15, help="nb d'actions tenues (défaut 15)")
    args = ap.parse_args()

    print(f"\n{BOLD}{MAGENTA}  ◆ BACKTEST — moteur tendance/momentum long terme ◆{RESET}")
    print(f"  {DIM}{datetime.now().strftime('%d/%m/%Y %H:%M')} — données Yahoo Finance{RESET}\n")
    print(f"  {YELLOW}⚠ Mesure la partie PRIX de la stratégie uniquement (pas les fondamentaux).{RESET}")
    print(f"  {YELLOW}  Biais de survivance présent. Les performances passées ne préjugent de rien.{RESET}\n")

    tickers, uni_name = get_universe(args.universe)
    print(f"  Univers : {uni_name} ({len(tickers)} titres)  |  Profondeur : {args.years} ans  |  Portefeuille : {args.top} lignes")

    closes = download_closes(tickers, args.years)
    spy = download_closes(["SPY"], args.years)
    if closes.empty or spy.empty:
        print(f"{RED}  Données indisponibles.{RESET}")
        sys.exit(1)
    spy = spy.iloc[:, 0]

    # Prix mensuels (fin de mois) + moyenne 200 j ramenée en mensuel
    m_close = closes.resample("ME").last()
    sma200 = closes.rolling(200).mean().resample("ME").last()
    above = m_close > sma200                       # tendance de fond OK ?
    mom = m_close.shift(1) / m_close.shift(12) - 1  # momentum 12-1 (hors dernier mois)
    fwd = m_close.shift(-1) / m_close - 1           # rendement du mois suivant

    spy_m = spy.resample("ME").last()
    spy_ret = (spy_m.shift(-1) / spy_m - 1)

    dates = m_close.index
    start = 12  # besoin de 12 mois d'historique pour le momentum
    port_rets = []
    idx = []
    n_pos = []
    for k in range(start, len(dates) - 1):
        d = dates[k]
        elig = above.loc[d] & mom.loc[d].notna() & fwd.loc[d].notna()
        cand = mom.loc[d][elig]
        if len(cand) < 3:
            continue
        picks = cand.sort_values(ascending=False).head(args.top).index
        ret = fwd.loc[d, picks].mean()
        port_rets.append(ret)
        idx.append(dates[k + 1])
        n_pos.append(len(picks))

    if len(port_rets) < 6:
        print(f"{RED}  Historique insuffisant pour conclure.{RESET}")
        sys.exit(1)

    port = pd.Series(port_rets, index=idx)
    bench = spy_ret.reindex(port.index)

    s_p = stats(port, "Stratégie (tendance+momentum)")
    s_b = stats(bench, "Achat-conservation S&P 500")
    win = float((port.values > bench.values).mean() * 100)

    print(f"\n{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}  RÉSULTATS — {s_p['months']} mois (~{s_p['months'] / 12:.1f} ans), "
          f"~{np.mean(n_pos):.0f} actions tenues{RESET}")
    print(f"{CYAN}{'─' * 70}{RESET}")
    hdr = f"  {'':<32}{'Stratégie':>16}{'S&P 500':>16}"
    print(BOLD + hdr + RESET)
    def row(lbl, a, b, suf="%", good_high=True):
        col = GREEN if ((a >= b) == good_high) else RED
        print(f"  {lbl:<32}{col}{a:>14.1f}{suf}{RESET}{b:>14.1f}{suf}")
    row("Performance totale", s_p["total"], s_b["total"])
    row("Rendement annualisé (CAGR)", s_p["cagr"], s_b["cagr"])
    row("Volatilité (risque)", s_p["vol"], s_b["vol"], good_high=False)
    print(f"  {'Ratio rendement/risque':<32}{s_p['sharpe']:>15.2f} {s_b['sharpe']:>15.2f}")
    row("Pire perte (max drawdown)", s_p["maxdd"], s_b["maxdd"], good_high=True)
    print(f"  {'Mois où on bat le S&P 500':<32}{win:>14.0f}%{'—':>16}")
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")

    diff = s_p["cagr"] - s_b["cagr"]
    verdict = (f"{GREEN}Le moteur SURPERFORME le S&P 500 de {diff:+.1f} pts/an{RESET}" if diff > 0.5
               else f"{RED}Le moteur SOUS-performe le S&P 500 de {diff:.1f} pts/an{RESET}" if diff < -0.5
               else f"{YELLOW}Le moteur fait ~jeu égal avec le S&P 500{RESET}")
    print(f"\n  Verdict : {verdict}")
    print(f"  {DIM}Rappel : ne mesure que la partie prix, biais de survivance présent. "
          f"Éducatif, pas un conseil.{RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}  Interrompu.{RESET}")
        sys.exit(130)
