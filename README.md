# Crypto Screener — opportunités swing (≤ 2 semaines)

Screener du marché spot Binance complet (paires USDT) avec score composite
multi-piliers fondé sur des concepts éprouvés, et filtre de régime de marché.

## Outils

- **`crypto_screener.py`** — scanne tout le marché, classe les actifs sur 100 points
  (tendance, momentum, timing, volume, qualité), applique un facteur de régime
  (tendance BTC + breadth + Fear & Greed), valide en confluence 4h et produit
  un plan de trade complet (stop ATR, objectifs 2R/4R, taille à risque fixe,
  sortie temporelle 14 jours).
- **`crypto_analyser.py`** — analyse mono-actif horaire (RSI, MACD, Bollinger,
  EMA, stochastique, volume).
- **`stock_investor.py`** — sélecteur d'actions **qualité + croissance pour le
  long terme** (horizon 6 mois à plusieurs années, philosophie Buffett / Lynch /
  Fisher), univers S&P 500 / Nasdaq 100 / Dow 30 via Yahoo Finance. Score sur 6
  piliers (qualité, croissance, santé financière, valorisation, tendance long
  terme + doctrines des grands investisseurs). Pas de stop serré : il fournit la
  thèse « pourquoi détenir », les signaux de vente, une allocation de portefeuille
  suggérée et une date de revue trimestrielle.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 crypto_screener.py                      # scan complet, top 15
python3 crypto_screener.py --min-vol 2          # élargit l'univers scanné
python3 crypto_screener.py --top 25 --detail 8  # plus de fiches détaillées
python3 crypto_screener.py SOL INJ FET          # analyse ciblée
python3 crypto_screener.py --json out.json      # export JSON
```

```bash
python3 stock_investor.py                         # S&P 500, top 20 (long terme)
python3 stock_investor.py --universe nasdaq100    # autre univers
python3 stock_investor.py AAPL MSFT GOOGL         # analyse ciblée
python3 stock_investor.py --portfolio 12 --capital 10000  # portefeuille suggéré
python3 stock_investor.py --json out.json         # export JSON
```

Aucune clé API requise (endpoints publics Binance ; Yahoo Finance pour les actions).

## ⚠ Avertissement

Outil strictement éducatif. Ne constitue pas un conseil en investissement.
Les performances passées ne préjugent pas des performances futures.
