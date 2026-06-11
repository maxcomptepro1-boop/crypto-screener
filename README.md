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

Aucune clé API requise (endpoints publics Binance).

## ⚠ Avertissement

Outil strictement éducatif. Ne constitue pas un conseil en investissement.
Les performances passées ne préjugent pas des performances futures.
