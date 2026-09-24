# Experiments

Every `python run.py` invocation writes a directory here named `<experiment>_<YYYYMMDD_HHMMSS>/` containing:

- `config.yaml` — the merged configuration (base + profile) used for the run
- `results.json` — overall and per-window metrics for MAS, Buy & Hold and SMA 20/50
- `equity_curve.csv` — daily out-of-sample equity (used by the dashboard overlay)

The `experiment` prefix encodes the active agents (`fase3_matematico`, `fase5_mat_cazador`,
`fase6_mat_cazador_conspiranoico`, …) and the profile name, if any.

## Canonical experiment selection

The run used for the dashboard, the frozen models and paper trading is chosen by a deterministic rule:

1. **Primary:** highest Sharpe ratio over the walk-forward validation period (2021–2024).
2. **Exclusion:** runs with max drawdown worse than -30 % are discarded.
3. **Tiebreak:** higher Calmar ratio.

## Selected experiment

| Field | Value |
|---|---|
| Experiment ID | `fase6_mat_cazador_conspiranoico_exp1_menos_friccion_20260622_035259` |
| Profile | `profiles/exp1_menos_friccion.yaml` |
| Agents | Matemático + Cazador + Conspiranoico (Analista off) |
| Timestamp | 2026-06-22 03:52:59 |

### Walk-forward metrics (out-of-sample 2021–2024, 1 005 trading days)

| Metric | MAS | Buy & Hold | SMA 20/50 |
|---|---:|---:|---:|
| Total return | **86.64 %** | 67.90 % | 14.47 % |
| Annualised return | **17.78 %** | 15.25 % | 4.21 % |
| Sharpe | **0.861** | 0.723 | 0.329 |
| Max drawdown | -29.63 % | -31.36 % | -22.25 % |
| Calmar | **0.60** | 0.49 | 0.19 |
| Win rate | 52.34 % | 52.44 % | 54.08 % |

### Window breakdown

| Iter | Validation period | Return | Sharpe | Max DD | Win rate |
|---|---|---:|---:|---:|---:|
| 1 | 2021 | +23.97 % | 1.426 | -10.23 % | 54.76 % |
| 2 | 2022 | -21.19 % | -0.765 | -29.28 % | 43.43 % |
| 3 | 2023 | +44.55 % | 2.334 | -10.19 % | 53.60 % |
| 4 | 2024 | +34.95 % | 1.505 | -14.40 % | 57.14 % |

### Holdout 2025 (evaluated once)

`reports/holdout_2025_exp1_menos_friccion.json` — MAS +26.78 % / Sharpe 1.148 / MaxDD -20.77 % vs
Buy & Hold +23.93 % / 1.213 / -18.26 %.

## Reproducing

```bash
python run.py --cazador --conspiranoico --profile profiles/exp1_menos_friccion.yaml
```

Seeds are fixed (`general.random_seed: 42`) and inputs are cached in `data/cache/`, but two runs are only
bit-identical when the cache is identical: refreshing OHLCV (yfinance adjusts historical prices) or the SEC
Form 4 cache moves the numbers slightly. Runs of this profile in June and September 2026 landed at Sharpe
0.86–0.91 with the same window pattern (2022 negative, other years strongly positive); baselines are
unaffected. Treat the canonical numbers as the reference snapshot rather than an exact target.

## Grid search

`scripts/run_experiment_grid_carry.py` runs the whole Exp1–Exp10 profile grid sequentially and summarises it
in `reports/experiment_grid_carry.json`. The high-level outcome of each experiment is in the root README.
