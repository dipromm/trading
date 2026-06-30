# Canonical Experiment Selection

## Selection Criteria

The canonical experiment used for the dashboard and paper trading is selected by the following reproducible rule:

1. **Primary criterion:** Highest Sharpe Ratio in walk-forward validation period (2021–2024).
2. **Exclusion:** Experiments with Max Drawdown > 30% are excluded.
3. **Tiebreak:** If two experiments have the same Sharpe Ratio, prioritize the higher Calmar Ratio.

This criterion is deterministic: running the walk-forward from scratch with the same seeds (`random_seed: 42`) produces the same results.

## Selected Experiment

| Field | Value |
|-------|-------|
| Experiment ID | `fase6_mat_cazador_conspiranoico_exp1_menos_friccion_20260622_035259` |
| Config Hash | `1782089830.517147` |
| Timestamp | 2026-06-22T03:52:59 |

## Metrics (Walk-Forward Validation, 2021–2024)

| Metric | Value |
|--------|-------|
| Total Return | 86.64% |
| Annualized Return | 17.78% |
| Sharpe Ratio | 0.861 |
| Max Drawdown | -29.63% |
| Calmar Ratio | 0.60 |
| Win Rate | 52.34% |
| Trading Days | 1005 |

## Window Breakdown

| Iter | Period | Return | Sharpe | MaxDD |
|------|--------|--------|--------|-------|
| 1 | 2021 | +23.97% | 1.426 | -10.23% |
| 2 | 2022 | -21.19% | — | — |
| 3 | 2023 | +44.55% | 2.334 | — |
| 4 | 2024 | — | — | — |

## Reproducibility

To reproduce this experiment:

```bash
python run.py --profile profiles/exp1_menos_friccion.yaml
```

The resulting `results.json` should match the config hash above when using the same input data (cached in `data/cache/`).
