"""
Diagnóstico de El Conspiranoico — Análisis del veto de régimen.

Genera un reporte visual (consola + CSV) para validar que el Isolation Forest
detecta correctamente los períodos de crisis documentados:
    - Marzo 2020 (crash COVID)
    - 2022 (caídas por subida de tipos de la Fed)

El script NO modifica el backtester. Entrena el Conspiranoico sobre el
período 2018-2019 y predice sobre 2020-2024 para ver qué días activa el veto.
Esto es un test visual de sanity check, no un experimento de backtest.

Uso:
    python scripts/diagnose_conspiranoico.py
    python scripts/diagnose_conspiranoico.py --train-end 2020-12-31 --val-start 2021-01-01
    python scripts/diagnose_conspiranoico.py --output reports/veto_analysis.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silenciar libs ruidosas
for noisy in ("yfinance", "urllib3", "requests"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_section(title: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _crisis_coverage(veto: pd.Series, label: str, start: str, end: str) -> None:
    """Imprime el % de días con veto en un período de crisis."""
    period = veto.loc[start:end]
    if period.empty:
        print(f"  {label}: sin datos en [{start} -> {end}]")
        return
    n_veto = int(period.sum())
    n_total = len(period)
    pct = n_veto / n_total * 100
    status = "[OK]" if n_veto > 0 else "[!!]"
    print(f"  {status} {label}: {n_veto}/{n_total} días con veto ({pct:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico del Conspiranoico")
    parser.add_argument(
        "--train-end", default="2019-12-31",
        help="Fecha de fin del período de entrenamiento (default: 2019-12-31)",
    )
    parser.add_argument(
        "--val-start", default="2020-01-01",
        help="Fecha de inicio del período de predicción/diagnóstico (default: 2020-01-01)",
    )
    parser.add_argument(
        "--val-end", default="2024-12-31",
        help="Fecha de fin del período de predicción (default: 2024-12-31)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Ruta opcional para guardar el CSV de diagnóstico",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-descarga VIX ignorando caché",
    )
    args = parser.parse_args()

    from utils.config_loader import load_config
    cfg = load_config()

    # -- 1. Cargar datos OHLCV --
    logger.info("Cargando datos OHLCV...")
    from data.downloader import download_all
    prices = download_all(cfg)
    if not prices:
        logger.error("No hay datos disponibles. Ejecuta run.py primero.")
        return 1
    logger.info("  %d tickers cargados", len(prices))

    # -- 2. Descargar VIX + features de régimen --
    logger.info("Descargando VIX y calculando features de régimen...")
    from data.regime import build_regime_features, download_vix
    vix = download_vix(cfg, force_download=args.force_download)
    regime_features = build_regime_features(prices, vix, cfg)
    logger.info(
        "  Features de régimen: %d fechas (%s -> %s)",
        len(regime_features),
        regime_features.index[0].date(),
        regime_features.index[-1].date(),
    )

    # -- 3. Entrenar Conspiranoico en período de train --
    logger.info("Entrenando Conspiranoico [%s -> %s]...", "2018-01-01", args.train_end)
    from agents.conspiranoico import Conspiranoico
    conspiranoico = Conspiranoico(cfg)

    train_data = regime_features.loc["2018-01-01":args.train_end]
    conspiranoico.fit(train_data)

    # -- 4. Predecir en período de diagnóstico --
    logger.info("Prediciendo veto [%s -> %s]...", args.val_start, args.val_end)
    val_data = regime_features.loc[args.val_start:args.val_end]
    veto = conspiranoico.predict(val_data)
    scores = conspiranoico.anomaly_scores(val_data)

    # -- 5. Reporte de crisis documentadas --
    _print_section("SANITY CHECK — Períodos de crisis documentados")
    _crisis_coverage(veto, "Crash COVID (Mar 2020)", "2020-03-01", "2020-03-31")
    _crisis_coverage(veto, "Crisis 2022 Q1", "2022-01-01", "2022-03-31")
    _crisis_coverage(veto, "Crisis 2022 completa", "2022-01-01", "2022-12-31")
    _crisis_coverage(veto, "2021 (bull market, veto esperado bajo)", "2021-01-01", "2021-12-31")
    _crisis_coverage(veto, "2023 (recuperación, veto esperado bajo)", "2023-01-01", "2023-12-31")
    _crisis_coverage(veto, "2024 (bull market IA)", "2024-01-01", "2024-12-31")

    # -- 6. Resumen global --
    _print_section("RESUMEN GLOBAL")
    total_days = len(veto)
    veto_days = int(veto.sum())
    print(f"  Período total:    {val_data.index[0].date()} -> {val_data.index[-1].date()}")
    print(f"  Días evaluados:   {total_days}")
    print(f"  Días con veto:    {veto_days} ({veto_days/total_days*100:.1f}%)")
    print(f"  Umbral de veto:   {conspiranoico.veto_threshold:.4f}")

    # -- 7. Top 10 días con mayor anomalía (scores más negativos) --
    _print_section("TOP 20 DÍAS MÁS ANÓMALOS (menor anomaly score)")
    top_anomalies = scores.dropna().sort_values().head(20)
    print(f"  {'Fecha':<12} {'Score':>10} {'Veto':>6} {'VIX':>8}")
    print("  " + "-" * 38)
    vix_aligned = vix.reindex(val_data.index, method="ffill")
    for date, score in top_anomalies.items():
        vix_val = vix_aligned.get(date, float("nan"))
        veto_val = int(veto.get(date, 0))
        print(
            f"  {str(date.date()):<12}"
            f" {score:>10.4f}"
            f" {'SI' if veto_val else 'no':>6}"
            f" {vix_val:>8.2f}"
        )

    # -- 8. Distribución anual del veto --
    _print_section("VETO POR AÑO")
    veto_df = veto.to_frame("veto")
    veto_df["year"] = veto_df.index.year
    annual = veto_df.groupby("year")["veto"].agg(["sum", "count"])
    annual["pct"] = annual["sum"] / annual["count"] * 100
    print(f"  {'Año':<6} {'Días c/veto':>12} {'Total días':>12} {'% veto':>8}")
    print("  " + "-" * 42)
    for year, row in annual.iterrows():
        print(
            f"  {year:<6}"
            f" {int(row['sum']):>12}"
            f" {int(row['count']):>12}"
            f" {row['pct']:>7.1f}%"
        )

    # -- 9. Guardar CSV de diagnóstico --
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diag_df = pd.DataFrame({
            "veto": veto,
            "anomaly_score": scores,
            "vix": vix_aligned,
        })
        diag_df.to_csv(output_path)
        print(f"\n  CSV guardado en: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
