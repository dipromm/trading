"""
El Analista — Sentimiento de noticias financieras con FinBERT.

Pipeline:
    1. Recibe titulares de noticias por ticker y fecha
    2. Procesa cada titular con FinBERT (positivo/negativo/neutral)
    3. Agrega los scores del dia en un score medio
    4. Calibra con Platt Scaling (CalibratedClassifierCV)
    5. Retorna probabilidad calibrada p in [0, 1] al Juez

IMPORTANTE sobre datos historicos:
    Las noticias de fin de semana se asignan al lunes siguiente.
    El backtester solo opera en dias habiles (NYSE calendar).
    Noticias publicadas despues del cierre se asignan al dia siguiente.

Fuente de datos: Alpaca News API (gratuita con cuenta paper trading).
FinBERT: ProsusAI/finbert (~438 MB, descarga automatica de Hugging Face).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from mas.agents.base_agent import AgentBase
from mas.utils.config_loader import load_config

logger = logging.getLogger(__name__)

_SENTIMENT_COL = "sentiment_raw"
_TARGET_COL = "target_binary"
_MIN_TRAIN_ROWS = 60


class Analista(AgentBase):
    """
    Agente de analisis de sentimiento basado en FinBERT.

    Dos fases de uso:
        1. **Precomputo** (una vez antes del walk-forward):
           ``precompute_sentiment(news_data, trading_dates)`` puntua
           todos los titulares con FinBERT y agrega por ticker/dia.
           El resultado se cachea en Parquet.

        2. **Walk-forward** (por cada ventana):
           ``fit(train_data)`` calibra el score de sentimiento contra
           el target real (subida/bajada).
           ``predict(data)`` retorna probabilidad calibrada p in [0,1].
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        self.config = config["analista"]
        self._full_config = config

        self._pipeline = None
        self._calibrator = None
        self._is_fitted = False
        self._n_train_rows: int = 0
        self._calibration_method: str = self.config.get("calibration_method", "sigmoid")
        self._cv_folds: int = self.config.get("calibration_cv_folds", 5)

    # ── FinBERT Pipeline ────────────────────────────────────────────────────

    def _load_finbert(self) -> None:
        """Lazy-load the FinBERT pipeline from Hugging Face."""
        if self._pipeline is not None:
            return

        import torch
        from transformers import pipeline as hf_pipeline

        model_name = self.config["model_name"]
        device = 0 if torch.cuda.is_available() else -1

        logger.info(
            "Cargando FinBERT (%s) en %s...",
            model_name,
            "GPU" if device == 0 else "CPU",
        )
        self._pipeline = hf_pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            truncation=True,
            max_length=self.config.get("max_length", 512),
        )
        logger.info("FinBERT cargado correctamente")

    def score_headline(self, headline: str) -> float:
        """
        Score a single headline with FinBERT.

        Returns:
            Score in [0, 1]: 1 = very positive, 0 = very negative, 0.5 = neutral.
        """
        self._load_finbert()
        text = headline[: self.config.get("max_length", 512)]
        result = self._pipeline(text)[0]

        label = result["label"].lower()
        score = result["score"]

        if label == "positive":
            return 0.5 + score / 2
        if label == "negative":
            return 0.5 - score / 2
        return 0.5

    def score_headlines_batch(self, headlines: list[str]) -> np.ndarray:
        """
        Score multiple headlines with FinBERT in batch.

        Returns:
            Array of scores in [0, 1] with same length as headlines.
        """
        if not headlines:
            return np.array([])

        self._load_finbert()
        batch_size = self.config.get("batch_size", 32)
        max_length = self.config.get("max_length", 512)

        truncated = [h[:max_length] for h in headlines]
        results = self._pipeline(truncated, batch_size=batch_size)

        scores = np.empty(len(results))
        for i, result in enumerate(results):
            label = result["label"].lower()
            s = result["score"]
            if label == "positive":
                scores[i] = 0.5 + s / 2
            elif label == "negative":
                scores[i] = 0.5 - s / 2
            else:
                scores[i] = 0.5

        return scores

    # ── Precomputo de Sentimiento ──────────────────────────────────────────

    def precompute_sentiment(
        self,
        news_data: dict[str, pd.DataFrame],
        trading_dates: pd.DatetimeIndex,
        cache_dir: Path | None = None,
        force: bool = False,
    ) -> dict[str, pd.Series]:
        """
        Run FinBERT on all headlines and aggregate daily per ticker.

        This is the expensive step (FinBERT inference). Results are cached
        to Parquet to avoid re-computing on subsequent runs.

        Args:
            news_data: {ticker: DataFrame} from download_all_news().
                       Each DataFrame has columns [headline, created_at, ...].
            trading_dates: NYSE trading days for the full backtest period.
            cache_dir: Directory for sentiment cache. Defaults to data/cache/sentiment/.
            force: If True, recompute even if cache exists.

        Returns:
            {ticker: Series} where each Series is indexed by trading day
            with the aggregated sentiment score for that day.
            Days without news get NaN (not 0.5, so the Juez can detect missing data).
        """
        from mas.data.news import assign_to_trading_days

        if cache_dir is None:
            cache_dir = Path(self._full_config["data"]["cache_dir"]) / "sentiment"
        cache_dir.mkdir(parents=True, exist_ok=True)

        aggregation = self.config.get("aggregation", "mean")
        result: dict[str, pd.Series] = {}

        tickers_to_process = []
        for ticker, news_df in news_data.items():
            cache_file = cache_dir / f"{ticker}_sentiment.parquet"
            if cache_file.exists() and not force:
                try:
                    cached = pd.read_parquet(cache_file)
                    result[ticker] = cached.set_index("trading_day")["sentiment_raw"]
                    continue
                except Exception:
                    pass
            tickers_to_process.append(ticker)

        if tickers_to_process:
            self._load_finbert()
            logger.info(
                "Precomputando sentimiento FinBERT para %d tickers...",
                len(tickers_to_process),
            )

            from tqdm import tqdm

            for ticker in tqdm(tickers_to_process, desc="Sentimiento", unit="ticker"):
                news_df = news_data[ticker]
                if news_df.empty:
                    result[ticker] = pd.Series(
                        dtype=float, name="sentiment_raw",
                    )
                    continue

                sentiment = self._process_ticker_news(
                    news_df, trading_dates, aggregation,
                )
                result[ticker] = sentiment

                cache_file = cache_dir / f"{ticker}_sentiment.parquet"
                save_df = sentiment.reset_index()
                save_df.columns = ["trading_day", "sentiment_raw"]
                save_df.to_parquet(cache_file, index=False)

        n_with_data = sum(1 for s in result.values() if not s.empty)
        logger.info(
            "Sentimiento precomputado: %d/%d tickers con datos",
            n_with_data, len(result),
        )
        return result

    def _process_ticker_news(
        self,
        news_df: pd.DataFrame,
        trading_dates: pd.DatetimeIndex,
        aggregation: str,
    ) -> pd.Series:
        """Score and aggregate news for a single ticker."""
        from mas.data.news import assign_to_trading_days

        mapped = assign_to_trading_days(news_df, trading_dates)
        mapped = mapped.dropna(subset=["trading_day"])

        if mapped.empty:
            return pd.Series(dtype=float, name="sentiment_raw")

        headlines = mapped["headline"].tolist()
        scores = self.score_headlines_batch(headlines)
        mapped = mapped.copy()
        mapped["sentiment_score"] = scores

        if aggregation == "mean":
            daily = mapped.groupby("trading_day")["sentiment_score"].mean()
        elif aggregation == "median":
            daily = mapped.groupby("trading_day")["sentiment_score"].median()
        else:
            daily = mapped.groupby("trading_day")["sentiment_score"].mean()

        daily.name = "sentiment_raw"
        return daily

    # ── Merge into Features ────────────────────────────────────────────────

    @staticmethod
    def merge_sentiment_into_features(
        features: dict[str, pd.DataFrame],
        sentiment: dict[str, pd.Series],
    ) -> dict[str, pd.DataFrame]:
        """
        Add sentiment_raw column to each ticker's features DataFrame.

        Days without news data get NaN (the Analista handles this gracefully
        in predict() by returning NaN for those rows).
        """
        for ticker, feat_df in features.items():
            if ticker in sentiment and not sentiment[ticker].empty:
                sent = sentiment[ticker]
                if not isinstance(feat_df.index, pd.DatetimeIndex):
                    feat_df.index = pd.to_datetime(feat_df.index)
                if not isinstance(sent.index, pd.DatetimeIndex):
                    sent.index = pd.to_datetime(sent.index)
                feat_df[_SENTIMENT_COL] = sent.reindex(feat_df.index)
            else:
                feat_df[_SENTIMENT_COL] = np.nan
        return features

    # ── AgentBase Interface ────────────────────────────────────────────────

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train the probability calibrator on historical sentiment vs. outcomes.

        The calibrator maps raw FinBERT sentiment to calibrated P(up).
        Uses CalibratedClassifierCV wrapping LogisticRegression.

        Args:
            train_data: DataFrame with 'sentiment_raw' and 'target_binary'.
                        Only data from the walk-forward training window.

        Raises:
            ValueError: If insufficient clean rows for calibration.
        """
        if _SENTIMENT_COL not in train_data.columns:
            raise ValueError(
                f"Columna '{_SENTIMENT_COL}' no encontrada. "
                f"Ejecuta precompute_sentiment() y merge_sentiment_into_features() primero."
            )
        if _TARGET_COL not in train_data.columns:
            raise ValueError(
                f"Columna '{_TARGET_COL}' no encontrada en train_data."
            )

        clean = train_data.dropna(subset=[_SENTIMENT_COL, _TARGET_COL])

        if len(clean) < _MIN_TRAIN_ROWS:
            raise ValueError(
                f"Solo {len(clean)} filas con sentimiento y target validos "
                f"(minimo requerido: {_MIN_TRAIN_ROWS}). "
                f"Posiblemente hay poca cobertura de noticias para este periodo."
            )

        X = clean[[_SENTIMENT_COL]].values
        y = clean[_TARGET_COL].values

        base = LogisticRegression(
            random_state=self.config.get("random_state", 42),
            max_iter=1000,
        )
        self._calibrator = CalibratedClassifierCV(
            estimator=base,
            method=self._calibration_method,
            cv=min(self._cv_folds, len(clean) // 20),
        )
        self._calibrator.fit(X, y)
        self._is_fitted = True
        self._n_train_rows = len(clean)

        logger.info(
            "Analista entrenado -- filas: %d, calibracion: %s",
            self._n_train_rows,
            self._calibration_method,
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Return calibrated P(up) from sentiment data.

        Args:
            data: DataFrame with 'sentiment_raw' column.
                  Rows with NaN sentiment get NaN prediction.

        Returns:
            Series of calibrated probabilities p in [0, 1].
        """
        if not self.is_fitted():
            raise RuntimeError(
                "Analista.predict() llamado sin entrenar. Llama a fit() primero."
            )

        result = pd.Series(np.nan, index=data.index, name="prob_up")

        if _SENTIMENT_COL not in data.columns:
            logger.warning("predict() sin columna '%s', retornando NaN", _SENTIMENT_COL)
            return result

        clean = data.dropna(subset=[_SENTIMENT_COL])
        if clean.empty:
            return result

        X = clean[[_SENTIMENT_COL]].values
        proba = self._calibrator.predict_proba(X)[:, 1]
        result.loc[clean.index] = proba

        return result

    # ── Diagnostics ──────────────────────────────────────────────────────

    def calibration_report(
        self,
        data: pd.DataFrame,
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """
        Evaluate calibration quality on validation data.

        Args:
            data: DataFrame with 'sentiment_raw' and 'target_binary'.

        Returns:
            dict with Brier Score, Log Loss, and reliability diagram data.
        """
        if not self.is_fitted():
            raise RuntimeError("Llama a fit() antes de calibration_report().")

        clean = data.dropna(subset=[_SENTIMENT_COL, _TARGET_COL])
        if clean.empty:
            raise ValueError("No hay filas limpias para evaluar.")

        X = clean[[_SENTIMENT_COL]].values
        y_true = clean[_TARGET_COL].values
        y_prob = self._calibrator.predict_proba(X)[:, 1]

        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform",
        )

        report = {
            "brier_score": round(float(brier_score_loss(y_true, y_prob)), 5),
            "log_loss": round(float(log_loss(y_true, y_prob)), 5),
            "fraction_of_positives": fraction_pos,
            "mean_predicted_value": mean_pred,
            "n_samples": len(clean),
        }

        logger.info(
            "Analista calibracion -- Brier: %.4f, LogLoss: %.4f, muestras: %d",
            report["brier_score"], report["log_loss"], report["n_samples"],
        )
        return report

    def sentiment_coverage(
        self,
        features: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Report how many days have sentiment data per ticker.

        Useful for diagnosing sparse news coverage.
        """
        rows = []
        for ticker, df in features.items():
            total = len(df)
            with_sent = df[_SENTIMENT_COL].notna().sum() if _SENTIMENT_COL in df.columns else 0
            rows.append({
                "ticker": ticker,
                "total_days": total,
                "days_with_sentiment": int(with_sent),
                "coverage_pct": round(100 * with_sent / total, 1) if total > 0 else 0,
            })
        return pd.DataFrame(rows).sort_values("coverage_pct", ascending=False)
