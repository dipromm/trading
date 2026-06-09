"""
Métricas de evaluación cuantitativa del sistema.

El éxito del proyecto NO se mide por beneficio neto (fomenta overfitting).
Se mide por Sharpe Ratio, Maximum Drawdown y outperformance sobre los baselines.

Objetivos mínimos en validación walk-forward (NO en entrenamiento):
    - Sharpe Ratio > 1.0
    - Maximum Drawdown < 25%
    - Calmar Ratio > 0.5
    - Sharpe MAS > Sharpe Buy & Hold
"""

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe Ratio anualizado.

    Args:
        returns: Retornos diarios del sistema
        risk_free_rate: Tasa libre de riesgo anualizada (0 para simplificar)
        periods_per_year: 252 para datos diarios de mercado

    Returns:
        Sharpe Ratio anualizado. Objetivo: > 1.0 en walk-forward.
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum Drawdown: peor caída desde un máximo histórico.

    Returns:
        Valor negativo o cero. Ejemplo: -0.25 = caída máxima del 25%.
        Objetivo: > -0.25 (es decir, drawdown menor al 25%).
    """
    if equity_curve.empty:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calmar Ratio: retorno anual medio / |Max Drawdown|.

    Un Calmar > 0.5 indica que el retorno compensa razonablemente el riesgo.
    """
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    annual_return = returns.mean() * periods_per_year
    return annual_return / mdd


def win_rate(returns: pd.Series) -> float:
    """
    Porcentaje de días con retorno positivo.

    Nota: esta métrica es menos importante que el Sharpe. Un sistema puede
    ser rentable con win rate < 50% si las ganancias son mayores que las pérdidas.
    """
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def total_return(equity_curve: pd.Series) -> float:
    """
    Retorno total como fracción del capital inicial.
    Ejemplo: 0.35 = +35% de retorno total.
    """
    if equity_curve.empty or equity_curve.iloc[0] == 0:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Retorno anualizado medio."""
    return float(returns.mean() * periods_per_year)


def summary(equity_curve: pd.Series, returns: pd.Series) -> dict:
    """
    Calcula y retorna todas las métricas en un diccionario.

    Uso:
        metrics = summary(equity_curve, returns)
        print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    return {
        "total_return_pct": round(total_return(equity_curve) * 100, 2),
        "annualized_return_pct": round(annualized_return(returns) * 100, 2),
        "sharpe_ratio": round(sharpe_ratio(returns), 3),
        "max_drawdown_pct": round(max_drawdown(equity_curve) * 100, 2),
        "calmar_ratio": round(calmar_ratio(returns, equity_curve), 3),
        "win_rate_pct": round(win_rate(returns) * 100, 2),
        "n_trading_days": len(returns),
    }


def compare_to_baseline(
    system_equity: pd.Series,
    system_returns: pd.Series,
    baseline_equity: pd.Series,
    baseline_returns: pd.Series,
    baseline_name: str = "Buy & Hold",
) -> dict:
    """
    Compara las métricas del sistema contra un baseline.

    Returns:
        dict con métricas del sistema, baseline, y la diferencia.
    """
    sys_metrics = summary(system_equity, system_returns)
    base_metrics = summary(baseline_equity, baseline_returns)

    return {
        "system": sys_metrics,
        "baseline": {**base_metrics, "name": baseline_name},
        "outperformance": {
            "sharpe_delta": round(sys_metrics["sharpe_ratio"] - base_metrics["sharpe_ratio"], 3),
            "return_delta_pct": round(sys_metrics["total_return_pct"] - base_metrics["total_return_pct"], 2),
            "drawdown_improvement_pct": round(
                base_metrics["max_drawdown_pct"] - sys_metrics["max_drawdown_pct"], 2
            ),
        },
    }
