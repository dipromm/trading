# Trading MAS — Sistema Multi-Agente para Predicción Bursátil

Arquitectura de ensemble de agentes especializados para la toma de decisiones de inversión sobre un universo diversificado: **40 acciones del Nasdaq 100** + **6 ETFs** de renta fija, oro, sectores defensivos e internacional.

## Arquitectura

| Agente | Función | Tecnología |
|---|---|---|
| **El Matemático** | Análisis técnico (RSI, MACD, ATR...) | XGBoost + Platt Scaling |
| **El Analista** | Sentimiento de noticias financieras | FinBERT (Hugging Face) |
| **El Cazador** | Actividad de insiders (SEC Form 4) | OpenInsider + reglas |
| **El Conspiranoico** | Detección de régimen anómalo (veto) | Isolation Forest + HMM |
| **El Gestor de Riesgos** | Tamaño de posición | Fractional Kelly (Half-Kelly) |
| **El Juez Central** | Decisión final | Ensemble meta-modelo |
| **El Explorador** *(v2)* | Gestión dinámica del universo | Estadística sobre trades + yfinance |

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Uso rápido

```bash
# Descargar datos
python -m data.downloader

# Ejecutar backtesting completo (disponible a partir de la Fase 3)
# python -m backtester.walk_forward

# Ejecutar con un perfil de trading específico (Fase 8)
# python run.py --profile profiles/long_term.yaml
# python run.py --profile profiles/day_simulated.yaml

# Lanzar dashboard (disponible a partir de la Fase 7)
# streamlit run dashboard/app.py
```

## Tests

```bash
pytest tests/ -v
```

## Hoja de ruta

| Fase | Semanas | Objetivo |
|---|---|---|
| 1 — Datos | 1-2 | Pipeline de datos limpio |
| 2 — Backtester | 3-4 | Framework de validación + baselines |
| 3 — MVP | 5-6 | El Matemático + Juez v1 |
| 4 — Analista | 7-8 | NLP + ensemble de 2 agentes |
| 5 — Riesgos/Insiders | 9-10 | Kelly + Cazador |
| 6 — Conspiranoico/RL | 11-12 | Detección de régimen + Juez v2 |
| 7 — Dashboard | 13-14 | Streamlit + XAI logs |
| 8 — Perfiles | 15-16 | Swing, Long-term y Day Simulated comparados |
| 9 — El Explorador | 17-18 | Gestión dinámica del universo con Human-in-the-Loop |
| 10 — Demo pública | 19-20 | Dashboard público + web portfolio |

## Perfiles de trading

El mismo sistema opera con tres configuraciones de horizonte temporal, comparables entre sí:

| Perfil | Frecuencia | Horizonte | Nota |
|---|---|---|---|
| **Swing** | Diaria | Días–semanas | Perfil por defecto |
| **Long-term** | Mensual | Meses | SMA 50/200, menos comisiones |
| **Day simulated** | Diaria (Open→Close) | 1 día | **Solo académico.** Proxy con datos diarios, no modela spread/slippage real |

## Reproducibilidad

Todos los resultados son reproducibles con `random_seed: 42` en `config.yaml`.

```bash
pytest tests/test_backtester.py::test_reproducibility_same_seed -v
```

## Universo de activos

| Clase | Tickers | Cap por ticker | Cap total |
|---|---|---|---|
| Renta Variable (Nasdaq) | 40 acciones | 15% | 80% |
| Renta Fija | TLT, IEF | 20% | 30% |
| Oro | GLD | 10% | 10% |
| Defensivos | XLU, XLP | 15% | 20% |
| Internacional | EFA | 10% | 10% |

## Métricas objetivo (walk-forward, no holdout)

| Métrica | Objetivo |
|---|---|
| Sharpe Ratio | > 1.0 |
| Maximum Drawdown | < 25% |
| Calmar Ratio | > 0.5 |
| Outperformance | Sharpe MAS > Sharpe Buy & Hold |
