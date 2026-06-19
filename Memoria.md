# Memoria Técnica del Proyecto
## Sistema Multi-Agente (MAS) para Predicción y Toma de Decisiones en Mercados Financieros

> **Propósito de este documento:** Registro completo de las decisiones de diseño, metodología, implementación y estado actual del proyecto. Funciona como referencia técnica ante cualquier duda futura sobre el porqué de cada decisión.

---

## Tabla de Contenidos

1. [Contexto y Objetivos](#1-contexto-y-objetivos)
2. [Decisiones de Diseño Fundamentales](#2-decisiones-de-diseño-fundamentales)
3. [Arquitectura del Sistema](#3-arquitectura-del-sistema)
4. [El Universo de Activos](#4-el-universo-de-activos)
5. [Framework Metodológico](#5-framework-metodológico)
6. [Implementación: Componentes Completados](#6-implementación-componentes-completados)
7. [Implementación: Stubs (Pendientes de Completar)](#7-implementación-stubs-pendientes-de-completar)
8. [Sistema de Configuración](#8-sistema-de-configuración)
9. [Tests Implementados](#9-tests-implementados)
10. [Estructura del Repositorio](#10-estructura-del-repositorio)
11. [Expansión v2: Perfiles de Trading y El Explorador](#11-expansión-v2-perfiles-de-trading-y-el-explorador)
12. [Trampas Técnicas y Cómo Se Evitan](#12-trampas-técnicas-y-cómo-se-evitan) *(12 entradas)*
13. [Sesgos Conocidos y Limitaciones](#13-sesgos-conocidos-y-limitaciones)
14. [Stack Tecnológico](#14-stack-tecnológico)
15. [Hoja de Ruta](#15-hoja-de-ruta)

---

## 1. Contexto y Objetivos

### 1.1 Origen del proyecto

El proyecto nació de un plan generado por Gemini para un sistema de predicción bursátil. Ese plan inicial fue revisado y sustancialmente mejorado para incorporar buenas prácticas de quant trading que el borrador original omitía.

**Problemas encontrados en el plan original (y cómo se corrigieron):**

| Problema en el plan original | Corrección aplicada |
|---|---|
| Backtesting colocado al final del roadmap | Movido a Fase 2, antes de entrenar ningún modelo |
| Juez Central con RL desde el MVP | RL aplazado a v2; MVP usa ensemble estático |
| Clasificación ternaria (sube/baja/lateral) | Cambiado a binaria (sube/no sube); la abstención la decide Kelly |
| Sin calibración de probabilidades | Platt Scaling obligatorio antes de pasar p a Kelly |
| Sin baselines definidos | Buy & Hold y SMA Crossover como referencia mínima |
| Sin control de reproducibilidad | Seeds fijas en todo el stack (Python, NumPy, PyTorch) |
| Sin centralización de configuración | `config.yaml` como fuente única de verdad |
| Solo acciones tech del Nasdaq | 40 acciones + 6 ETFs de diversificación |
| Sin caps diferenciados por clase de activo | Caps por ticker y por clase en `GestorRiesgos` |
| Sin discusión de data leakage | Walk-forward obligatorio, tests anti-leakage |

### 1.2 Objetivos reales del proyecto

- **Objetivo principal:** aprender arquitecturas de ML para finanzas cuantitativas.
- **Objetivo secundario:** resultado visible para portfolio web de un ingeniero informático.
- **Objetivo terciario:** si el sistema funciona bien, podría usarse para decisiones de inversión simulada.

El sistema opera **solo en paper trading** (simulación con datos reales, sin dinero real). No está diseñado para trading en producción real.

### 1.3 Restricciones de diseño no negociables

| Parámetro | Decisión | Justificación |
|---|---|---|
| **Mercado** | Acciones US (Nasdaq 100) | Mayor disponibilidad de datos gratuitos y de calidad |
| **Universo** | 40 acciones + 6 ETFs | Sweet spot entre suficientes datos y manejabilidad |
| **Horizonte principal** | Señales diarias (swing) | Un horizonte coherente evita combinar señales incompatibles |
| **Solo Long** | No posiciones cortas | Evita complejidades de borrowing fees y margin requirements |
| **Capital simulado** | 10.000 € ficticios | Referencia realista para las métricas |
| **Comisiones** | 0,08% por operación (ida) | Aproximación de IBKR/Degiro; destruye estrategias con exceso de operaciones |
| **Período de datos** | 2018-01-01 a 2024-12-31 | 7 años de entrenamiento/validación |
| **Holdout** | 2025-01-01 en adelante | Reservado y sellado hasta que el sistema esté terminado |

---

## 2. Decisiones de Diseño Fundamentales

### 2.1 Por qué clasificación binaria y no ternaria

El plan original proponía predecir tres clases: sube, baja, lateral. Se rechazó por dos razones:

1. **El umbral de "lateral" es un hiperparámetro arbitrario:** definir qué es "lateral" (+/- 0.1%, +/- 0.5%...) es una decisión que introduce sobreajuste implícito.
2. **La abstención ya está resuelta por Kelly:** si el modelo predice p=0.5 (total incertidumbre), Kelly calcula f*=0 automáticamente. La señal de "no operar" no necesita ser una clase explícita.

**Implementación:** target binario donde 1 = retorno del día siguiente > 0%, 0 = retorno ≤ 0%.

### 2.2 Por qué calibración de probabilidades es obligatoria

El Criterio de Kelly asume que `p` es una probabilidad real (frecuentista). Si el modelo XGBoost dice 0.8 pero la frecuencia real de subidas con esa predicción es solo 0.55, Kelly calculará una posición absurdamente grande.

**Solución:** `CalibratedClassifierCV` de scikit-learn con Platt Scaling (`method='sigmoid'`) como calibrador por defecto. La calibración se verifica con un **reliability diagram** (`calibration_curve` de sklearn).

El ciclo de verificación es:
- Si la curva del reliability diagram se aproxima a la diagonal → calibración correcta.
- Si se desvía significativamente → probar Isotonic Regression (`method='isotonic'`).

### 2.3 Por qué Half-Kelly y no Kelly puro

El Criterio de Kelly puro maximiza la tasa de crecimiento geométrico, pero:

- Es extremadamente sensible a errores en la estimación de `p`.
- Un error del 1% en `p` puede llevar a posiciones ruinosas.
- En finanzas, las probabilidades son siempre estimaciones imperfectas.

**Solución:** Fractional Kelly con `ρ = 0.5` (Half-Kelly):

```
f* = ρ × (p - (1-p)/b)
```

donde:
- `p` = probabilidad calibrada de subida
- `b` = ratio ganancia media / pérdida media histórico (ventana de 60 días de entrenamiento)
- `ρ = 0.5` = factor de seguridad Half-Kelly

El resultado se capea adicionalmente por clase de activo (ver sección 4.3).

### 2.4 Por qué el Juez v1 no usa RL

El plan original proponía Aprendizaje por Refuerzo desde el inicio. Se rechazó por:

1. El RL en finanzas tiende a aprender la política trivial de "no invertir nunca" si la función de recompensa no está muy bien diseñada.
2. Añade dificultad muy alta sin un baseline sólido previo que permita medir si mejora algo.
3. Puede bloquear el proyecto completo si la función de recompensa falla.

**El Juez v1** es un meta-modelo estático: una regresión logística o XGBoost entrenado sobre las probabilidades de los agentes como features. Simple, interpretable, y suficiente para el MVP.

**El Juez v2** (RL con PPO via `stable-baselines3`) queda como mejora opcional en Fase 6, solo si el v1 funciona bien.

### 2.5 Por qué el backtesting va en la Fase 2

Sin un framework de validación correcto, cualquier resultado de modelo es inválido. Si se construye primero el modelo y luego el backtester, es muy probable encontrar data leakage que invalide meses de trabajo.

**Regla:** el backtester y los baselines se construyen primero. Los agentes se desarrollan después, usando el backtester como juez.

### 2.6 Por qué walk-forward y no un único train/test split

Un único split temporal (ej: 2018-2022 train, 2023-2024 test) no detecta si el modelo se sobreajusta a un régimen de mercado específico. El mercado de 2020-2021 (bull run post-COVID) es radicalmente distinto al de 2022 (caídas de tipos).

**Walk-forward con la configuración actual:**
```
ITER 1: ENTRENA [2018-2021] → VALIDA [2021]
ITER 2: ENTRENA [2018-2022] → VALIDA [2022]
ITER 3: ENTRENA [2018-2023] → VALIDA [2023]
ITER 4: ENTRENA [2018-2024] → VALIDA [2024]
HOLDOUT [2025]: NO TOCAR hasta que el sistema esté terminado
```

Las predicciones de validación de las 4 iteraciones se concatenan para obtener ~4 años de predicciones out-of-sample, sobre las que se calculan las métricas finales.

---

## 3. Arquitectura del Sistema

### 3.1 Visión general

El sistema es un **ensemble de agentes especializados**. Cada agente evalúa el mercado desde una perspectiva aislada y envía su "voto" al Juez Central, que pondera las señales y emite la orden final. El Gestor de Riesgos traduce esa señal en un tamaño de posición.

```
                    ┌─────────────────────────────────────┐
                    │           DATOS DE ENTRADA           │
                    │  OHLCV + Noticias + Form 4 + VIX    │
                    └──────────────────┬──────────────────┘
                                       │
               ┌───────────┬───────────┴───────────┬───────────┐
               ▼           ▼                       ▼           ▼
        ┌─────────┐  ┌──────────┐           ┌──────────┐ ┌──────────────┐
        │   EL    │  │    EL    │           │    EL    │ │      EL      │
        │MATEMÁ-  │  │ANALISTA  │           │ CAZADOR  │ │CONSPIRANOICO │
        │  TICO   │  │(NLP)     │           │(Insiders)│ │  (Anomalías) │
        │XGBoost  │  │FinBERT   │           │Form 4    │ │ Isolation    │
        │+ Platt  │  │+ Platt   │           │Reglas    │ │  Forest      │
        └────┬────┘  └────┬─────┘           └────┬─────┘ └──────┬───────┘
             │            │                      │              │
             │  p∈[0,1]   │  p∈[0,1]            │  alerta/0   │  veto/0
             └────────────┴──────────────────────┘              │
                                  │                              │
                                  ▼                              │
                         ┌─────────────────┐                    │
                         │  EL JUEZ CENTRAL │◄──────────────────┘
                         │  v1: Logistic /  │ (veto reduce expo a 0%)
                         │  XGBoost meta    │
                         └────────┬────────┘
                                  │  señal final p ∈ [0,1]
                                  ▼
                         ┌─────────────────┐
                         │  EL GESTOR DE   │
                         │     RIESGOS     │
                         │  Fractional     │
                         │  Kelly + caps   │
                         │  + stop-loss    │
                         └────────┬────────┘
                                  │  f* ∈ [0, cap_clase]
                                  ▼
                         ┌─────────────────┐
                         │   BACKTEST      │
                         │   ENGINE        │
                         │ BUY/SELL/HOLD   │
                         │ + comisiones    │
                         │ + log auditable │
                         └─────────────────┘
```

### 3.2 Tabla de agentes

| Agente | Función | Tecnología | Dificultad | Output al Juez | Estado |
|---|---|---|---|---|---|
| **El Matemático** | Análisis técnico (RSI, MACD, ATR, Bollinger) | XGBoost + CalibratedClassifierCV | Media | p ∈ [0,1] calibrada | ✅ Implementado |
| **El Analista** | Sentimiento de noticias financieras | FinBERT (ProsusAI/finbert) | Media-Alta | p ∈ [0,1] calibrada | ✅ Implementado |
| **El Cazador** | Actividad de insiders (SEC Form 4) | SEC EDGAR + reglas | Media | Señal binaria (alerta/silencio) | ✅ Fase 5 |
| **El Conspiranoico** | Detección de régimen anómalo (crisis) | Isolation Forest + HMM | Alta | Veto binario (activo/inactivo) | 🔲 Stub |
| **El Gestor de Riesgos** | Tamaño de posición + stop-loss | Fractional Kelly (Half-Kelly) | Baja | f* ∈ [0, cap_clase] | ✅ Implementado |
| **El Juez Central** | Combina señales de agentes | v1: Logistic/XGBoost; v2: RL/PPO | Media (v1) | Señal final p ∈ [0,1] | ✅ Implementado |
| **El Explorador** *(v2)* | Gestión dinámica del universo | Estadística + yfinance | Baja-Media | Recomendación (HITL) | 🔲 Stub |

### 3.3 El baseline — referencia mínima

Sin baselines, cualquier métrica es ininterpretable. Antes de evaluar el sistema, se documentan dos baselines:

| Baseline | Descripción | Por qué importa |
|---|---|---|
| **Buy & Hold** | Comprar todo el universo en partes iguales y no tocar | Si el sistema no supera esto, es peor que no hacer nada |
| **SMA Crossover (20/50)** | Comprar cuando SMA20 cruza sobre SMA50 | Mínimo bar para un sistema ML |

Si el Consejo de Agentes no supera ambos baselines en walk-forward, los resultados no son creíbles.

---

## 4. El Universo de Activos

### 4.1 Criterios de selección

El universo se seleccionó siguiendo criterios explícitos para garantizar que el backtest sea creíble ante revisión técnica:

- **Fecha de congelación: 1 de enero de 2018.** Los tickers se eligieron tal como estaban disponibles en esa fecha, no a posteriori. Esto evita el survivorship bias retrospectivo (elegir empresas que "sabemos" que sobrevivieron hasta 2026).
- **40 acciones del Nasdaq 100** por capitalización bursátil en enero 2018.
- **Inclusión de underperformers:** INTC, GILD, BIIB, WBA, BIDU, PYPL, ILMN. Se incluyen deliberadamente para que el backtest sea creíble y no parezca cherry-picking.
- **6 ETFs de diversificación** (detallados abajo).

### 4.2 Los 6 ETFs de diversificación

Se añadieron para proteger la cartera ante una crisis del mercado americano o tecnológico:

| Ticker | Nombre | Clase | Función |
|---|---|---|---|
| **TLT** | iShares 20+ Year Treasury Bond ETF | `bond` | Refugio principal en crisis: correlación negativa con acciones |
| **IEF** | iShares 7-10 Year Treasury Bond ETF | `bond` | Estabilidad; menos sensible a tipos que TLT |
| **GLD** | SPDR Gold Shares ETF | `gold` | Refugio clásico; descorrelacionado de acciones y bonos |
| **XLU** | Utilities Select Sector SPDR | `defensive_equity` | Sector estable (electricidad, agua); baja correlación con tech |
| **XLP** | Consumer Staples Select Sector SPDR | `defensive_equity` | Empresas defensivas (P&G, Coca-Cola); resistentes a recesión |
| **EFA** | iShares MSCI EAFE ETF | `international_equity` | Diversificación geográfica (Europa + Japón + Australia) |

**Total del universo: 46 activos** (40 acciones + 6 ETFs).

### 4.3 Caps diferenciados por clase de activo

El Gestor de Riesgos aplica restricciones en dos niveles para cada clase:

| Clase de activo | Cap por ticker individual | Cap total de esa clase en cartera |
|---|---|---|
| `equity` (acciones Nasdaq) | 15% | 80% |
| `bond` (TLT, IEF) | 20% | 30% |
| `gold` (GLD) | 10% | 10% |
| `defensive_equity` (XLU, XLP) | 15% | 20% |
| `international_equity` (EFA) | 10% | 10% |

Los bonos tienen cap por ticker mayor (20%) porque su volatilidad es significativamente menor que la de las acciones. Kelly lo reflejaría de todas formas, pero el cap actúa como límite de seguridad adicional.

### 4.4 Lo que NO se incluye (y por qué)

| Activo | Decisión | Justificación |
|---|---|---|
| **Bitcoin / Crypto** | ❌ Excluido en v1 | Volatilidad 5-10× mayor; opera 24/7 rompiendo calendario NYSE |
| **Acciones europeas individuales** | ❌ Excluido | Festivos distintos, divisas, cobertura de noticias insuficiente |
| **Acciones emergentes individuales** | ❌ Excluido | Form 4 de SEC no aplica; datos de menor calidad |
| **Materias primas (petróleo, trigo)** | ❌ Excluido | Alta especificidad de señales; GLD cubre el refugio |

Bitcoin se puede añadir en v2 con un sub-agente propio, cap del 2-3%, y ajuste del backtester para calendario 24/7.

### 4.5 Los dos universos del sistema

Una distinción fundamental que afecta al diseño de varios componentes:

| Universo | Archivo | Quién lo usa | Puede cambiar |
|---|---|---|---|
| **Backtest** | `data/universe_2018-01-01.csv` | `WalkForwardValidator`, backtester | **NUNCA** — modificarlo invalida los resultados históricos |
| **Producción** | `data/universe_live.csv` | `BacktestEngine` en modo paper trading | Sí, con aprobación humana via El Explorador |

---

## 5. Framework Metodológico

### 5.1 El pipeline completo

```
Datos OHLCV (yfinance)
    │
    ▼
compute_all_features()  ← data/features.py
    │   RSI, MACD, Bollinger, ATR, volumen, SMA ratios
    │   + target_binary (shift(-1) del retorno)
    │
    ▼
WalkForwardValidator.run()  ← backtester/walk_forward.py
    │
    ├─ Por cada ventana walk-forward:
    │   ├─ Cortar datos (train / val)
    │   ├─ Agentes.fit(train_data)
    │   ├─ Agentes.predict(val_data) → p por ticker
    │   ├─ Juez.predict(p_matrix, veto) → señal final
    │   ├─ GestorRiesgos.compute_position_size() → f*
    │   ├─ GestorRiesgos.normalize_portfolio_by_class() → f* normalizado
    │   ├─ BacktestEngine.run() → equity_curve, daily_returns
    │   └─ BacktestEngine.save_trade_logs() → logs/trades/
    │
    └─ Concatenar returns de todas las ventanas
        └─ metrics.summary() → Sharpe, MaxDD, Calmar, WinRate
```

### 5.2 Features técnicas implementadas

Todos los indicadores son **lookback-safe**: el valor del día T solo usa datos hasta T-1 (o T para el propio precio de cierre). Nunca datos futuros.

| Feature | Descripción | Parámetros |
|---|---|---|
| `close_norm` | Precio vs SMA20 | — |
| `return_1d`, `return_5d`, `return_20d` | Retornos históricos | — |
| `rsi_14`, `rsi_28` | RSI — Relative Strength Index | ventana 14, 28 |
| `macd`, `macd_signal`, `macd_hist` | MACD | fast=12, slow=26, signal=9 |
| `bb_bandwidth`, `bb_pct_b` | Posición en Bollinger Bands | window=20, std=2.0 |
| `atr`, `atr_pct` | Average True Range (volatilidad) | window=14 |
| `volume_ratio` | Volumen relativo vs media N días | window=20 |
| `sma_20_50_ratio`, `sma_50_200_ratio` | Ratios de medias móviles | — |
| `target_binary` | **Solo para entrenar.** Retorno del día siguiente > 0 | shift(-1) |

### 5.3 El target y el anti-leakage del target

El target `target_binary` se calcula como `close.pct_change(1).shift(-1) > 0`. El `shift(-1)` asigna el retorno del día T+1 como label del día T.

**Riesgo crítico:** si el modelo usara `target_binary` como feature de entrada (en vez de solo como label de entrenamiento), estaría usando información del futuro. La columna `target_binary` nunca aparece en `FEATURE_COLUMNS`.

### 5.4 Métricas de evaluación

El sistema no se mide por beneficio neto (que fomenta overfitting). Se mide por:

| Métrica | Qué mide | Objetivo walk-forward |
|---|---|---|
| **Sharpe Ratio** | Retorno ajustado por volatilidad | > 1.0 |
| **Maximum Drawdown** | Peor caída desde máximo | < -25% (es decir, drawdown < 25%) |
| **Calmar Ratio** | Retorno anual / |MaxDD| | > 0.5 |
| **Win Rate** | % días con retorno positivo | Informativo; menos importante que Sharpe |
| **Outperformance** | Sharpe MAS > Sharpe Buy & Hold | Obligatorio para justificar la complejidad |

**Expectativa honesta:** un Sharpe de 1.0-1.5 con MaxDD < 25% en walk-forward es técnicamente sólido y creíble para portfolio. Un Sharpe > 2.0 en un único split debe tratarse con escepticismo.

---

## 6. Implementación: Componentes Completados

### 6.1 `data/features.py` — Pipeline de features técnicas

**Estado:** Completamente implementado.

Funciones disponibles:
- `compute_rsi(close, window)` — RSI con EWM (más estable que SMA rolling)
- `compute_macd(close, fast, slow, signal)` — retorna DataFrame con macd, macd_signal, macd_hist
- `compute_bollinger_bands(close, window, num_std)` — retorna bb_middle, bb_upper, bb_lower, bb_bandwidth, bb_pct_b
- `compute_atr(high, low, close, window)` — True Range con EWM
- `compute_volume_ratio(volume, window)` — volumen relativo vs media
- `compute_returns(close, periods)` — retorno pct entre N días
- `compute_sma_ratio(close, fast, slow)` — ratio SMA rápida / SMA lenta
- `compute_all_features(df)` — pipeline completo, retorna DataFrame con todas las features + targets

**Nota de implementación:** el target (`target_binary`) se calcula en `compute_all_features()` pero **no** se incluye en `FEATURE_COLUMNS`. Es responsabilidad del entrenador (Matemático) asegurarse de que esta columna no se usa como feature de entrada.

### 6.2 `data/downloader.py` — Descarga y caché de OHLCV

**Estado:** Implementado.

- Descarga datos via `yfinance` para todos los tickers del universo
- Caché local en Parquet (evita re-descargar en cada ejecución)
- Verificación de calidad: gaps, valores nulos, duplicados, precios negativos
- Soporte para el universo CSV (`data/universe_2018-01-01.csv`)

### 6.3 `agents/base_agent.py` — Interfaz abstracta de agentes

**Estado:** Implementado.

Interfaz con tres métodos que todos los agentes deben implementar:
- `fit(train_data: pd.DataFrame) -> None`
- `predict(data: pd.DataFrame) -> pd.Series`
- `is_fitted() -> bool`

### 6.4 `agents/matematico.py` — El Matemático

**Estado:** Completamente implementado (el agente más completo).

**Componentes internos:**
- `XGBClassifier` con parámetros configurables desde `config.yaml`
- `CalibratedClassifierCV` wrapeando el XGBoost (calibración Platt Scaling por defecto)
- Cross-validation interna de 5 folds para la calibración (configurable)

**Métodos públicos:**
- `fit(train_data)` — entrena XGBoost + calibra probabilidades. Mínimo 60 filas limpias.
- `predict(data)` — retorna Serie de probabilidades calibradas p ∈ [0,1]
- `calibration_report(data, n_bins)` — métricas de calibración + datos del reliability diagram:
  - Brier Score Loss (0 = perfecto)
  - Log Loss
  - `fraction_of_positives` y `mean_predicted_value` para dibujar el reliability diagram
- `feature_importances()` — importancia promedio de cada feature sobre todos los calibradores internos

**FEATURE_COLUMNS (16 features):**
```python
["close_norm", "return_1d", "return_5d", "return_20d",
 "rsi_14", "rsi_28",
 "macd", "macd_signal", "macd_hist",
 "bb_bandwidth", "bb_pct_b",
 "atr_pct",
 "volume_ratio",
 "sma_20_50_ratio", "sma_50_200_ratio"]
```

### 6.5 `agents/gestor_riesgos.py` — El Gestor de Riesgos

**Estado:** Completamente implementado (incluyendo caps por clase de activo).

**Responsabilidades:**
1. Calcular `f*` (fracción de capital) via Fractional Kelly
2. Aplicar cap por ticker según clase de activo
3. Aplicar cap total de clase en el portfolio
4. Aplicar cap total del portfolio (≤ 100%)
5. Calcular si se debe activar el stop-loss dinámico

**Método central: `compute_position_size(p, b, asset_class)`**

```python
kelly_full = p - (1 - p) / b
f_star = kelly_fraction * kelly_full   # Half-Kelly: kelly_fraction = 0.5
f_star = max(0.0, f_star)             # Si negativo → 0 (no operar)
f_star = min(f_star, cap_de_clase)    # Cap por ticker según clase
```

Casos especiales:
- `b <= 0` → retorna 0.0 (ratio indefinido)
- `f* < min_kelly_threshold` (0.005) → retorna 0.0 (ruido, no vale la pena)

**Normalización de portfolio: `normalize_portfolio_by_class(positions, ticker_classes)`**

Tres niveles en cascada:
1. **Nivel 1 — cap por ticker:** ningún ticker supera `get_position_cap(su_clase)`
2. **Nivel 2 — cap por clase:** suma de cada clase no supera `get_portfolio_cap(clase)`. Si supera, se reducen proporcionalmente los tickers de esa clase.
3. **Nivel 3 — cap total:** suma total ≤ 1.0. Si supera, se reduce todo proporcionalmente.

**Stop-loss dinámico: `should_stop_loss(current_price, entry_price, atr)`**

```
stop_level = entry_price - (N × ATR)    donde N = stop_loss_atr_multiplier = 2.0
activar si current_price <= stop_level
```

**Ratio b: `compute_gain_loss_ratio(returns, window=60)`**

Calcula `b = media(ganancias) / media(pérdidas absolutas)` sobre los últimos 60 días de la ventana de entrenamiento. Se recalcula por ticker en cada iteración walk-forward.

**Constructor alternativo: `GestorRiesgos.from_config(config)`**

Lee los caps directamente de `config.yaml` → sección `asset_classes`, cargando `max_position_pct` y `max_portfolio_pct` por clase.

### 6.6 `backtester/metrics.py` — Métricas de evaluación

**Estado:** Completamente implementado.

Funciones disponibles:
- `sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)` — Sharpe anualizado
- `max_drawdown(equity_curve)` — retorna valor negativo (ej: -0.25 = -25%)
- `calmar_ratio(returns, equity_curve, periods_per_year=252)`
- `win_rate(returns)` — % días con retorno positivo
- `total_return(equity_curve)` — retorno total como fracción
- `annualized_return(returns, periods_per_year=252)`
- `summary(equity_curve, returns)` — dict con todas las métricas
- `compare_to_baseline(system_equity, system_returns, baseline_equity, baseline_returns, baseline_name)` — comparativa con outperformance en Sharpe, retorno y drawdown

### 6.7 `backtester/engine.py` — Motor de Backtesting

**Estado:** Completamente implementado (el componente más crítico).

**Dataclasses:**
- `Position` — ticker, entry_price, entry_date, shares, capital_invested
- `TradeLog` — date, ticker, action (BUY/SELL), price, shares, capital, commission, kelly_fraction, agent_votes, reason

**Método principal: `run(prices, signals, kelly_fractions, veto_signal, atr_data, agent_votes_log)`**

Proceso por cada día T:
1. Calcular precios actuales de todos los tickers
2. Verificar stop-loss en posiciones abiertas (si `atr_data` disponible)
3. Para cada ticker con señal del Juez:
   - Si `f* <= 0` y hay posición abierta → VENTA
   - Si `f* > min_threshold` y no hay posición y sin veto → COMPRA
4. Calcular valor total del portfolio (efectivo + posiciones abiertas)
5. Registrar en `equity_curve`

**Método `_buy()`:**
- `f_star = min(f_star, max_position_pct)` — red de seguridad adicional
- `capital_to_invest = portfolio_val × f_star`
- `capital_to_invest = min(capital_to_invest, self.capital)` — no invertir más de lo que hay
- Comisión aplicada al comprar: `commission = capital_to_invest × commission_pct`
- `shares = (capital_to_invest - commission) / price`

**Método `_sell()`:**
- `sale_value = shares × price`
- `commission = sale_value × commission_pct`
- `net_proceeds = sale_value - commission`
- `self.capital += net_proceeds`

**Método `close_all_positions(fecha, current_prices)`:**
- Cierra todas las posiciones abiertas al final de cada ventana walk-forward
- Obligatorio antes de llamar a `reset()` para evitar posiciones "fantasma" entre ventanas

**Método `save_trade_logs(filename)`:**
- Guarda todas las operaciones en `logs/trades/{filename}` en formato JSONL
- Una línea por operación, JSON serializable
- Usado por El Explorador (v2) para analizar el historial

**Razones de venta registradas:**
- `"signal"` — el Juez / Kelly indica cerrar posición
- `"stop_loss"` — precio cayó más de N × ATR desde la entrada
- `"end_of_period"` — fin de ventana walk-forward, se cierra todo

### 6.8 `backtester/walk_forward.py` — Validación Walk-Forward

**Estado:** Completamente implementado (incluyendo el orquestador completo).

**`generate_windows(config)`:**
- Lee `start_date`, `holdout_start`, `walk_forward_train_years`, `walk_forward_val_years` de config
- Genera lista de `WalkForwardWindow` (dataclass con iteration, train_start, train_end, val_start, val_end)

**`WalkForwardValidator.run(agents, judge, gestor, prices, features, ticker_classes)`:**

10 pasos por ventana:
1. Cortar datos por ventana (con anti-leakage explícito: excluir val_start del train)
2. Entrenar el Matemático sobre todos los tickers concatenados
3. Calcular ratio `b` por ticker (sobre datos de entrenamiento)
4. Predecir por ticker en período de validación
5. Entrenar el Juez con datos de la iteración anterior (pass-through en iteración 1)
6. El Juez emite señal final (con veto del Conspiranoico si está activo)
7. Calcular fracciones de Kelly por (fecha, ticker) con normalización por clase
8. Extraer ATR para stop-loss dinámico
9. Ejecutar BacktestEngine para esta ventana
10. Preparar datos del Juez para la siguiente iteración

**Resultado final:**
- Retornos de todas las ventanas concatenados en cronológico
- `full_equity = (1 + full_returns).cumprod() × initial_capital`
- `metrics.summary()` sobre el período completo out-of-sample

### 6.9 `baselines/buy_and_hold.py` — Baseline Buy & Hold

**Estado:** Implementado.

- Compra todo el universo en partes iguales al inicio
- Mantiene sin tocar
- Calcula curva de capital y métricas para comparación

### 6.10 `utils/reproducibility.py` — Seeds fijas

**Estado:** Implementado.

```python
def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Llamar a `set_all_seeds(cfg["general"]["random_seed"])` como primera instrucción en cualquier script de entrenamiento.

**Por qué importa para el portfolio:** si un reclutador clona el repositorio y ejecuta el backtest, debe obtener exactamente el mismo Sharpe Ratio que aparece en la web. Si no, la credibilidad del proyecto queda en entredicho.

### 6.11 `utils/config_loader.py` — Cargador de configuración

**Estado:** Implementado.

- Carga `config.yaml` con caché (evita re-leer el archivo en cada `import`)
- Retorna el dict de configuración completo
- Todos los módulos importan su configuración a través de esta función

### 6.12 `data/universe_2018-01-01.csv` — Universo de Backtest

**Estado:** Creado.

CSV con 46 filas y columnas: `ticker`, `name`, `asset_class`, `sector`, `notes`.

Incluye:
- 40 acciones del Nasdaq 100 (selección por capitalización en enero 2018)
- 6 ETFs de diversificación (TLT, IEF, GLD, XLU, XLP, EFA)
- 8 acciones underperformers explícitas (INTC, GILD, BIIB, WBA, BIDU, PYPL, ILMN) para reducir survivorship bias

### 6.13 `data/universe_manager.py` — Gestor del Universo de Producción

**Estado:** Implementado (para Fase 9 — El Explorador).

Gestiona la separación entre universo de backtest (inmutable) y universo de producción (evolutivo):

- `ensure_live_universe_exists()` — crea `universe_live.csv` como copia del backtest si no existe
- `apply_change(action, ticker, asset_class, reason, approved_by)` — aplica ADD/REMOVE al universo live con log auditable
- `diff_universes()` — compara backtest vs live (qué se añadió, qué se retiró)
- `get_change_history()` — lee `logs/universe_changes.jsonl` completo

Cada cambio aprobado genera una línea en `logs/universe_changes.jsonl`:
```json
{"fecha": "2026-06-09 10:30:00", "accion": "REMOVE", "ticker": "CHKP", "asset_class": "equity", "razon": "f*=0 el 94% de los dias", "aprobado_por": "human"}
```

### 6.14 `judge/judge_v1.py` — El Juez Central v1

**Estado:** Completamente implementado (Junio 2026).

**Dos modos de operación:**
- **Pass-through** (`is_pass_through=True`, por defecto): no requiere `fit()`. Pasa directamente las señales del Matemático. Válido para Fase 3.
- **Meta-modelo** (tras `fit()`): clasificador calibrado (Logistic o XGBoost) entrenado sobre las señales combinadas de varios agentes. Válido desde Fase 4.

**Métodos públicos:**
- `fit(agent_predictions, targets)` — entrena el meta-modelo. Requiere señales de al menos 2 agentes y mínimo 60 filas limpias. Tras `fit()`, el modo pasa automáticamente de pass-through a meta-modelo.
- `predict(agent_predictions, veto_signal)` — soporta dos formatos:
  - **Multi-ticker (Fase 3):** columnas = tickers, retorna mismo DataFrame aplicando veto
  - **Multi-agente (Fase 4+):** columnas = agentes, retorna DataFrame con columna `prob_up` calibrada
- `predict_single_agent(signal, veto_signal)` — conveniencia para Fase 3: pasa la Serie del Matemático directamente
- `evaluation_report(agent_predictions, targets)` — Brier Score, Log Loss y nº de muestras (solo en modo meta-modelo)
- `agent_weights()` — coeficientes de la regresión logística o feature importances del XGBoost por agente

**Gestión del veto del Conspiranoico:**
El método `_apply_veto()` pone a 0 todas las señales en las fechas donde `veto_signal == 1`. Es transparente: funciona igual en modo pass-through que en modo meta-modelo.

**Anti-leakage del Juez:**
En la Iteración 1 del walk-forward, el Juez no tiene datos previos → permanece en pass-through. En Iteraciones 2+, se entrena con las predicciones de validación de la iteración anterior (nunca con datos de la ventana actual).

### 6.15 `baselines/sma_crossover.py` — Baseline SMA Crossover

**Estado:** Completamente implementado (Junio 2026).

**Diseño:** equal weight independiente por ticker. Cada ticker gestiona su propio sub-portfolio con 1/N del capital inicial. Esto evita rebalanceos constantes entre tickers y mantiene la comparabilidad con Buy & Hold.

**Función pública `run(prices, start_date, end_date, config, fast_window, slow_window)`:**
- Construye DataFrame de cierres alineados para el período de evaluación
- Para cada ticker llama a `_simulate_ticker()` con el capital asignado
- Suma todos los sub-portfolios para obtener la curva total
- Retorna `(equity_curve, daily_returns)`

**Función interna `_simulate_ticker(close_series, capital, commission_pct, fast_window, slow_window)`:**
- Calcula `sma_fast` y `sma_slow` sobre la serie completa (para que las SMAs arranquen con datos suficientes antes del período de evaluación)
- Señal = `sma_fast > sma_slow`
- Cruce alcista → COMPRA con todo el efectivo disponible, aplicando comisión
- Cruce bajista → VENTA de todas las acciones, aplicando comisión
- Durante los primeros `slow_window` días (sin señal), el capital permanece en efectivo sin comisiones

**Nota sobre lookback:** los primeros `slow_window` días (por defecto 50) no generan señal porque las SMAs no tienen datos suficientes. Esto es correcto y esperado.

### 6.16 `run.py` — Script de ejecución del pipeline completo

**Estado:** Actualizado (Junio 2026, Fase 4).

Script de entrada para ejecutar el pipeline MAS completo. Soporta dos modos:

- **Fase 3** (`python run.py`): solo Matemático
- **Fase 4** (`python run.py --analista`): Matemático + Analista

Flags disponibles:
- `--analista` — incluir El Analista (requiere noticias)
- `--force-download` — re-descargar OHLCV y noticias
- `--force-sentiment` — re-computar sentimiento FinBERT
- `--no-baselines` — saltar comparativa
- `--debug` — logging verbose

Cuando se usa `--analista`, se añade la comparativa Fase 4: ¿Matemático+Analista supera a Matemático solo? (cargando el último experimento de Fase 3 si existe).

Cada ejecución genera un directorio en `experiments/{nombre}_{timestamp}/` con:
- `config.yaml` — copia exacta de la configuración usada
- `results.json` — métricas por ventana walk-forward + métricas agregadas + métricas de baselines

### 6.17 `agents/analista.py` — El Analista (Fase 4)

**Estado:** Completamente implementado (Junio 2026).

**Dos fases de uso:**

1. **Precomputo** (una vez antes del walk-forward):
   - `precompute_sentiment(news_data, trading_dates)` puntúa todos los titulares con FinBERT
   - Los scores se agregan por ticker y día de mercado (media por defecto)
   - El resultado se cachea en `data/cache/sentiment/{ticker}_sentiment.parquet`

2. **Walk-forward** (por cada ventana):
   - `fit(train_data)` calibra el score de sentimiento contra el target real con `CalibratedClassifierCV(LogisticRegression, method='sigmoid')`
   - `predict(data)` retorna probabilidad calibrada p ∈ [0,1]
   - Filas sin dato de sentimiento reciben NaN (gestionado por el fallback al Matemático)

**FinBERT scoring:**
```python
if label == "positive":
    score = 0.5 + raw_score / 2   # Mapea a [0.5, 1.0]
elif label == "negative":
    score = 0.5 - raw_score / 2   # Mapea a [0.0, 0.5]
else:
    score = 0.5                    # Neutral
```

**Métodos públicos:**
- `precompute_sentiment(news_data, trading_dates, cache_dir, force)` — FinBERT batch + agregación + caché
- `merge_sentiment_into_features(features, sentiment)` (static) — añade columna `sentiment_raw` a cada ticker
- `fit(train_data)` — entrena calibrador sobre `sentiment_raw` vs `target_binary`
- `predict(data)` — retorna p ∈ [0,1] calibrada
- `calibration_report(data)` — Brier Score, Log Loss, datos del reliability diagram
- `sentiment_coverage(features)` — cobertura de sentimiento por ticker (diagnóstico)

### 6.18 `data/news.py` — Pipeline de noticias financieras (Fase 4)

**Estado:** Completamente implementado (Junio 2026).

**Fuente de datos:** Alpaca News API (gratuita con cuenta paper trading).

**Configuración requerida:**
- Variables de entorno `ALPACA_API_KEY` y `ALPACA_SECRET_KEY`
- Obtener gratis en https://alpaca.markets (crear cuenta paper trading)

**Funciones públicas:**
- `download_all_news(tickers, config, force)` — descarga y cachea noticias en `data/cache/news/{ticker}.parquet`
- `load_cached_news(config)` — carga noticias desde caché sin descargar
- `assign_to_trading_days(news_df, trading_dates)` — asigna cada noticia al día de mercado correspondiente
- `get_trading_dates(start, end)` — obtiene calendario NYSE vía `pandas_market_calendars`

**Sincronización temporal:**
- Noticias durante horario de mercado → ese día
- Noticias después de las 16:00 ET → siguiente día hábil
- Noticias de fin de semana/festivos → siguiente día hábil

**Degradación grácil:** si no hay API keys configuradas, el sistema advierte y continúa sin El Analista.

### 6.19 Walk-Forward actualizado para multi-agente (Fase 4)

**Estado:** Actualizado (Junio 2026).

**Cambios respecto a Fase 3:**

El `WalkForwardValidator.run()` ahora soporta múltiples agentes predictores:

1. **Step 2b (nuevo):** entrena El Analista sobre todos los tickers concatenados. Si falla (datos insuficientes), continúa sin él para esa ventana.

2. **Step 4b (nuevo):** predice con El Analista por ticker. Registra cuántos tickers tienen cobertura de sentimiento.

3. **Step 6 (modificado):** la función `_combine_agent_signals()` gestiona dos modos:
   - **Pass-through** (Iter 1 o sin Analista): pasa las señales del Matemático directamente al Juez
   - **Meta-modelo** (Iter 2+ con Analista): combina señales por ticker a través del meta-modelo del Juez, con fallback al Matemático para tickers sin cobertura de noticias

4. **Step 10 (modificado):** los datos para entrenar al Juez en la siguiente iteración incluyen columnas `["matematico", "analista"]` en vez de solo `["matematico"]`.

**Compatibilidad hacia atrás:** si `agents` solo contiene `"matematico"`, el walk-forward funciona exactamente como en Fase 3.

---

## 7. Implementación: Stubs (Pendientes de Completar)

### 7.1 `agents/analista.py` — El Analista (Fase 4)

**Estado:** ✅ Completamente implementado (Junio 2026).

Ver sección 6.17 para la documentación detallada.

### 7.2 `agents/cazador.py` — El Cazador (Fase 5)

**Estado:** ✅ Implementado (Junio 2026).

**Arquitectura:**
- No usa ML. Reglas condicionales sobre transacciones de insiders de la SEC EDGAR (Form 4).
- `data/insiders.py` — módulo de datos separado (patrón idéntico a `data/news.py`):
  - API oficial SEC EDGAR: ticker → CIK, submissions JSON, Form 4 XML parsing con `xml.etree.ElementTree` (stdlib)
  - `time.sleep(0.12)` entre requests (~8 req/s, bajo el límite de 10 req/s de la SEC)
  - Caché Parquet por ticker en `data/cache/insiders/`
  - Solo descarga tickers con `asset_class == "equity"` (los ETFs no tienen Form 4)
- `Cazador.precompute_insider_signals(tickers)` — precarga caché antes del walk-forward
- `Cazador.predict_ticker(data, ticker)` — retorna Serie binaria 0/1 por fecha

**Regla de señal:**
- Si un CEO/CFO/Director/President/COO/CTO vende más del 20% de sus acciones en los últimos 30 días → alerta = 1

**Anti-leakage (obligatorio):**
- La señal usa `filing_date` (cuando el Form 4 se declara públicamente) + `form4_lag_days=2` días hábiles de shift sobre calendario NYSE.
- Una transacción en día T solo es visible en el backtest a partir de T + 2 días hábiles.
- Implementado en `data.insiders.build_alert_series()` y `_add_business_days()`.

**Integración en el pipeline:**
- Cazador entra como columna binaria `"cazador"` (0/1) en el meta-modelo del Juez junto con `"matematico"` y `"analista"`.
- `walk_forward.py` paso 4c: `cazador.predict_ticker(v_sl, ticker)` por ticker.
- `walk_forward.py` paso 10: columna `"cazador"` incluida en `prev_judge_inputs` para entrenamiento del Juez en iter N+1.
- `agent_votes_log`: los trade logs JSONL ahora incluyen `{"matematico": 0.57, "cazador": 0}` por operación.

**Nota operativa (actualizada):**
- **Fuente migrada a SEC EDGAR** (junio 2026): OpenInsider bloqueaba conexiones. La nueva implementación usa la API oficial de la SEC (`data.sec.gov`) con parsing de Form 4 XML. Requiere `cazador.sec_user_agent` en `config.yaml` (nombre + email, política fair-access SEC). Rate limit: 10 req/s; el módulo usa 0.12s de pausa.
- La descarga es one-time con caché Parquet. Si falla, el Cazador retorna silencio (serie de ceros) y el pipeline continúa. El meta-modelo ignorará el feature si siempre es cero.
- Los códigos de transacción cambian de etiquetas OpenInsider (`"S - Sale"`) a códigos SEC (`"S"`, `"S-"`). El DataFrame de caché anterior es incompatible → borrar `data/cache/insiders/` y re-descargar.

### 7.3 `agents/conspiranoico.py` — El Conspiranoico (Fase 6)

**Estado:** Stub con estructura definida.

**Diseño:**
- Isolation Forest para detección de anomalías de régimen
- Features de régimen (todas calculables con datos diarios de yfinance):
  - Volatilidad realizada rolling (std de retornos a 5, 20 y 60 días)
  - VIX descargado como ticker `^VIX`
  - Correlación rolling entre acciones del universo (alta correlación = movimiento en manada = pánico)
  - Volumen relativo vs media de 20 días
  - Amplitud del mercado: % de acciones del universo que suben ese día
- Output: señal de veto binaria (1 = régimen anómalo, no operar)

**Validación crítica:** comprobar que el veto se activa en períodos de crisis documentados:
- Marzo 2020 (crash COVID)
- 2022 (caídas por subida de tipos)

**Nota de nomenclatura:** el agente se llamaba originalmente "El Esquizofrénico" pero se renombró a "El Conspiranoico" por ser más presentable para el portfolio. El cambio también incluyó actualizar las features para eliminar las que requieren datos intraday (bid-ask spread no disponible con datos diarios de yfinance).

### 7.4 `judge/judge_v1.py` — El Juez Central v1

**Estado:** ✅ Completamente implementado (Junio 2026). Ver sección 6.14.

### 7.5 `backtester/engine.py` (modo perfiles) — Fase 8

El `BacktestEngine` actual opera en modo swing (señales diarias). En la Fase 8 se necesita extender para:
- **Long-term:** saltar señales diarias según `rebalancing_days` del perfil
- **Day simulated:** usar Open del día T como precio de entrada y Close del mismo día T como precio de salida

### 7.6 `baselines/sma_crossover.py` — Baseline SMA Crossover

**Estado:** ✅ Completamente implementado (Junio 2026). Ver sección 6.15.

### 7.7 `dashboard/app.py` — Dashboard Streamlit (Fase 7)

**Estado:** Stub con estructura de secciones definida.

Secciones planificadas:
- Curva de capital vs baselines
- Estado actual de la cartera
- "Veredicto del Consejo": votos de cada agente en tiempo real
- Log auditable de operaciones
- Pestaña comparativa de perfiles (Fase 8)
- Pestaña "El Explorador" con recomendaciones pendientes (Fase 9)

### 7.8 `agents/explorador.py` — El Explorador (Fase 9)

**Estado:** Stub con estructura completa definida.

Métodos planificados:
- `load_trade_history()` — parsea `logs/trades/*.jsonl`
- `find_removal_candidates(trade_history)` — detecta activos con f*=0 crónico o accuracy < 50%
- `evaluate_candidate(ticker, asset_class, current_prices)` — evalúa candidato externo con yfinance
- `generate_recommendations(trade_history, current_prices, external_candidates)` — lista para HITL
- `save_recommendations(recommendations)` — guarda en `logs/explorador_recommendations.jsonl`

---

## 8. Sistema de Configuración

### 8.1 Estructura de `config.yaml`

El archivo centraliza **todos** los hiperparámetros y parámetros del proyecto. Nunca hardcodear valores en el código.

```yaml
universe:
  tickers_file: "data/universe_2018-01-01.csv"
  max_tickers: 46

data:
  start_date: "2018-01-01"
  end_date: "2025-12-31"
  holdout_start: "2025-01-01"
  cache_dir: "data/cache"

backtester:
  commission_pct: 0.0008
  initial_capital: 10000.0
  walk_forward_train_years: 3
  walk_forward_val_years: 1

risk_manager:
  kelly_fraction: 0.5
  max_position_pct: 0.15
  stop_loss_atr_multiplier: 2.0
  min_kelly_threshold: 0.005

asset_classes:
  equity:   {max_position_pct: 0.15, max_portfolio_pct: 0.80}
  bond:     {max_position_pct: 0.20, max_portfolio_pct: 0.30}
  gold:     {max_position_pct: 0.10, max_portfolio_pct: 0.10}
  defensive_equity: {max_position_pct: 0.15, max_portfolio_pct: 0.20}
  international_equity: {max_position_pct: 0.10, max_portfolio_pct: 0.10}

matematico:
  n_estimators: 300
  max_depth: 5
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  calibration_method: "sigmoid"
  random_state: 42

analista:
  model_name: "ProsusAI/finbert"
  batch_size: 32
  max_length: 512
  aggregation: "mean"
  calibration_method: "sigmoid"
  random_state: 42

conspiranoico:
  n_estimators: 100
  contamination: 0.05
  vix_ticker: "^VIX"
  volatility_windows: [5, 20, 60]
  correlation_window: 20
  volume_ratio_window: 20
  random_state: 42

cazador:
  sell_threshold_pct: 0.20
  lookback_days: 30
  form4_lag_days: 2

judge_v1:
  meta_model: "logistic"
  calibration_method: "sigmoid"
  random_state: 42

explorador:
  neutral_signal_days_threshold: 0.90
  min_prediction_accuracy: 0.50
  max_correlation_with_existing: 0.85
  min_data_years: 5
  min_avg_daily_volume: 500000
  review_period_days: 90

profile:
  active: "swing"
  available: ["profiles/swing.yaml", "profiles/long_term.yaml", "profiles/day_simulated.yaml"]

general:
  random_seed: 42
  log_level: "INFO"
  log_dir: "logs"
  experiments_dir: "experiments"
```

### 8.2 Perfiles de trading (`profiles/*.yaml`)

Los perfiles no reemplazan `config.yaml`; lo complementan. Un perfil define overrides de parámetros específicos.

**`profiles/swing.yaml`** — Sin overrides (es el config base).

**`profiles/long_term.yaml`** — Overrides principales:
```yaml
profile:
  signal_frequency: "monthly"
  rebalancing_days: 21

matematico:
  features_override:
    sma_windows: [50, 200]
    rsi_window: 28
    macd_windows: [26, 52, 9]

risk_manager:
  max_position_pct: 0.20
  stop_loss_atr_mult: 5.0
```

**`profiles/day_simulated.yaml`** — Overrides con disclaimer obligatorio:
```yaml
profile:
  disclaimer: "IMPORTANTE: Este perfil usa Open/Close diarios como proxy de day trading..."
  execution: "buy_at_open_sell_at_close"
  max_holding_days: 1

backtester:
  commission_pct: 0.0020
  slippage_proxy_pct: 0.0010

risk_manager:
  kelly_fraction: 0.25
  max_position_pct: 0.10
```

---

## 9. Tests Implementados

### 9.1 `tests/test_kelly.py` — Tests del Gestor de Riesgos

**Estado:** Completamente implementado con 4 clases de tests.

**`TestComputePositionSize` (8 tests):**
- `test_negative_kelly_returns_zero` — f* negativo nunca genera compra
- `test_cap_at_max_position` — f* nunca supera el cap del 15%
- `test_half_kelly_applied` — verifica ρ=0.5 matemáticamente (p=0.6, b=1.0 → f*=0.1)
- `test_zero_b_returns_zero` — b=0 no causa ZeroDivisionError
- `test_negative_b_returns_zero` — b negativo no tiene sentido financiero
- `test_p_exactly_zero` — p=0 → f*=0
- `test_p_exactly_one` — p=1 → f* capado al máximo de clase
- `test_result_in_valid_range` — rango [0, max_position_pct] para varios (p, b)

**`TestNormalizePortfolio` (4 tests):**
- `test_no_normalization_needed` — si suma ≤ 1.0 no se modifica
- `test_normalization_when_over_100` — normalización reduce suma a ≤ 1.0
- `test_cap_maintained_after_normalization` — ningún ticker supera el cap después
- `test_empty_portfolio` — portfolio vacío no causa errores

**`TestStopLoss` (3 tests):**
- `test_stop_loss_triggered` — entry=100, ATR=2, mult=2.0, precio=95 → activar (stop=96)
- `test_stop_loss_not_triggered` — precio=97 > stop=96 → no activar
- `test_stop_loss_at_exact_level` — precio=96 = stop → activar

**`TestAssetClassCaps` (9 tests):**
- `test_bond_cap_higher_than_equity` — bonos 20% > acciones 15%
- `test_gold_cap_lower_than_equity` — oro 10% < acciones 15%
- `test_bond_position_respects_bond_cap` — f* de bono ≤ 20%
- `test_gold_position_respects_gold_cap` — f* de oro ≤ 10%
- `test_unknown_class_uses_default_cap` — clase desconocida usa cap por defecto
- `test_portfolio_class_cap_bonds_respected` — suma bonos ≤ 30%
- `test_portfolio_class_cap_equity_respected` — suma equity ≤ 80%
- `test_mixed_portfolio_total_under_100` — portfolio mixto total ≤ 100%
- `test_existing_tests_still_pass_without_classes` — backward compatibility

### 9.2 `tests/test_backtester.py` — Tests del Backtester

**Estado:** Implementado (Junio 2026).

Clases de tests y checks obligatorios:
- **`TestAntiLeakage`:** verifica que `target_binary` y `target_return_1d` no están en `FEATURE_COLUMNS`; verifica ausencia de `shift(-N)` en el código de features (excepto en el target)
- **`TestCommissions`:** comprar y vender el mismo día resulta en pérdida; 100 operaciones round-trip cuestan más del 10% del capital
- **`TestReproducibility`:** misma seed → mismo resultado exacto; seeds distintas → resultados distintos
- **`TestMetrics`:** casos conocidos para Sharpe, MaxDrawdown y WinRate (verificación de la implementación matemática)

### 9.3 `tests/test_data_integrity.py` — Tests de Calidad de Datos

**Estado:** Implementado (Junio 2026). Se saltan automáticamente si `data/cache/` no existe.

Tests por cada ticker en caché (parametrizado con `@pytest.fixture`):
- `test_no_nan_in_close` — sin NaN en precio de cierre
- `test_no_duplicate_dates` — sin fechas duplicadas en el índice
- `test_close_prices_positive` — todos los cierres > 0
- `test_date_index_is_monotonic` — índice estrictamente creciente
- `test_minimum_rows` — al menos 1000 filas (≈ 4 años de datos diarios)

---

## 10. Estructura del Repositorio

### 10.1 Árbol de directorios actual

```
c:\Proyectos\trading\
│
├── config.yaml                          # Fuente única de verdad — todos los parámetros
├── requirements.txt                     # Dependencias con versiones mínimas
├── run.py                               # Script principal de ejecución del pipeline ✅
├── .gitignore
├── README.md
├── Plan.md                              # Plan técnico del proyecto (documento vivo)
├── Memoria.md                           # Este documento
│
├── data/
│   ├── universe_2018-01-01.csv          # 40 acciones + 6 ETFs — NUNCA MODIFICAR
│   ├── universe_live.csv                # Universo de producción (se crea al iniciar paper trading)
│   ├── raw/                             # OHLCV descargados sin tocar
│   ├── processed/                       # Features calculadas
│   ├── cache/                           # Caché Parquet de yfinance
│   ├── downloader.py                    # Descarga + caché + control de calidad
│   ├── features.py                      # Pipeline de features técnicas ✅
│   ├── news.py                          # Pipeline de noticias (Alpaca API + caché) ✅
│   └── universe_manager.py             # Gestión del universo de producción ✅
│
├── agents/
│   ├── base_agent.py                    # Interfaz abstracta (fit, predict, is_fitted) ✅
│   ├── matematico.py                    # XGBoost + Platt Scaling ✅
│   ├── analista.py                      # FinBERT + calibración ✅
│   ├── cazador.py                       # SEC EDGAR + reglas ✅
│   ├── conspiranoico.py                 # Isolation Forest + HMM 🔲
│   ├── gestor_riesgos.py                # Fractional Kelly + caps + stop-loss ✅
│   └── explorador.py                    # Gestión dinámica del universo v2 🔲
│
├── judge/
│   ├── __init__.py                      # Necesario para que Python reconozca el paquete ✅
│   ├── judge_v1.py                      # Meta-modelo estático (Logistic/XGBoost) ✅
│   └── judge_v2_rl.py                   # RL/PPO via stable-baselines3 (opcional v2) 🔲
│
├── backtester/
│   ├── __init__.py
│   ├── engine.py                        # Motor día a día + comisiones + logs ✅
│   ├── metrics.py                       # Sharpe, MaxDD, Calmar, WinRate ✅
│   └── walk_forward.py                  # Orquestador walk-forward completo ✅
│
├── baselines/
│   ├── buy_and_hold.py                  # ✅
│   └── sma_crossover.py                 # ✅
│
├── profiles/
│   ├── swing.yaml                       # Perfil por defecto (señales diarias)
│   ├── long_term.yaml                   # Rebalanceo mensual, ventanas largas
│   └── day_simulated.yaml               # Open→Close, solo académico
│
├── tests/
│   ├── test_kelly.py                    # 24 tests sobre GestorRiesgos ✅
│   ├── test_backtester.py               # Anti-leakage, comisiones, reproducibilidad ✅
│   └── test_data_integrity.py           # Calidad de datos ✅
│
├── dashboard/
│   └── app.py                           # Streamlit 🔲
│
├── utils/
│   ├── config_loader.py                 # Carga y caché de config.yaml ✅
│   └── reproducibility.py              # Seeds fijas Python/NumPy/PyTorch ✅
│
├── logs/
│   ├── trades/                          # JSONL por operación (generados por BacktestEngine)
│   ├── explorador_recommendations.jsonl # Recomendaciones pendientes de HITL (v2)
│   └── universe_changes.jsonl           # Historial auditable de cambios al universo (v2)
│
├── notebooks/
│   └── exploration/                     # EDA y experimentos
│
└── experiments/                         # Config + resultados por experimento
```

### 10.2 Dependencias principales (`requirements.txt`)

| Categoría | Biblioteca | Uso |
|---|---|---|
| Datos | `yfinance`, `pandas`, `numpy`, `pyarrow`, `pandas-market-calendars` | OHLCV, manipulación, Parquet, calendario NYSE |
| ML | `scikit-learn`, `xgboost`, `hmmlearn`, `scipy` | Modelos, calibración, HMM, estadística |
| NLP | `transformers`, `torch` | FinBERT |
| SEC EDGAR API | `requests` | Form 4 (submissions JSON + XML parsing con stdlib `xml.etree`) |
| Dashboard | `streamlit`, `plotly` | Visualización |
| Config | `PyYAML`, `tqdm`, `python-dotenv` | Configuración, progreso, variables de entorno |
| Testing | `pytest`, `pytest-cov` | Tests y cobertura |
| Visualización | `matplotlib`, `seaborn`, `jupyter` | Reliability diagrams, EDA |
| RL (opcional v2) | `stable-baselines3`, `gymnasium` | Juez v2 con PPO |

---

## 11. Expansión v2: Perfiles de Trading y El Explorador

### 11.1 Los tres perfiles de trading

Los perfiles no requieren cambiar los agentes; solo cambian los parámetros con los que operan y cómo el BacktestEngine ejecuta las señales.

| Perfil | Señal | Ejecución | Estado |
|---|---|---|---|
| **Swing** | Diaria (cierre T) | Compra a cierre T, vende cuando Kelly indica | Perfil actual |
| **Long-term** | Mensual (rebalanceo) | Señales diarias ignoradas hasta que toca rebalancear | Fase 8 |
| **Day simulated** | Diaria (cierre T-1) | Compra a apertura T, vende a cierre T | Fase 8 |

**Limitación crítica del perfil Day Simulated:**

Los datos diarios de yfinance solo tienen precios de apertura y cierre. El day trading real requiere:
- Datos de minutos u horas
- Modelado del spread bid-ask
- Modelado del slippage al ejecutar órdenes grandes

Los resultados del perfil Day Simulated son **optimistas** respecto a un sistema real de day trading. Su único valor es académico: entender cómo afecta el horizonte temporal al mismo conjunto de agentes. Esta limitación se documenta explícitamente en el dashboard y en la web de portfolio.

El perfil Day Simulated usa comisiones más altas (0.20%) y un `slippage_proxy_pct` adicional (0.10%) para al menos penalizar parcialmente la ausencia de modelado de slippage real.

### 11.2 El Explorador — Diseño detallado

#### Separación de universos (fundamental)

```
data/universe_2018-01-01.csv → INMUTABLE — backtest histórico
data/universe_live.csv        → EVOLUTIVO — paper trading con HITL
logs/universe_changes.jsonl   → LOG AUDITABLE — cada cambio registrado
```

El Explorador **nunca** toca el universo de backtest. Modificarlo invalidaría los resultados históricos.

#### Flujo de trabajo del Explorador

```
Cada 90 días (configurable):
    ┌─────────────────────────────────┐
    │   1. Analizar trade history     │
    │      logs/trades/*.jsonl        │
    │                                 │
    │   Para cada ticker:             │
    │   - % días con f*=0            │
    │   - accuracy del Matemático     │
    │                                 │
    │   → Candidatos a RETIRAR        │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │   2. Evaluar candidatos nuevos  │
    │      (solo si el usuario los    │
    │      sugiere explícitamente)    │
    │                                 │
    │   Para cada candidato:          │
    │   - Descargar datos (yfinance)  │
    │   - Calcular correlación        │
    │   - Verificar liquidez          │
    │   - Verificar años disponibles  │
    │                                 │
    │   → Candidatos a AÑADIR         │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │   3. Guardar recomendaciones    │
    │      logs/explorador_recs.jsonl │
    │      → aparecen en dashboard    │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │   4. REVISIÓN HUMANA            │
    │   Aprobar / Rechazar            │
    │   (botones en el dashboard)     │
    └────────────────┬────────────────┘
                     │ Solo si aprueba
    ┌────────────────▼────────────────┐
    │   5. UniverseManager.apply()    │
    │   Actualiza universe_live.csv   │
    │   Registra en universe_changes  │
    └─────────────────────────────────┘
```

#### Criterios para RETIRAR un activo

El Explorador propone retirar un activo del universo de producción si:
1. El Gestor de Riesgos asigna f*=0 el **90%+** de los días de trading → sin señal útil
2. La accuracy del Matemático en ese ticker es **< 50%** → peor que azar

#### Criterios para AÑADIR un activo

El Explorador evalúa candidatos externos si se le proporciona una lista:
1. Correlación de retornos < 0.10 con la media de la cartera → aporta diversificación real
2. Al menos 5 años de historia disponible en yfinance
3. Volumen medio diario > 500.000 (liquidez mínima)
4. No redundante con activos ya presentes (correlación < 0.85 con todos)

#### Por qué El Explorador no puede añadir activos usando trades propios

Esta es una distinción crucial: El Explorador solo puede analizar trades de activos que **ya operamos**. Para un activo que nunca ha estado en el universo, por definición no tenemos trades ni predicciones del Matemático sobre él.

- Para **RETIRAR**: usa trades propios (datos disponibles)
- Para **AÑADIR**: usa datos externos de mercado via yfinance (es la única fuente posible)

---

## 12. Trampas Técnicas y Cómo Se Evitan

### 12.1 Data leakage — Error nº1

**El problema:** el modelo accede accidentalmente a datos del futuro durante el entrenamiento o la predicción.

Ejemplos concretos:
- Calcular la media móvil de 20 días **incluyendo el día actual** para decidir una operación de ese mismo día (el precio de cierre del día actual no se conoce hasta el cierre)
- Normalizar los datos usando la media/std de **todo el dataset** antes de dividir train/test (la normalización "filtra" el futuro hacia el pasado)
- Usar noticias publicadas a las 18:00 para decidir una operación de apertura del mismo día

**Cómo se evita en este proyecto:**
- `compute_all_features()` usa exclusivamente `.rolling()`, `.ewm()` y `.shift()` hacia atrás (nunca `.shift(-N)` excepto para el target)
- El target (`shift(-1)`) **nunca** aparece en `FEATURE_COLUMNS`
- El walk-forward en `WalkForwardValidator.run()` tiene un bloque explícito de anti-leakage:
  ```python
  if not t_sl.empty and str(t_sl.index[-1].date()) >= window.val_start:
      t_sl = t_sl.iloc[:-1]  # excluir la fecha de inicio de validación del train
  ```
- El Form 4 (Cazador) aplica un lag de 2 días hábiles antes de usar la información

### 12.2 Survivorship bias

**El problema:** elegir solo empresas que existen hoy y han tenido buenos resultados (Apple, Nvidia...) es trampa: el mercado de 2018 incluía muchas empresas que luego desaparecieron o cayeron.

**Cómo se mitiga:**
- El universo se congela a **1 de enero de 2018** (no se pueden elegir ganadores a posteriori)
- Se incluyen explícitamente 8 activos con rendimiento mediocre: INTC, GILD, BIIB, WBA, BIDU, PYPL, ILMN

**Nota:** la mitigación es parcial. Para un estudio académico riguroso haría falta un universo con delisting histórico (activos que fueron eliminados del Nasdaq 100 entre 2018 y 2024). Esto se documenta como sesgo conocido.

### 12.3 Overfitting temporal (data snooping silencioso)

**El problema:** si se prueban 50 variantes del modelo y se elige la que da mejor Sharpe en el período 2021-2024, ese período ya está contaminado.

**Cómo se evita:**
- Walk-forward obligatorio (los datos de validación de cada ventana no se ven en el entrenamiento de esa ventana)
- Holdout de 2025 en adelante, no tocado hasta que el sistema esté completamente terminado

### 12.4 Costes de transacción ignorados

**El problema:** sin comisiones, una estrategia que opera 20 veces por semana puede parecer rentable y ser un desastre neto.

**Cálculo:**
- 20 operaciones semanales × 0,08% × 2 (ida y vuelta) = 3,2% de costes semanales
- Ningún edge de ML aguanta eso

**Cómo se evita:** la simulación de comisiones en `BacktestEngine._buy()` y `._sell()` es **obligatoria** y siempre activa. No hay modo sin comisiones.

### 12.5 Calibración incorrecta de probabilidades con Kelly

**El problema:** Kelly asume que `p` es una probabilidad real. Un XGBoost no calibrado puede dar scores que no son probabilidades: el modelo puede decir 0.8 cuando la frecuencia real de subidas en esa situación es solo 0.55.

**Cómo se evita:**
- `CalibratedClassifierCV` obligatorio en el Matemático y el Analista
- `calibration_report()` en el Matemático genera datos del reliability diagram
- Si la curva se desvía de la diagonal → probar isotonic en vez de sigmoid

### 12.6 Sincronización temporal de señales

**El problema:** el Matemático genera señales de lunes a viernes. El Analista genera noticias 7 días a la semana.

**Regla implementada:**
- Las noticias publicadas fuera de horario de mercado se **acumulan** hasta la siguiente apertura
- El backtester solo toma decisiones en días hábiles de mercado (NYSE calendar via `pandas_market_calendars`)

### 12.7 Ventana del Juez en la primera iteración walk-forward

**El problema:** en la primera iteración walk-forward, el Juez no tiene datos previos de agentes para entrenarse (es la primera vez).

**Solución implementada en `WalkForwardValidator`:**
- Iteración 1: el Juez actúa en modo pass-through (pasa directamente la señal del Matemático)
- Iteraciones 2+: el Juez se entrena con las predicciones de validación de la iteración anterior
- Si `judge.fit()` lanza `ValueError` (ej: menos de 2 agentes activos), se ignora y permanece en pass-through

### 12.8 Nombres de columnas de yfinance — "Close" vs "close"

**El problema:** `yfinance` devuelve columnas con la primera letra en mayúscula: `"Close"`, `"Open"`, `"High"`, `"Low"`, `"Volume"`. Si el código accede a `df["close"]` (minúscula), lanza `KeyError` en tiempo de ejecución.

**Cómo ocurrió:** `backtester/engine.py` accedía a `prices[ticker].loc[fecha, "close"]`. El error no se detecta en los tests unitarios porque estos usan datos sintéticos con columnas propias; solo aparece en el primer run real con datos de yfinance.

**Corrección aplicada:** cambiado a `"Close"` (mayúscula) en `engine.py`. Convención de todo el proyecto: usar siempre las columnas con mayúscula inicial tal como las devuelve yfinance.

### 12.9 El último día del `target_binary` era 0 en vez de NaN

**El problema:** el cálculo original `(close.pct_change(1).shift(-1) > 0).astype(float)` convierte el NaN del último día en `0.0`, porque en pandas `(NaN > 0)` evalúa a `False` y `.astype(float)` convierte `False` a `0.0`.

**Consecuencia:** el Matemático entrenaba con el último día de cada ticker etiquetado como "bajada" cuando no hay etiqueta disponible. Introduce ruido de entrenamiento y es un bug silencioso difícil de detectar sin revisar el código con atención.

**Corrección aplicada en `data/features.py`:**
```python
features["target_binary"] = np.where(
    features["target_return_1d"].isna(),
    np.nan,
    (features["target_return_1d"] > 0).astype(float),
)
```

### 12.10 Windows y la codificación cp1252 — caracteres Unicode en logs

**El problema:** Windows usa por defecto la codificación `cp1252` en la consola de PowerShell. El carácter `→` (`\u2192`) no existe en `cp1252`, lo que causa `UnicodeEncodeError` al hacer `print()` o `logger.info()` con ese carácter. El crash ocurre incluso aunque el pipeline haya terminado correctamente, porque el error se lanza al imprimir la tabla de resultados final.

**Cómo ocurrió:** varios mensajes en `run.py`, `walk_forward.py` y `downloader.py` usaban `→` como separador visual (ej: `"train [2018-01-01 → 2021-01-01]"`). En Linux/macOS (UTF-8 por defecto) funciona sin problema.

**Corrección aplicada:** reemplazado `→` por `->` en todos los archivos `.py` del proyecto. La alternativa `$env:PYTHONIOENCODING='utf-8'` funciona en PowerShell, pero el fix en el código fuente es más robusto y portátil.

### 12.11 Parámetro deprecado `use_label_encoder` en XGBoost

**El problema:** versiones recientes de XGBoost eliminaron el parámetro `use_label_encoder`. Si se pasa igualmente (aunque sea `False`), XGBoost lanza un `UserWarning` por cada estimador interno del `CalibratedClassifierCV`, contaminando el log con decenas de advertencias por ventana walk-forward.

**Corrección aplicada en `agents/matematico.py`:** eliminado el parámetro `"use_label_encoder": False` del diccionario `_base_params`. La codificación de labels es automática en versiones modernas de XGBoost y no requiere configuración explícita.

### 12.12 Cobertura de noticias desigual entre tickers — El Analista (Fase 4)

**El problema:** la Alpaca News API no tiene cobertura uniforme para todos los tickers. Empresas grandes (AAPL, MSFT, GOOG) tienen cientos de artículos al día; empresas medianas o ETFs pueden tener días sin ninguna noticia.

**Consecuencia:** si no se maneja, el Analista tendría NaN para muchos tickers/días, y el Juez meta-modelo no podría combinar las señales.

**Solución implementada:**
1. Los días sin noticias reciben `sentiment_raw = NaN` (no 0.5, para distinguir "sin datos" de "sentimiento neutral")
2. El Analista retorna `NaN` para esas filas en `predict()`
3. El Juez meta-modelo detecta NaN en la columna `analista` y esas filas producen NaN en `prob_up`
4. `_combine_agent_signals()` aplica fallback: para tickers/fechas donde el Juez retorna NaN, se usa la señal del Matemático directamente
5. `sentiment_coverage()` permite diagnosticar la cobertura por ticker

**Implicación para el backtest:** tickers con poca cobertura de noticias se comportan como si solo tuvieran el Matemático. El Analista añade valor solo donde tiene datos suficientes.

---

## 13. Sesgos Conocidos y Limitaciones

| Sesgo / Limitación | Descripción | Impacto estimado |
|---|---|---|
| **Survivorship bias parcial** | El universo incluye underperformers pero no empresas que desaparecieron del Nasdaq 100 entre 2018-2024 | Resultados ligeramente optimistas |
| **Look-ahead implícito en selección** | Los 6 ETFs se eligieron sabiendo que funcionan como refugio — en 2018 no sabríamos esto de antemano | Leve sesgo en la diversificación |
| **Datos de noticias limitados** | NewsAPI solo ofrece 1 mes gratuito; datos históricos de noticias son caros | El Analista necesitará fuentes alternativas para backtest completo |
| **Form 4 con lag incompleto** | El lag de 2 días hábiles es el mínimo legal pero en la práctica puede ser mayor | Estimación optimista del Cazador |
| **Sin modelado de impacto de mercado** | En portfolio real, órdenes grandes mueven el precio (slippage) | Resultados optimistas en capital simulado grande |
| **Perfil Day Simulated** | Open/Close diarios no modelan spread ni slippage intraday | Resultados muy optimistas para ese perfil específico |
| **Un solo modelo base (XGBoost)** | El Matemático no ensemble modelos de distintas familias | Menor robustez ante cambios de régimen |

---

## 14. Stack Tecnológico

### 14.1 Elección de tecnologías y justificación

| Capa | Tecnología | Justificación |
|---|---|---|
| **Lenguaje** | Python 3.11+ | Ecosistema ML más maduro; type hints modernos |
| **Datos OHLCV** | `yfinance` | Gratuito, sin API key, ajuste splits/dividendos automático |
| **Manipulación** | `pandas`, `numpy` | Estándar de facto en datos financieros |
| **Almacenamiento** | Parquet via `pyarrow` | Sin servidor, comprimido, muy rápido para series temporales largas |
| **Calendario** | `pandas_market_calendars` | NYSE holidays y sesiones correctas |
| **ML** | `scikit-learn`, `xgboost` | Calibración nativa en sklearn; XGBoost el mejor gradient boosting para tabular |
| **Calibración** | `CalibratedClassifierCV` (sklearn) | Platt Scaling e Isotonic Regression en un wrapper |
| **NLP** | `transformers` (Hugging Face) + `torch` | FinBERT pre-entrenado descargable sin API |
| **RL (v2)** | `stable-baselines3` + `gymnasium` | Estándar para RL aplicado en Python |
| **Anomalías** | `scikit-learn` (IsolationForest) + `hmmlearn` | Detección no supervisada sin labels de crisis |
| **Dashboard** | `Streamlit` | Prototipos rápidos con aspecto profesional; despliegue gratuito en Streamlit Cloud |
| **Visualización** | `plotly`, `matplotlib`, `seaborn` | Plotly para dashboard interactivo; matplotlib/seaborn para EDA |
| **Testing** | `pytest`, `pytest-cov` | Estándar; fixtures reutilizables |
| **Config** | `PyYAML` | YAML es legible y soporta comentarios |

### 14.2 Por qué no se usa un LLM de API para el Analista

El plan original insinuaba usar GPT o similar vía API. Se rechazó por:

1. **Coste:** procesar titulares de 46 activos × 252 días × 7 años sería caro para backtesting completo
2. **Reproducibilidad:** los modelos de API no son deterministas (no se puede fijar una seed en ellos)
3. **Latencia:** llamadas HTTP para cada titular hacen el backtesting lento
4. **FinBERT es mejor para esta tarea:** está específicamente entrenado en texto financiero (informes SEC, artículos de finanzas) y supera a GPT genérico en sentiment financiero según la literatura

FinBERT (`ProsusAI/finbert`) se descarga localmente (~438 MB) y corre en CPU/GPU del desarrollador. Coste: cero.

---

## 15. Hoja de Ruta

### 15.1 Fases del proyecto

| Fase | Semanas | Objetivo | Criterio de éxito |
|---|---|---|---|
| **1 — Datos** | 1-2 | Pipeline de datos limpio | 46 activos en local, sin gaps, features calculadas |
| **2 — Backtester** | 3-4 | Framework de validación + baselines | Baselines documentados y reproducibles |
| **3 — MVP** | 5-6 | El Matemático + Juez v1 | Matemático supera SMA Crossover en walk-forward |
| **4 — Analista** | 7-8 | NLP + ensemble 2 agentes | Matemático+Analista > Matemático solo |
| **5 — Riesgos/Insiders** | 9-10 | Kelly + Cazador | MaxDD se reduce al añadir Gestor de Riesgos |
| **6 — Conspiranoico/RL** | 11-12 | Detección de régimen | Conspiranoico reduce pérdidas en crisis documentadas |
| **7 — Dashboard** | 13-14 | Streamlit + XAI | Demo funcional: datos → decisión → log → visualización |
| **8 — Perfiles** | 15-16 | Swing, Long-term, Day Simulated | Tabla comparativa de Sharpe/MaxDD/Calmar |
| **9 — El Explorador** | 17-18 | Universo dinámico con HITL | Flujo recomendación → aprobación → cambio aplicado |
| **10 — Demo pública** | 19-20 | Web portfolio | Dashboard público + README + métricas honestas |

### 15.2 Regla de oro: no avanzar sin el criterio de éxito de la fase anterior

La tentación es avanzar a la siguiente fase aunque la anterior no esté limpia. En este proyecto, un bug en el backtester invalida todos los resultados. Cada fase tiene un criterio de éxito explícito que debe cumplirse antes de continuar.

### 15.3 Estado actual del proyecto (19 Junio 2026)

**Completado:**
- `config.yaml` con todos los parámetros
- `data/universe_2018-01-01.csv` con los 46 activos
- `data/features.py` — pipeline completo de features técnicas
- `data/downloader.py` — descarga, caché y control de calidad
- `data/news.py` — pipeline de noticias financieras (Alpaca API + caché) ✅ **Nuevo Fase 4**
- `data/insiders.py` — pipeline de transacciones de insiders (SEC EDGAR API + caché Parquet) ✅ **Nuevo Fase 5 / migrado EDGAR jun-2026**
- `agents/base_agent.py` — interfaz abstracta
- `agents/matematico.py` — XGBoost + Platt Scaling completamente implementado
- `agents/analista.py` — FinBERT + Platt Scaling completamente implementado ✅ **Nuevo Fase 4**
- `agents/cazador.py` — reglas SEC EDGAR + lag Form 4 completamente implementado ✅ **Nuevo Fase 5 / migrado EDGAR jun-2026**
- `agents/gestor_riesgos.py` — Fractional Kelly + caps por clase completamente implementado
- `backtester/metrics.py` — métricas completas (incluyendo `compare_to_baseline`)
- `backtester/engine.py` — motor completo con comisiones, stop-loss dinámico y logs auditables con `agent_votes`
- `backtester/walk_forward.py` — orquestador walk-forward multi-agente con Cazador y `agent_votes_log` ✅ **Actualizado Fase 5**
- `baselines/buy_and_hold.py` — implementado
- `baselines/sma_crossover.py` — implementado (sub-portfolios independientes por ticker)
- `judge/judge_v1.py` — implementado (modo pass-through + meta-modelo, veto del Conspiranoico)
- `run.py` — script de ejecución con soporte multi-agente y flag `--cazador` ✅ **Actualizado Fase 5**
- `utils/reproducibility.py` — seeds fijas
- `utils/config_loader.py` — carga y caché
- `tests/test_kelly.py` — 24 tests completamente implementados
- `tests/test_cazador.py` — 33 tests (parsing Form 4 XML, reglas, lag Form 4, agent) ✅ **Nuevo Fase 5**
- `tests/test_backtester.py` — tests de anti-leakage, comisiones y reproducibilidad implementados
- `tests/test_data_integrity.py` — tests de calidad de datos implementados
- `data/universe_manager.py` — gestor del universo de producción (para Fase 9)
- `agents/explorador.py` — estructura completa con docstrings (para Fase 9)
- `profiles/swing.yaml`, `profiles/long_term.yaml`, `profiles/day_simulated.yaml` — perfiles definidos

**Pendiente (stubs):**
- `agents/conspiranoico.py` → Fase 6
- `dashboard/app.py` → Fase 7

**Fase actual:** Fase 5 completada. El Cazador implementado y pipeline integrado end-to-end. Ver sección 15.6.

**Próximo paso:** Fase 6 — El Conspiranoico (Isolation Forest para detección de régimen), baseline Mat + Cazador (Run A, Sharpe 0.638). Prioritario para mitigar pérdidas en 2022.

### 15.4 Primer experimento ejecutado — Fase 3 MVP

**Experimento:** `experiments/fase3_matematico_20260617_225944/`

Pipeline: Matemático (XGBoost + Platt Scaling) + Juez v1 (pass-through en iter 1, meta-modelo en iters 2-4) + Gestor de Riesgos (Half-Kelly) + BacktestEngine.

**Período:** 2021–2024 (4 ventanas walk-forward). Holdout 2025 no tocado.

**Resultados agregados (out-of-sample, 1005 días de trading):**

| Sistema | Retorno Total | Retorno Anual | Sharpe | Max Drawdown | Calmar |
|---|---|---|---|---|---|
| **MAS (Matemático)** | +37.66% | +9.37% | 0.568 | -25.71% | 0.365 |
| **Buy & Hold** | +67.90% | +15.25% | 0.723 | -31.36% | 0.486 |
| **SMA Crossover** | +14.47% | +4.21% | 0.329 | -22.25% | 0.189 |

**Resultados por ventana walk-forward:**

| Iteración | Período validación | Sharpe | Max DD | Retorno |
|---|---|---|---|---|
| 1 | 2021 | **1.364** | -7.51% | +19.91% |
| 2 | 2022 | -0.774 | -25.23% | -18.60% |
| 3 | 2023 | **1.935** | -9.56% | +28.77% |
| 4 | 2024 | 0.773 | -7.29% | +9.53% |

**Análisis de los resultados:**

1. **El Matemático supera al SMA Crossover:** Sharpe 0.568 vs 0.329 (+0.239). El ML añade valor respecto a la estrategia técnica más simple. Criterio de éxito de Fase 3 superado en ese sentido.

2. **El Matemático no supera a Buy & Hold en Sharpe:** 0.568 vs 0.723. Sin embargo, el sistema reduce el Max Drawdown notablemente: -25.71% vs -31.36% del Buy & Hold. Hay un trade-off retorno/riesgo.

3. **Alta varianza entre ventanas:** la iteración 2 (año 2022, mercado bajista por subida de tipos) fue muy negativa (-18.60%), mientras que 2021 y 2023 fueron excelentes. Esto es esperado con un solo agente y sin El Conspiranoico para vetar en regímenes adversos.

4. **Interpretación honesta:** un Sharpe de 0.568 en validación out-of-sample con un solo agente técnico es un resultado técnicamente sólido como punto de partida. No es el objetivo final (Sharpe > 1.0), pero es la base sobre la que construir los siguientes agentes.

5. **El año 2022 es el test más difícil:** el mercado cayó un 33% (Nasdaq). Que el sistema perdiera un 18.6% en ese contexto (vs ~33% del mercado) puede interpretarse como una señal positiva de gestión del riesgo, aunque las métricas anuales son negativas.

**Decisión:** continuar con Fase 4 (El Analista). El Conspiranoico (Fase 6) es especialmente relevante para mitigar el problema de 2022.

### 15.5 Primer experimento ejecutado — Fase 4 Analista

**Experimento:** `experiments/fase4_mat_analista_20260619_004415/`

Pipeline: Matemático (XGBoost + Platt Scaling) + Analista (FinBERT + Platt Scaling) + Juez v1 (pass-through en iter 1, meta-modelo con [matemático, analista] en iters 2-4) + Gestor de Riesgos (Half-Kelly) + BacktestEngine.

**Fuente de noticias:** Alpaca News API (`data.alpaca.markets/v1beta1/news`), caché local en `data/cache/news/` y sentimiento precomputado en `data/cache/sentiment/`.

**Período:** 2021–2024 (4 ventanas walk-forward). Holdout 2025 no tocado.

**Resultados agregados (out-of-sample, 1005 días de trading):**

| Sistema | Retorno Total | Retorno Anual | Sharpe | Max Drawdown | Calmar |
|---|---|---|---|---|---|
| **MAS (Mat + Analista)** | +44.10% | +10.68% | 0.612 | -26.57% | 0.402 |
| **Buy & Hold** | +67.90% | +15.25% | 0.723 | -31.36% | 0.486 |
| **SMA Crossover** | +14.47% | +4.21% | 0.329 | -22.25% | 0.189 |

**Resultados por ventana walk-forward:**

| Iteración | Período validación | Sharpe | Max DD | Retorno |
|---|---|---|---|---|
| 1 | 2021 | **1.356** | -7.67% | +20.62% |
| 2 | 2022 | -0.782 | -26.18% | -19.64% |
| 3 | 2023 | **1.998** | -9.54% | +34.24% |
| 4 | 2024 | 0.840 | -8.28% | +10.75% |

**Comparativa directa con Fase 3 (Matemático solo):**

| Métrica | Fase 3 (Mat solo) | Fase 4 (Mat + Analista) | Delta |
|---|---|---|---|
| Sharpe | 0.568 | **0.612** | **+0.044** |
| Retorno total | +37.66% | **+44.10%** | **+6.44 pp** |
| Retorno anual | +9.37% | **+10.68%** | +1.31 pp |
| Max Drawdown | -25.71% | -26.57% | -0.86 pp |
| Calmar | 0.365 | **0.402** | +0.037 |

**Análisis de los resultados:**

1. **El Analista añade valor sobre el Matemático solo:** Sharpe 0.612 vs 0.568 (+0.044) y retorno total +44.10% vs +37.66% (+6.44 pp). Criterio de éxito de Fase 4 cumplido.

2. **La mejora es modesta pero consistente:** el delta de Sharpe (+0.044) no es espectacular, pero es positivo en el período completo out-of-sample. Las ventanas que más mejoran son 2023 (+0.063 Sharpe) y 2024 (+0.067 Sharpe); 2021 y 2022 permanecen prácticamente iguales.

3. **El ensemble no supera a Buy & Hold en Sharpe:** 0.612 vs 0.723. El gap se reduce respecto a Fase 3 (0.568 vs 0.723), pero sigue sin cerrarse. Max Drawdown del MAS (-26.57%) sigue siendo mejor que Buy & Hold (-31.36%).

4. **2022 sigue siendo el año problemático:** Sharpe -0.782, retorno -19.64%. El Analista no mitiga regímenes bajistas estructurales; eso es tarea del Conspiranoico (Fase 6).

5. **Max Drawdown ligeramente peor que Fase 3:** -26.57% vs -25.71% (+0.86 pp). Trade-off aceptable dado el incremento de retorno, pero documentar como sesgo conocido: más señales activas pueden implicar más exposición en momentos adversos.

6. **Interpretación honesta:** un Sharpe de 0.612 con dos agentes en walk-forward es un resultado técnicamente creíble. La señal de sentimiento aporta información complementaria al análisis técnico, especialmente en regímenes alcistas (2023-2024). No es suficiente por sí sola para alcanzar Sharpe > 1.0 agregado.

**Nota sobre ejecuciones fallidas previas:** las primeras ejecuciones con `--analista` produjeron resultados idénticos a Fase 3 porque la caché de noticias (`data/cache/news/`) contenía archivos Parquet vacíos de un intento sin API keys configuradas. Solución: borrar `data/cache/news/` y `data/cache/sentiment/`, configurar `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`, y re-ejecutar con `--force-download`.

**Decisión:** continuar con Fase 5 (El Cazador + Gestor de Riesgos). El Conspiranoico (Fase 6) sigue siendo prioritario para mitigar el problema de 2022.

### 15.6 Experimentos ejecutados — Fase 5 Cazador

Se ejecutaron dos configuraciones el mismo día (19 jun 2026), tras migrar la fuente de datos de OpenInsider a SEC EDGAR:

| Run | Comando | Experimento |
|---|---|---|
| **A (recomendado)** | `python run.py --cazador` | `experiments/fase5_mat_cazador_20260619_133226/` |
| **B** | `python run.py --analista --cazador` | `experiments/fase5_mat_analista_cazador_20260619_135320/` |

Pipeline común: Matemático (XGBoost + Platt Scaling) + Cazador (reglas Form 4 SEC, señal binaria 0/1) + Juez v1 (pass-through en iter 1, meta-modelo con columnas de agentes activos en iters 2-4) + Gestor de Riesgos (Half-Kelly) + BacktestEngine.

**Fuente de insiders:** SEC EDGAR API (`data.sec.gov`), caché local en `data/cache/insiders/` (Parquet por ticker + `_cik_map.json`). Requiere `cazador.sec_user_agent` en `config.yaml`.

**Período:** 2021–2024 (4 ventanas walk-forward). Holdout 2025 no tocado.

#### Run A — Mat + Cazador (mejor resultado Fase 5)

**Experimento:** `experiments/fase5_mat_cazador_20260619_133226/`

**Resultados agregados (out-of-sample, 1005 días de trading):**

| Sistema | Retorno Total | Retorno Anual | Sharpe | Max Drawdown | Calmar |
|---|---|---|---|---|---|
| **MAS (Mat + Cazador)** | +49.43% | +11.78% | 0.638 | -27.89% | 0.422 |
| **Buy & Hold** | +67.90% | +15.25% | 0.723 | -31.36% | 0.486 |
| **SMA Crossover** | +14.47% | +4.21% | 0.329 | -22.25% | 0.189 |

**Resultados por ventana walk-forward:**

| Iteración | Período validación | Sharpe | Max DD | Retorno |
|---|---|---|---|---|
| 1 | 2021 | **1.493** | -7.68% | +22.04% |
| 2 | 2022 | -0.758 | -27.48% | -20.94% |
| 3 | 2023 | **2.101** | -10.29% | +40.16% |
| 4 | 2024 | 0.813 | -8.30% | +10.49% |

#### Run B — Mat + Analista + Cazador

**Experimento:** `experiments/fase5_mat_analista_cazador_20260619_135320/`

**Resultados agregados (out-of-sample, 1005 días de trading):**

| Sistema | Retorno Total | Retorno Anual | Sharpe | Max Drawdown | Calmar |
|---|---|---|---|---|---|
| **MAS (Mat + Analista + Cazador)** | +44.80% | +10.81% | 0.619 | -27.59% | 0.392 |
| **Buy & Hold** | +67.90% | +15.25% | 0.723 | -31.36% | 0.486 |
| **SMA Crossover** | +14.47% | +4.21% | 0.329 | -22.25% | 0.189 |

**Resultados por ventana walk-forward:**

| Iteración | Período validación | Sharpe | Max DD | Retorno |
|---|---|---|---|---|
| 1 | 2021 | **1.493** | -7.68% | +22.04% |
| 2 | 2022 | -0.772 | -27.18% | -19.82% |
| 3 | 2023 | **1.993** | -9.62% | +33.79% |
| 4 | 2024 | 0.832 | -8.48% | +10.60% |

#### Comparativa directa con Fase 4 (Mat + Analista)

Referencia: `experiments/fase4_mat_analista_20260619_004415/` (Sharpe 0.612, retorno +44.10%).

| Métrica | Fase 4 | Run A (Mat+Cazador) | Delta A | Run B (Mat+Ana+Caz) | Delta B |
|---|---|---|---|---|---|
| Sharpe | 0.612 | **0.638** | **+0.026** | 0.619 | +0.007 |
| Retorno total | +44.10% | **+49.43%** | **+5.33 pp** | +44.80% | +0.70 pp |
| Retorno anual | +10.68% | **+11.78%** | +1.10 pp | +10.81% | +0.13 pp |
| Max Drawdown | **-26.57%** | -27.89% | -1.32 pp | -27.59% | -1.02 pp |
| Calmar | 0.402 | **0.422** | +0.020 | 0.392 | -0.010 |

**Comparativa por ventana vs Fase 4 (Sharpe):**

| Ventana | Fase 4 | Run A | Delta A | Run B | Delta B |
|---|---|---|---|---|---|
| 2021 | 1.356 | **1.493** | +0.137 | **1.493** | +0.137 |
| 2022 | -0.782 | -0.758 | +0.024 | -0.772 | +0.010 |
| 2023 | 1.998 | **2.101** | +0.103 | 1.993 | -0.005 |
| 2024 | **0.840** | 0.813 | -0.027 | 0.832 | -0.008 |

**Análisis de los resultados:**

1. **El Cazador añade valor sobre Fase 4, pero solo en la config Mat + Cazador (Run A):** Sharpe 0.638 vs 0.612 (+0.026) y retorno total +49.43% vs +44.10% (+5.33 pp). Criterio de éxito de Fase 5 cumplido en Run A.

2. **Mat + Analista + Cazador (Run B) no mejora de forma significativa a Fase 4:** Sharpe 0.619 vs 0.612 (+0.007) y retorno +44.80% vs +44.10% (+0.70 pp). Añadir el Cazador encima del Analista aporta casi nada; Run A supera a Run B en Sharpe (+0.019) y retorno (+4.63 pp).

3. **La mejora del Run A está concentrada en 2021 y 2023:** en 2023 el retorno sube de +34.24% (Fase 4) a +40.16% (+5.92 pp) y el Sharpe de 1.998 a 2.101. En 2024, Fase 4 sigue siendo ligeramente mejor (Sharpe 0.840 vs 0.813), lo que sugiere que el Analista aporta señal útil en el último año cuando no compite con el Cazador solo.

4. **2022 sigue siendo el año problemático:** Sharpe negativo en ambos runs (-0.758 / -0.772). El Cazador no mitiga regímenes bajistas estructurales; eso sigue siendo tarea del Conspiranoico (Fase 6).

5. **Trade-off de riesgo:** Max Drawdown algo peor que Fase 4 (-27.89% vs -26.57% en Run A). Más retorno a cambio de mayor drawdown. Sigue siendo mejor que Buy & Hold (-31.36%).

6. **Iteración 1 idéntica en ambos runs de Fase 5:** en la primera ventana walk-forward el Juez opera en pass-through (solo Matemático); las diferencias aparecen a partir de la iter 2, cuando el meta-modelo entrena con votos de agentes de la ventana anterior.

7. **El ensemble no supera a Buy & Hold en Sharpe:** mejor Run A 0.638 vs 0.723. El gap se reduce respecto a Fase 4 (0.612 vs 0.723), pero no se cierra.

8. **Interpretación honesta:** un Sharpe de 0.638 con Mat + Cazador en walk-forward es el mejor resultado acumulado del proyecto hasta la fecha. La señal de ventas de insiders aporta información complementaria al análisis técnico, especialmente en 2023. No obstante, combinar Analista y Cazador simultáneamente no genera sinergia — posible redundancia de señales o conflicto en el meta-modelo del Juez.

**Nota sobre la fuente de datos:** las primeras implementaciones usaban scraping de OpenInsider, que bloqueaba conexiones. Se migró a SEC EDGAR (junio 2026). Tras la migración, borrar `data/cache/insiders/` si contenía caché del formato antiguo (`transaction_type` = `"S - Sale"` en lugar de `"S"`).

**Config recomendada tras Fase 5:** `python run.py --cazador` (Run A). Para producción futura con todos los agentes activos, evaluar si Analista y Cazador deben coexistir o alternarse por régimen.

**Decisión:** Fase 5 cerrada. Continuar con Fase 6 — El Conspiranoico (Isolation Forest para detección de régimen), usando Run A (Mat + Cazador) como baseline. El Conspiranoico es prioritario para mitigar el problema de 2022.



