# Sistema Multi-Agente (MAS) para Predicción y Toma de Decisiones en Mercados Financieros

---

## 1. Descripción General

El proyecto consiste en el desarrollo de una arquitectura de Inteligencia Artificial basada en un **Ensemble de Modelos Expertos** (conceptualmente, un "Consejo de Agentes"). Diferentes agentes especializados evalúan el mercado desde perspectivas aisladas y envían sus conclusiones a un **Juez Central** o Meta-Modelo, que pondera la fiabilidad de cada agente para tomar la decisión final de inversión.

### Restricciones de Diseño (no negociables desde el inicio)

| Parámetro | Decisión | Justificación |
|---|---|---|
| **Mercado** | Acciones US (Nasdaq 100) | Mayor disponibilidad de datos gratuitos y de calidad |
| **Universo** | **40 acciones del Nasdaq 100** + **6 ETFs de diversificación** (ver sección 3.1) | Acciones: motor de crecimiento. ETFs: amortiguadores ante crisis del mercado americano |
| **Horizonte temporal** | **Señales diarias** — decisión al cierre de cada día | Un solo horizonte evita el problema de mezclar señales incompatibles |
| **Capital simulado** | 10.000 € ficticios | Referencia realista para métricas |
| **Comisiones simuladas** | 0,08% por operación (ida) | Aproximación de brokers como IBKR o Degiro; destruye estrategias con demasiadas operaciones |

> **Sobre el horizonte temporal:** Mezclar day trading, swing e inversión a largo plazo en el mismo sistema hace que el Juez combine señales incompatibles (una señal diseñada para 5 minutos no puede combinarse con una señal de insider trading que tarda semanas en materializarse). Esto se puede explorar en una v2 una vez el sistema base funcione.

> **Sobre el universo de activos:** La selección se congela a **1 de enero de 2018** para evitar survivorship bias retrospectivo (no elegir empresas que "sabemos" que sobrevivieron hasta 2026). El universo incluye 8 acciones con rendimiento mediocre o negativo (INTC, GILD, BIIB, WBA, BIDU, PYPL, ILMN) para que el backtest sea creíble ante revisión técnica. Esta limitación debe documentarse explícitamente como sesgo conocido del estudio.

---

## 2. El Consejo de Agentes

| Agente | Función Principal | Tecnología | Dificultad Real | Voto al Juez |
|---|---|---|---|---|
| **El Matemático** | Análisis técnico puro: tendencias en precio y volumen histórico | XGBoost sobre features técnicas (RSI, MACD, Bollinger, ATR) + LSTM opcional en v2 | Media | Probabilidad calibrada p ∈ [0, 1] de que el precio suba mañana |
| **El Analista (Noticias)** | Mide el pánico o euforia leyendo titulares financieros | FinBERT (modelo pre-entrenado) sobre noticias de NewsAPI o Alpaca News | Media-Alta | Probabilidad calibrada p ∈ [0, 1] por ticker |
| **El Cazador (Insiders)** | Busca actividad inusual de directivos o políticos | API de OpenInsider o scraping de Form 4 de SEC/EDGAR | **Media** (no Baja: el parsing XML de Form 4 no es trivial) | Señal binaria: alerta/silencio |
| **El Conspiranoico** | Detección de régimen de mercado anómalo. Vota "abstenerse" si detecta un entorno de alta incertidumbre | Isolation Forest + HMM (Hidden Markov Model) para detección de régimen | Alta | Señal de veto: activo/inactivo |
| **El Gestor de Riesgos** | No predice. Calcula tamaño de posición (solo Long) y condiciones de salida de emergencia | **Fractional Kelly** (Half-Kelly, ρ=0.5) + stop-loss dinámico basado en ATR + cap del 15% por ticker | Baja | Fracción de capital f* ∈ [0, 0.15] por ticker; si f* ≤ 0 → no operar / cerrar posición |
| **El Juez Central** | Combina los votos, ponderados por su fiabilidad histórica reciente, y emite la orden final | **v1: Ensemble estático ponderado** (regresión logística o XGBoost como meta-modelo). **v2: RL (PPO)** si el v1 funciona | Media (v1) / Muy Alta (v2) | Orden: COMPRA / VENTA / MANTENER |
| **El Explorador** *(v2)* | Monitorea el historial de paper trades para proponer cambios al universo activo. Solo opera en producción, nunca modifica el backtest histórico | Estadística sobre `logs/trades/` (retiros) + `yfinance` (nuevas incorporaciones) | Baja-Media | Recomendación a revisar por humano (**Human-in-the-Loop obligatorio**) |

### Nota crítica sobre El Conspiranoico

El agente de anomalías no vota "alcista" o "bajista" — eso es indefinible en un mercado anómalo. Su único voto válido es: **"el entorno actual no es fiable para operar"**. Cuando está activo, el Juez reduce la exposición al 0% o al mínimo definido por el Gestor de Riesgos, independientemente de lo que digan los demás agentes. Esto es lo que lo hace útil.

### Nota crítica sobre El Juez Central

El plan original proponía Aprendizaje por Refuerzo (RL) desde el inicio. **Esto es un error de diseño para un MVP** por dos razones:
1. El RL en finanzas tiende a aprender a "no invertir nunca" si la función de recompensa no está muy bien diseñada.
2. Es un componente de "Muy Alta" dificultad que puede bloquear el proyecto completo.

La estrategia correcta: **construir el Juez v1 como un meta-modelo de ensemble simple**. Una vez que el sistema completo funciona de extremo a extremo y existe un benchmark sólido, introducir RL como mejora medible es un objetivo de v2.

---

## 3. El Baseline — Lo más importante que faltaba en el plan original

Antes de evaluar si el sistema "funciona", necesitas una referencia contra la que comparar. Sin baseline, un Sharpe de 1.2 en backtest no significa nada.

| Estrategia Baseline | Descripción | Por qué importa |
|---|---|---|
| **Buy & Hold** | Comprar las 30-50 acciones del universo en partes iguales y no tocarlas | Si tu sistema no supera esto, es peor que no hacer nada |
| **SMA Crossover (20/50)** | Comprar cuando la media de 20 días cruza sobre la de 50 | Estrategia técnica clásica; el mínimo bar a superar para un sistema ML |

Si el Consejo de Agentes no supera ambos baselines de forma consistente en validación walk-forward, los resultados no son creíbles.

---

## 3.1 Diversificación del Universo de Activos

El universo no se limita a acciones del Nasdaq. Se añaden **6 ETFs** de clases de activos distintas para proteger la cartera ante una crisis del mercado americano o tecnológico. Todos son descargables con `yfinance` sin cambiar ningún agente.

### Los 6 ETFs de diversificación

| Ticker | Nombre | Clase | Función en la cartera |
|---|---|---|---|
| **TLT** | iShares 20+ Year Treasury Bond ETF | Renta Fija (largo plazo) | Refugio principal en crisis: sube cuando las acciones caen |
| **IEF** | iShares 7-10 Year Treasury Bond ETF | Renta Fija (medio plazo) | Estabilidad; menos sensible a tipos que TLT |
| **GLD** | SPDR Gold Shares ETF | Oro | Refugio clásico; descorrelacionado de acciones y bonos |
| **XLU** | Utilities Select Sector SPDR | Renta Variable Defensiva | Sector muy estable (electricidad, agua); baja correlación con tech |
| **XLP** | Consumer Staples Select Sector SPDR | Renta Variable Defensiva | Empresas defensivas (P&G, Coca-Cola); resisten recesiones |
| **EFA** | iShares MSCI EAFE ETF | Renta Variable Internacional | Diversificación geográfica (Europa + Japón + Australia) |

### Caps diferenciados por clase de activo

El Gestor de Riesgos aplica límites distintos según la clase de activo, en dos niveles:

| Clase de activo | Cap por ticker | Cap total del portfolio |
|---|---|---|
| `equity` (acciones Nasdaq) | 15% | 80% |
| `bond` (TLT, IEF) | 20% | 30% |
| `gold` (GLD) | 10% | 10% |
| `defensive_equity` (XLU, XLP) | 15% | 20% |
| `international_equity` (EFA) | 10% | 10% |

Los bonos tienen un cap por ticker ligeramente mayor (20%) porque su volatilidad es significativamente menor que la de las acciones; Kelly lo reflejaría de todas formas, pero el cap actúa como límite de seguridad adicional.

### Lo que NO se incluye (y por qué)

| Activo | Decisión | Justificación |
|---|---|---|
| **Bitcoin / Crypto** | ❌ Excluido en v1 | Volatilidad 5-10× mayor que acciones; opera 24/7 rompiendo el calendario NYSE; señales del Analista con ruido extremo |
| **Acciones europeas individuales** | ❌ Excluido | Festivos distintos, divisas, cobertura de noticias insuficiente en inglés |
| **Acciones emergentes individuales** | ❌ Excluido | El Cazador (Form 4 de SEC) no aplica fuera de EE.UU.; datos de menor calidad |
| **Materias primas (petróleo, trigo)** | ❌ Excluido | Alta especificidad de señales; `GLD` cubre el refugio sin esta complejidad |

> **Bitcoin como v2 opcional:** si se quiere añadir en el futuro, necesitaría un sub-agente propio con calibración de Kelly separada y un cap máximo del 2-3% en `config.yaml`. El sistema actual no lo soporta sin ajustes en el backtester (calendario 24/7).

---

## 3.2 Perfiles de Trading y El Explorador *(Expansión v2)*

Estas dos funcionalidades se construyen **después de que el sistema base esté validado** (Fase 7 completada). Son extensiones naturales que aprovechan la infraestructura ya creada.

### Tres perfiles de trading comparables

El núcleo del sistema (agentes, backtester, métricas) no cambia. Los perfiles son simplemente conjuntos de parámetros distintos definidos en `profiles/*.yaml` que el backtester usa según indicación.

| Perfil | Frecuencia de señal | Horizonte de posición | Caso de uso |
|---|---|---|---|
| **Swing** *(por defecto)* | Diaria (al cierre) | Días a semanas | El sistema principal; el más equilibrado |
| **Long-term** | Mensual (rebalanceo) | Meses | Ventanas largas (SMA 50/200); menos operaciones; menor impacto de comisiones |
| **Day simulated** | Diaria (al cierre T-1) | Un día (abre en T, cierra en T) | **Solo académico.** Proxy de day trading con Open/Close diarios. No modela spread ni slippage real |

> **Limitación crítica del perfil Day Simulated:** los datos diarios de yfinance solo tienen precios de apertura y cierre. El day trading real requiere datos de minutos u horas, además de modelar el spread bid-ask y el slippage. Los resultados de este perfil son una **aproximación optimista** útil para entender el impacto del horizonte temporal, pero no son representativos de un sistema de day trading real. Se documenta explícitamente esta limitación en el dashboard y en la web de portfolio.

El objetivo de los tres perfiles es **comparativo**: mostrar cómo el mismo conjunto de agentes se comporta de forma diferente según el horizonte temporal, lo cual es un resultado educativo valioso para el portfolio.

### El Explorador — Gestión dinámica del universo *(Human-in-the-Loop)*

El universo de **backtest** está fijado para siempre en `data/universe_2018-01-01.csv`. Esto es innegociable: un universo cambiante invalidaría los resultados históricos.

El universo de **paper trading / producción** puede evolucionar, con supervisión humana, gracias a El Explorador. Gestiona dos casos distintos:

**Para RETIRAR activos** (basado en paper trades propios):
- Si El Matemático genera f*=0 el 90%+ de los días para un ticker → sin señal útil
- Si la accuracy en ese ticker es < 50% → el modelo no sabe predecirlo
- → Propone retirarlo y espera aprobación humana

**Para AÑADIR activos** (basado en datos externos de mercado):
- Evalúa candidatos externos con `yfinance` (correlación con la cartera, calidad de datos, liquidez)
- Si la correlación media con la cartera < -0.10 → aporta diversificación real
- → Propone añadirlo y espera aprobación humana
- *(Extensión futura — ver **RECORDATORIO** en Fase 9)*: criterio adicional con compras de insiders para candidatos ADD

**La aprobación humana es obligatoria** para cualquier cambio. El Explorador nunca modifica el universo directamente; genera recomendaciones en `logs/explorador_recommendations.jsonl` que aparecen en el dashboard para ser aprobadas o rechazadas. Cada cambio aprobado se registra en `logs/universe_changes.jsonl` con fecha, razón y evidencia.

---

## 4. Hoja de Ruta Corregida

> **Cambio crítico respecto al plan original:** el framework de backtesting se construye en la **Fase 2**, antes de cualquier modelo. Construir modelos sin validación garantiza data leakage que solo se descubrirá al final.

### Fase 1 — Infraestructura de Datos (Semanas 1-2)

**Objetivo:** pipeline de datos limpio, reproducible y libre de errores antes de tocar ningún modelo.

- [x] Crear `config.yaml` con todos los parámetros del proyecto (seeds, comisiones, tickers, etc.)
- [x] Crear `data/nasdaq100_top40_2018-01-01.csv` con el universo de tickers fijado a esa fecha
- [x] Descarga de OHLCV diario para las 30-50 acciones del universo via `yfinance` (2018–hoy)
- [x] Ajuste por splits y dividendos (yfinance lo hace automáticamente, pero hay que verificarlo)
- [x] Almacenamiento en base de datos local (SQLite o Parquet)
- [x] Script de verificación de calidad: gaps, valores nulos, fechas duplicadas
- [x] Caché local para no depender de la API en cada ejecución
- [x] Cálculo y almacenamiento de features técnicas base: RSI, MACD, Bollinger Bands, ATR, volumen relativo

**Entregable:** `data/` con datos limpios y un notebook de exploración que muestre las distribuciones y confirme la ausencia de errores.

---

### Fase 2 — Framework de Backtesting y Baselines (Semanas 3-4)

**Objetivo:** el "tubo" de validación que usarán todos los agentes. Sin esto, cualquier resultado es inválido.

- [x] Implementar backtester propio o integrar `vectorbt` / `backtesting.py`
- [x] Implementar **walk-forward validation**: ventanas de entrenamiento rodantes (ej: entrenar en 2018-2020, validar en 2021; entrenar en 2018-2021, validar en 2022; etc.)
- [x] Implementar simulación de comisiones: 0,08% por operación
- [x] Calcular métricas estándar: Sharpe Ratio, Max Drawdown, % de operaciones ganadoras, Calmar Ratio
- [x] Ejecutar y documentar los **dos baselines** (Buy & Hold y SMA Crossover)

**Entregable:** `backtester/` con los resultados de los baselines documentados. Estos son los números a batir.

> **Anti-patrón a evitar:** usar un único train/test split temporal (ej: 2018-2022 train, 2023-2024 test). Esto no detecta si el modelo se sobreajusta a un régimen de mercado específico. El walk-forward es obligatorio.

---

### Fase 3 — MVP: El Matemático + Juez v1 (Semanas 5-6)

**Objetivo:** sistema funcional de extremo a extremo. Datos → modelo → decisión → backtesting → métricas.

- [x] Entrenar **El Matemático**: XGBoost con las features técnicas de la Fase 1
  - Target: **clasificación binaria** — sube (retorno > 0%) / baja (retorno ≤ 0%) el día siguiente
  - **No usar 3 clases** (sube/baja/lateral): el umbral de "lateral" es un hiperparámetro arbitrario que introduce sobreajuste
  - La decisión de "no operar" por baja confianza se delega al Gestor de Riesgos vía Kelly, no al modelo
- [x] **Calibración de probabilidades** (obligatorio, dentro de cada ventana walk-forward):
  - Aplicar **Platt Scaling** (`CalibratedClassifierCV(method='sigmoid')`) como primera opción
  - Verificar con un **reliability diagram** (`calibration_curve` de sklearn): la curva debe aproximarse a la diagonal
  - Si la curva se desvía significativamente, probar **Isotonic Regression** (`method='isotonic'`) como alternativa
  - El output final es una probabilidad calibrada p ∈ [0, 1] que el Gestor de Riesgos usará directamente en Kelly
- [x] Implementar **El Juez v1**: pasa directamente la señal calibrada del Matemático
- [x] Conectar todo al backtester: el sistema opera automáticamente en el período de validación
- [x] Comparar resultados contra los baselines documentados en la Fase 2

> **Por qué la calibración es prerequisito de Kelly:** Kelly asume que `p` es una probabilidad real. Si el modelo dice 0.8 pero la probabilidad real es 0.55, Kelly calcula una posición absurdamente grande. Con probabilidades descalibradas, ni siquiera Half-Kelly es seguro.

**Entregable:** primer número real: ¿supera el Matemático solo al Buy & Hold y al SMA Crossover?

---

### Fase 4 — El Analista de Noticias (Semanas 7-8)

**Objetivo:** añadir la señal de sentimiento y actualizar el Juez para combinar dos agentes.

- [x] Integrar fuente de noticias: **Alpaca News API** (gratuita para datos recientes) — `data/news.py`
- [x] Implementar **El Analista**: pipeline de FinBERT sobre titulares, agregado diario por ticker — `agents/analista.py`
  - Output: score de sentimiento por ticker por día de mercado
  - Manejar la sincronización: noticias de fin de semana se asignan al lunes siguiente
- [x] Actualizar **El Juez**: ahora es un meta-modelo (XGBoost o regresión logística) entrenado sobre [señal_matemático, señal_analista] como features — `backtester/walk_forward.py`
- [x] Re-ejecutar backtesting completo con walk-forward: `python run.py --analista`
- [x] Comparar: ¿añade valor el Analista sobre el Matemático solo? — Sí: Sharpe +0.044, retorno +6.44 pp (ver Memoria.md §15.5)

---

### Fase 5 — El Gestor de Riesgos y El Cazador (Semanas 9-10)

**Objetivo:** añadir los agentes más sencillos en implementación pero muy importantes en valor real.

**El Gestor de Riesgos:**
- [x] El sistema opera **exclusivamente en Long** (comprar y vender). Sin posiciones cortas: evita complejidades de borrowing fees, margin requirements y short squeeze
- [x] Implementar **Fractional Kelly (Half-Kelly)** para el tamaño de posición por ticker:

$$f^* = \rho \left( p - \frac{1-p}{b} \right)$$

  Donde: `p` = probabilidad calibrada de subida, `b` = ratio ganancia media / pérdida media histórico, `ρ = 0.5` (factor de seguridad Half-Kelly)

- [x] **Regla de señal negativa:** si `f* ≤ 0` (el modelo predice más probabilidad de bajada que de subida), la acción es: no abrir posición nueva / cerrar posición existente si la hubiera
- [x] **Límite de concentración por ticker:** cap diferenciado según clase de activo (15% para acciones, 20% para bonos, 10% para oro/internacional) — `get_position_cap(asset_class)`
- [x] **Límite de concentración por clase:** la suma total de cada clase no supera su cap de portfolio (ej: bonos no superan el 30% del capital total) — `normalize_portfolio_by_class()`
- [x] **Gestión multi-posición:** la normalización se aplica en tres niveles: cap por ticker → cap por clase → cap total 100%
- [x] Implementar stop-loss dinámico: salir automáticamente si la pérdida en un ticker supera N×ATR (N configurable en `config.yaml`)
- [x] Integrar como paso final en la cadena de ejecución del Juez: `señal del Juez → Kelly → cap 15% → orden final`

**El Cazador de Insiders:**
- [x] Integrar API de [OpenInsider](https://openinsider.com) (tiene endpoints accesibles)
- [x] Definir reglas condicionales claras: si un CEO/CFO vende >20% de sus acciones en los últimos 30 días → señal bajista con peso fijo
- [x] Lag temporal explícito: las declaraciones Form 4 tienen hasta 2 días hábiles de retraso; aplicar ese lag en el backtest

---

### Fase 6 — El Conspiranoico y Juez v2 (Semanas 11-12)

**Objetivo:** añadir el agente de detección de régimen y, si el sistema base funciona bien, explorar el Juez con RL.

**El Conspiranoico:**
- [ ] Entrenar Isolation Forest sobre features de régimen disponibles con datos diarios de `yfinance`:
  - Volatilidad realizada rolling (std de retornos a 5, 20 y 60 días)
  - VIX descargado como ticker `^VIX` (índice de miedo del mercado)
  - Correlación rolling entre las acciones del universo (alta correlación = movimiento en manada, señal de pánico)
  - Volumen relativo vs media de 20 días (picos de volumen acompañan eventos extremos)
  - Amplitud del mercado: porcentaje de acciones del universo que suben ese día
- [ ] Definir el umbral de activación: cuando anomaly score > X, el agente veta operaciones
- [ ] Validar: ¿el veto en períodos de alta anomalía (ej: marzo 2020, 2022) mejora el Max Drawdown?

**El Juez v2 (opcional, solo si el v1 funciona):**
- [ ] Implementar agente RL (PPO via `stable-baselines3`) con estado = [votos de los agentes] y acción = {comprar, vender, mantener}
- [ ] Diseñar función de recompensa: Sharpe Ratio diferencial (no beneficio bruto, para evitar el "no invertir nunca")
- [ ] Comparar RL vs ensemble estático en validación walk-forward

---

### Fase 7 — Dashboard y XAI (Semanas 13-14)

**Objetivo:** hacer el sistema presentable para portfolio y para reclutadores técnicos.

- [ ] Dashboard en **Streamlit**: curva de capital, estado de cartera, operaciones recientes
- [ ] Panel de "Veredicto del Consejo": muestra en tiempo real los pesos y votos de cada agente
- [ ] Log auditable por operación (formato JSON + visualización en tabla): `{"fecha": "2024-03-15", "ticker": "AAPL", "orden": "COMPRA", "votos": {"matematico": 0.7, "analista": 0.4, "cazador": "silencio", "conspiranoico": "inactivo"}, "kelly_fraction": 0.12}`
- [ ] Modo paper trading: el sistema ingiere datos del día anterior y muestra la decisión del día actual

---

### Fase 8 — Perfiles de Trading (Semanas 15-16)

**Objetivo:** mostrar el mismo sistema operando con tres horizontes distintos. Requiere que la Fase 7 esté completa y el sistema sea estable.

- [ ] Configurar `profiles/long_term.yaml`: SMA 50/200, RSI 28, rebalanceo mensual
- [ ] Adaptar el `BacktestEngine` para leer el parámetro `rebalancing_days` del perfil activo (saltar señales diarias si no toca rebalanceo)
- [ ] Configurar `profiles/day_simulated.yaml`: ejecución Open→Close del mismo día, comisiones + slippage_proxy más altos
- [ ] Adaptar el `BacktestEngine` para el modo `buy_at_open_sell_at_close`
- [ ] Ejecutar backtesting walk-forward para los tres perfiles con las mismas ventanas
- [ ] Añadir pestaña comparativa en el dashboard: curvas de capital superpuestas, tabla de métricas lado a lado
- [ ] Documentar explícitamente las limitaciones del perfil Day Simulated en el dashboard y en el README

**Entregable:** tabla comparativa de métricas (Sharpe, Max DD, Calmar) para los tres perfiles sobre el mismo universo y período.

---

### Fase 9 — El Explorador (Semanas 17-18)

**Objetivo:** monitoreo activo del universo durante el paper trading, con propuestas de cambios supervisadas por el humano.

- [ ] Implementar `Explorador.load_trade_history()`: parsea `logs/trades/*.jsonl` generados por el BacktestEngine
- [ ] Implementar `Explorador.find_removal_candidates()`: detecta activos con f*=0 crónico o accuracy < 50%
- [ ] Implementar `Explorador.evaluate_candidate()`: evalúa un ticker externo (correlación, calidad de datos, liquidez)
- [ ] Implementar `UniverseManager`: aplica cambios aprobados a `data/universe_live.csv` con log auditable
- [ ] Añadir pestaña "El Explorador" en el dashboard de Streamlit:
  - Lista de recomendaciones pendientes (RETIRAR / AÑADIR) con evidencia
  - Botones de aprobación/rechazo para cada recomendación
  - Historial de cambios pasados (`logs/universe_changes.jsonl`)
- [ ] Programar revisión periódica: el Explorador analiza el historial cada 90 días (configurable en `config.yaml`)

**Entregable:** flujo completo de recomendación → revisión humana → cambio aplicado, demostrable en el dashboard.

> **Nota de diseño:** El Explorador solo puede proponer retirar activos que ya tenemos datos de trades (porque el sistema ha operado con ellos). Para añadir activos nuevos, usa `yfinance` para descargar datos de mercado del candidato y calcular correlaciones — pero no usa trades propios para esa decisión porque, por definición, no hemos operado ese activo aún.

> ### 🔔 RECORDATORIO — extensión insiders en El Explorador
>
> **Esto NO forma parte del MVP de Fase 9.** Es una idea acordada para no olvidar al desarrollar esta fase. Implementar solo después de tener el flujo base (RETIRAR / AÑADIR con yfinance) funcionando y **después de Fase 5** (`data/insiders.py` del Cazador).
>
> **Qué es:** usar datos de insiders (OpenInsider / Form 4) como **criterio extra** en `Explorador.evaluate_candidate()` para proponer **AÑADIR** tickers externos a `data/universe_live.csv` — nunca al universo de backtest (`universe_2018-01-01.csv`).
>
> **Qué NO es:** no sustituye al Cazador defensivo (ventas → alerta al Juez en backtest). Las **compras de insiders en tickers ya del universo** son trabajo de **Cazador v2** (columna alcista al Juez), no del Explorador.
>
> **Implementación prevista (cuando toque):**
> - Reutilizar `data/insiders.py` (misma caché que El Cazador; no duplicar scraping).
> - En `evaluate_candidate()`: además de correlación, liquidez y calidad de datos, calcular métricas como `insider_net_buy_pct`, `ceo_buy_flag` (ventana ~90 días, lag Form 4 de 2 días hábiles).
> - Umbrales en `config.yaml` → `explorador.insider_add` (ej. `min_ceo_buy_pct`, `require_net_buy`).
> - Mostrar evidencia insiders en la pestaña dashboard y en `logs/explorador_recommendations.jsonl`.
> - **HITL obligatorio:** insiders solo refuerzan la recomendación; nunca auto-aprueban un ADD.
>
> **Orden sugerido:** Fase 5 Cazador (ventas) → Fase 9 Explorador MVP → esta extensión insiders en ADD → opcionalmente Cazador v2 (compras al Juez).

---

### Fase 10 — Web Portfolio y Demo Pública (Semanas 19-20)

**Objetivo:** hacer el proyecto público, documentado y demostrable para reclutadores.

- [ ] Publicar el dashboard en **Streamlit Cloud** (gratuito) o exportar capturas para la web
- [ ] Página web de portfolio con explicación del sistema, métricas clave y comparativa de perfiles
- [ ] Video demo de 3-5 minutos mostrando: decisión diaria → log auditable → métricas históricas
- [ ] README final pulido: arquitectura, instalación, resultados, limitaciones honestas
- [ ] Código limpio, documentado y con tests pasando en CI (GitHub Actions)

---

## 5. Buenas Prácticas de Ingeniería (No Negociables)

Estas decisiones deben tomarse en la Fase 1, no al final del proyecto.

### config.yaml — Fuente única de verdad

Todos los hiperparámetros y parámetros de configuración viven en un único archivo `config.yaml` versionado con git. **Nunca hardcodear valores en el código.**

```yaml
universe:
  tickers_source: "nasdaq100_top40_2018-01-01.csv"
  max_tickers: 40

data:
  start_date: "2018-01-01"
  end_date: "2025-12-31"
  holdout_start: "2025-01-01"   # Período reservado, no tocar hasta el final

backtester:
  commission_pct: 0.0008         # 0.08% por operación (ida)
  walk_forward_train_years: 3
  walk_forward_val_years: 1

risk_manager:
  kelly_fraction: 0.5            # Half-Kelly (rho)
  max_position_pct: 0.15         # Cap del 15% por ticker
  stop_loss_atr_multiplier: 2.0  # Stop-loss a 2×ATR

matematico:
  n_estimators: 300
  max_depth: 5
  random_state: 42
  calibration_method: "sigmoid"  # "sigmoid" (Platt) o "isotonic"

general:
  random_seed: 42
```

Cada experimento relevante debe guardar una copia del config usado junto a sus resultados, para poder trazar cualquier número a su configuración exacta.

### Reproducibilidad — Seeds fijas en todo el stack

```python
import random, numpy as np, torch

SEED = 42  # Leer de config.yaml

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

> **Por qué importa para el portfolio:** si un reclutador clona el repositorio y ejecuta el backtest, debe obtener exactamente el mismo Sharpe Ratio que aparece en tu web. Si no, la credibilidad del proyecto queda en entredicho.

> **Nota sobre FinBERT:** si se usa solo para inferencia (sin fine-tuning), la seed de torch aplica a la reproducibilidad del sampling. Si en el futuro se hace fine-tuning, las seeds de CUDA son obligatorias.

### Tests mínimos (el backtester es el componente más crítico)

Un bug en el backtester invalida todos los resultados. Tests mínimos obligatorios:

- **Test anti-leakage:** verificar que ninguna feature en el día T usa datos de T+1 o posteriores
- **Test de comisiones:** una estrategia que compra y vende el mismo día debe terminar con pérdidas
- **Test de reproducibilidad:** ejecutar el backtest dos veces con la misma seed debe dar el mismo resultado exacto
- **Test de Kelly:** `f* ≤ 0` nunca debe generar una orden de compra; `f* > 0.15` nunca debe superar el cap

---

## 6. Trampas Técnicas (Reality Check Ampliado)

### Data Leakage — Error nº1, construir el backtester primero lo previene

Ocurre cuando el modelo tiene acceso accidental a datos futuros. Ejemplos concretos a evitar:
- Calcular la media móvil de 20 días **incluyendo el día actual** para decidir la operación de ese día
- Normalizar los datos usando la media/std de **todo el dataset** antes de dividir en train/test
- Usar noticias publicadas a las 18:00 para decidir una operación de apertura del mismo día

**Solución:** el framework de backtesting de la Fase 2, correctamente implementado con walk-forward, hace esto imposible por diseño si se programa bien desde el inicio.

### Sincronización Temporal de Señales

El Matemático opera de lunes a viernes. El Analista genera noticias 7 días a la semana. La regla es:
- Las noticias publicadas fuera de horario de mercado se **acumulan** y se asignan a la siguiente apertura
- El backtester solo toma decisiones en días hábiles de mercado (NYSE calendar via `pandas_market_calendars`)

### Límites de APIs Gratuitas

| API | Límite gratuito | Solución |
|---|---|---|
| yfinance | Sin límite oficial, pero throttled | Caché local en SQLite/Parquet |
| NewsAPI | 100 requests/día, solo 1 mes atrás | Alpaca News para histórico; caché local |
| OpenInsider | Sin API oficial | Scraping con `time.sleep(2)` entre requests |

### Costes de Transacción

El simulador de comisiones **no es opcional**. Sin él, una estrategia que opera 20 veces por semana puede parecer rentable en bruto y ser un desastre neto. Con 0,08% por operación (ida): 20 operaciones semanales × 0,08% × 2 (ida y vuelta) = 3,2% de costes semanales. Ningún edge aguanta eso.

### Overfitting Temporal (el data snooping silencioso)

Si pruebas 50 variantes del modelo y te quedas con la que da mejor Sharpe en el período 2021-2024, has hecho data snooping: el período de "test" ya está contaminado por tus decisiones de diseño. La única solución parcial es el walk-forward y reservar un período final (ej: 2025) que **no se toca hasta que el sistema está terminado**.

---

## 7. Métricas de Evaluación

El éxito del proyecto **no se mide por el beneficio neto**. Estas son las métricas que importan:

| Métrica | Qué mide | Objetivo mínimo razonable |
|---|---|---|
| **Sharpe Ratio** | Retorno ajustado por volatilidad | > 1.0 en validación walk-forward |
| **Maximum Drawdown** | Peor caída desde un máximo | < 25% |
| **Calmar Ratio** | Retorno anual / Max Drawdown | > 0.5 |
| **% Operaciones ganadoras** | Tasa de acierto bruto | Menos importante que Sharpe; informativo |
| **Outperformance vs Buy&Hold** | ¿El sistema añade valor real? | Sharpe MAS > Sharpe Buy&Hold |

> **Expectativa honesta:** si el sistema alcanza Sharpe ~1.0–1.5 con Max Drawdown < 25% en validación walk-forward (no en backtest de entrenamiento), es un resultado técnicamente sólido y creíble para portfolio. Un Sharpe > 2.0 en backtest con un solo split debe tratarse con escepticismo hasta validación adicional.

---

## 8. Stack Tecnológico

| Capa | Tecnología | Justificación |
|---|---|---|
| **Datos** | `yfinance`, `pandas`, `pandas_market_calendars` | Estándar, gratuito, bien mantenido |
| **Almacenamiento** | SQLite (datos pequeños) o Parquet (series largas) | Sin servidor, reproducible |
| **ML / Agentes** | `scikit-learn`, `xgboost`, `transformers` (FinBERT) | Ecosistema maduro |
| **RL (v2)** | `stable-baselines3` + `gymnasium` | Estándar actual para RL en Python |
| **Backtesting** | `vectorbt` o `backtesting.py` + lógica propia | vectorbt es muy rápido para backtests vectorizados |
| **Dashboard** | `Streamlit` | Rápido de desarrollar, aspecto profesional |
| **Logging XAI** | JSON estructurado + `pandas` para visualización | Simple, auditable, sin dependencias extra |

---

## 9. Estructura del Repositorio (objetivo final)

```
trading-mas/
├── config.yaml                          # Fuente única de verdad para todos los parámetros
├── data/
│   ├── raw/                             # Datos descargados sin tocar
│   ├── processed/                       # Features calculadas y limpias
│   ├── cache/                           # Caché de APIs
│   ├── universe_2018-01-01.csv          # Universo BACKTEST — fijado en fecha de inicio, NO TOCAR
│   └── universe_live.csv                # Universo PRODUCCIÓN — evoluciona con El Explorador
├── agents/
│   ├── base_agent.py                    # Interfaz abstracta: fit(), predict(), is_fitted()
│   ├── matematico.py                    # XGBoost binario + calibración
│   ├── analista.py                      # FinBERT + calibración
│   ├── cazador.py                       # OpenInsider + reglas condicionales
│   ├── conspiranoico.py                 # Isolation Forest + HMM (veto de régimen)
│   ├── gestor_riesgos.py                # Fractional Kelly + caps por clase + stop-loss
│   └── explorador.py                    # Gestión dinámica del universo (v2, Fase 9)
├── judge/
│   ├── judge_v1.py                      # Ensemble estático ponderado
│   └── judge_v2_rl.py                   # RL/PPO (Fase 6, opcional)
├── backtester/
│   ├── engine.py                        # Motor de backtesting (soporta perfiles)
│   ├── metrics.py                       # Sharpe, Drawdown, Calmar, reliability diagram
│   └── walk_forward.py                  # Validación walk-forward con ventanas rodantes
├── baselines/
│   ├── buy_and_hold.py
│   └── sma_crossover.py
├── profiles/
│   ├── swing.yaml                       # Perfil por defecto: señales diarias, posiciones días-semanas
│   ├── long_term.yaml                   # Rebalanceo mensual, ventanas largas (SMA 50/200)
│   └── day_simulated.yaml               # Proxy de day trading Open→Close. SOLO académico
├── tests/
│   ├── test_backtester.py               # Anti-leakage, comisiones, reproducibilidad
│   ├── test_kelly.py                    # f*≤0 nunca genera compra; caps por clase
│   └── test_data_integrity.py           # Sin gaps, sin NaN, sin fechas futuras en features
├── dashboard/
│   └── app.py                           # Streamlit (con pestaña comparativa de perfiles y El Explorador)
├── logs/
│   ├── trades/                          # Logs JSON auditables por operación (usados por El Explorador)
│   ├── explorador_recommendations.jsonl # Recomendaciones pendientes de aprobación humana
│   └── universe_changes.jsonl           # Historial auditable de cambios al universo activo
├── utils/
│   ├── config_loader.py                 # Carga config.yaml + overrides de perfil
│   └── reproducibility.py              # Seeds fijas: Python, NumPy, PyTorch
├── notebooks/
│   └── exploration/                     # EDA y experimentos
├── experiments/                         # Copia de config.yaml + resultados por experimento
├── requirements.txt                     # Versiones fijadas
└── README.md
```

---

## 10. Hitos y Criterios de Éxito por Fase

| Fase | Semanas | Criterio de éxito (no avanzar sin cumplirlo) |
|---|---|---|
| 1 — Datos | 1-2 | Los datos de las 46 acciones/ETFs del universo están en local, sin gaps, con features técnicas calculadas y `config.yaml` configurado |
| 2 — Backtester | 3-4 | Los dos baselines tienen resultados documentados y reproducibles |
| 3 — MVP | 5-6 | El Matemático supera el SMA Crossover baseline en walk-forward |
| 4 — Analista | 7-8 | El ensemble Matemático+Analista supera al Matemático solo |
| 5 — Riesgos/Insiders | 9-10 | El Max Drawdown se reduce al añadir el Gestor de Riesgos |
| 6 — Conspiranoico/RL | 11-12 | El Conspiranoico reduce pérdidas en períodos de crisis documentados |
| 7 — Dashboard | 13-14 | Demo funcional: datos → decisión → log auditable → visualización |
| 8 — Perfiles | 15-16 | Tabla comparativa de Sharpe/Drawdown para Swing, Long-term y Day Simulated |
| 9 — El Explorador | 17-18 | Flujo completo: recomendación → aprobación humana → cambio en universe_live.csv |
| 10 — Demo pública | 19-20 | Dashboard público + README final + métricas honestas documentadas |

---

*Última revisión: Junio 2026*
