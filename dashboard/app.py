"""
Dashboard interactivo del sistema MAS — Streamlit.

Ejecutar con: streamlit run dashboard/app.py

Secciones del dashboard (implementar en Fase 7):
    1. Resumen del portfolio: curva de capital, métricas principales
    2. Veredicto del Consejo: votos y pesos de cada agente por ticker/fecha
    3. Historial de operaciones: log auditable con filtros
    4. Paper trading: señal actual del sistema para el día de hoy
    5. Comparativa de baselines: MAS vs Buy & Hold vs SMA Crossover
"""

import streamlit as st

st.set_page_config(
    page_title="Trading MAS — Panel de Control",
    page_icon="📊",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📊 Trading MAS — Panel de Control")
st.caption("Sistema Multi-Agente de Predicción Bursátil")

# ── Placeholder hasta Fase 7 ──────────────────────────────────────────────────

st.info(
    "🚧 **Dashboard en construcción** — Disponible a partir de la Fase 7.\n\n"
    "Una vez implementado, este panel mostrará:\n"
    "- Curva de capital del sistema vs baselines\n"
    "- Veredicto del Consejo de Agentes en tiempo real\n"
    "- Log auditable de cada operación con desglose de votos\n"
    "- Señal actual del sistema para el día de hoy (paper trading)"
)

st.divider()

# ── Estructura del dashboard (esqueleto para Fase 7) ─────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Portfolio",
    "🧠 Veredicto del Consejo",
    "📋 Historial de Operaciones",
    "🔴 Paper Trading (Hoy)",
])

with tab1:
    st.subheader("Curva de Capital")
    # TODO: cargar equity_curve de logs/ y renderizar con plotly
    st.write("_Implementar en Fase 7_")

with tab2:
    st.subheader("Votos del Consejo de Agentes")
    # TODO: cargar último log de operaciones y mostrar desglose de votos
    # Ejemplo de formato:
    # | Agente        | Señal | Peso  |
    # |---------------|-------|-------|
    # | El Matemático | 0.72  | 60%   |
    # | El Analista   | 0.41  | 30%   |
    # | El Cazador    | —     | 10%   |
    # | El Conspiranoico | inactivo | veto |
    st.write("_Implementar en Fase 7_")

with tab3:
    st.subheader("Historial de Operaciones")
    # TODO: cargar logs/trades/*.jsonl y mostrar en tabla filtrable
    st.write("_Implementar en Fase 7_")

with tab4:
    st.subheader("Señal del Sistema para Hoy")
    # TODO: ejecutar pipeline con datos del día anterior y mostrar decisión
    st.write("_Implementar en Fase 7_")
