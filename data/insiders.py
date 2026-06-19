"""
Pipeline de descarga, cache y procesamiento de transacciones de insiders (SEC Form 4).

Fuente: SEC EDGAR (API oficial y gratuita).
Documentación: https://www.sec.gov/developer
               https://efts.sec.gov/LATEST/search-index

Flujo:
    1. Resolver ticker → CIK via company_tickers.json de la SEC
    2. Obtener lista de Form 4 via submissions API de EDGAR (con paginación)
    3. Parsear XML de cada filing para extraer transacciones relevantes
    4. Cachear en Parquet por ticker (data/cache/insiders/)

Lag temporal (anti-leakage):
    Las declaraciones Form 4 tienen hasta 2 días hábiles de plazo desde
    la transacción. El lag se aplica sobre días de negociación NYSE para
    evitar usar información que el mercado real no tendría disponible.

    Transacción en día T → señal visible a partir de T + form4_lag_days días hábiles.

Requisito SEC (política de fair access):
    El header User-Agent DEBE incluir nombre y email de contacto.
    Configura en config.yaml:
        cazador:
            sec_user_agent: "TuNombre email@dominio.com"

    Sin User-Agent válido, la SEC puede bloquear la IP temporalmente (~10 min).
    Rate limit: máximo 10 req/s. Este módulo usa 0.12s de pausa (~8 req/s).
"""

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.config_loader import load_config

logger = logging.getLogger(__name__)

# SEC EDGAR endpoints
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_SEC_SUBMISSIONS_OLDER_URL = "https://data.sec.gov/submissions/{filename}"
_SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{document}"

_REQUEST_DELAY_S = 0.12   # ~8 req/s, seguro bajo el límite de la SEC
_MAX_RETRIES = 3

# Roles cuyas ventas son señal bajista relevante
RELEVANT_ROLES = {"CEO", "CFO", "Director", "President", "COO", "CTO"}

# Códigos de transacción del Form 4 de la SEC que indican una venta
# S = venta en mercado abierto, S- = venta exenta (Section 16)
_SALE_CODES = {"S", "S-"}


# ---------------------------------------------------------------------------
# Helpers de red
# ---------------------------------------------------------------------------

def _get_headers(config: dict) -> dict[str, str]:
    """
    Devuelve el header User-Agent requerido por la SEC.

    La política de fair access de la SEC exige un identificador de contacto.
    Ver: https://www.sec.gov/os/accessing-edgar-data

    Raises:
        ValueError: Si sec_user_agent no está configurado.
    """
    ua = config.get("cazador", {}).get("sec_user_agent", "").strip()
    if not ua:
        raise ValueError(
            "La SEC requiere un User-Agent con nombre y email. "
            "Añade a config.yaml:\n"
            "  cazador:\n"
            "    sec_user_agent: \"TuNombre email@dominio.com\""
        )
    return {"User-Agent": ua}


def _get(url: str, headers: dict, timeout: int = 30) -> bytes | None:
    """
    GET con reintentos y backoff exponencial.

    Respeta la política de rate-limiting de la SEC:
    espera _REQUEST_DELAY_S entre requests y hace backoff en 429/503.
    """
    try:
        import requests as req_lib
    except ImportError:
        logger.error("requests no instalado. Ejecuta: pip install requests")
        return None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = req_lib.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.content
            if resp.status_code in (429, 503):
                wait = 2 ** (attempt + 1) * 5  # 10s, 20s, 40s
                logger.warning("Rate limit SEC (%d), esperando %ds...", resp.status_code, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                logger.debug("No encontrado: %s", url)
                return None
            logger.warning("HTTP %d: %s", resp.status_code, url)
            return None
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                logger.error("Error descargando %s: %s", url, exc)
                return None
            time.sleep(1)
    return None


# ---------------------------------------------------------------------------
# Resolución ticker → CIK
# ---------------------------------------------------------------------------

def _load_cik_map(config: dict) -> dict[str, int]:
    """
    Descarga y cachea el mapa ticker → CIK de la SEC.

    Cachea en data/cache/insiders/_cik_map.json para no re-descargarlo
    en cada run.

    Returns:
        {ticker_upper: cik_int}
    """
    cache_dir = Path(config["data"]["cache_dir"]) / "insiders"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "_cik_map.json"

    if cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    headers = _get_headers(config)
    raw = _get(_SEC_TICKERS_URL, headers)
    if raw is None:
        logger.error("No se pudo descargar el mapa de CIKs de la SEC.")
        return {}

    data = json.loads(raw)
    cik_map: dict[str, int] = {
        v["ticker"].upper(): int(v["cik_str"])
        for v in data.values()
        if "ticker" in v and "cik_str" in v
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cik_map, f)

    logger.debug("Mapa CIK descargado: %d tickers", len(cik_map))
    return cik_map


# ---------------------------------------------------------------------------
# Descarga de lista de filings (submissions API)
# ---------------------------------------------------------------------------

def _fetch_form4_filing_list(
    cik: int,
    headers: dict,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Obtiene la lista de Form 4 filings para un CIK en el rango de fechas.

    Pagina automáticamente a través de los archivos de submissions más
    antiguos si el rango solicitado supera los ~1000 filings recientes.

    Returns:
        Lista de dicts con: {filing_date, accession_number, primary_document}
    """
    url = _SEC_SUBMISSIONS_URL.format(cik=cik)
    raw = _get(url, headers)
    if raw is None:
        return []

    try:
        data = json.loads(raw)
    except Exception:
        return []

    results: list[dict] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    def _extract_filings(filings_block: dict) -> list[dict]:
        forms = filings_block.get("form", [])
        dates = filings_block.get("filingDate", [])
        accessions = filings_block.get("accessionNumber", [])
        docs = filings_block.get("primaryDocument", [])

        out = []
        for form, date, acc, doc in zip(forms, dates, accessions, docs):
            if form != "4":
                continue
            try:
                ts = pd.Timestamp(date)
            except Exception:
                continue
            if ts < start_ts or ts > end_ts:
                continue
            out.append({
                "filing_date": ts,
                "accession_number": acc,
                "primary_document": doc,
            })
        return out

    recent = data.get("filings", {}).get("recent", {})
    results.extend(_extract_filings(recent))

    # Paginar archivos más antiguos si los hay
    older_files = data.get("filings", {}).get("files", [])
    for older_file in older_files:
        filename = older_file.get("name", "")
        if not filename:
            continue

        older_url = _SEC_SUBMISSIONS_OLDER_URL.format(filename=filename)
        time.sleep(_REQUEST_DELAY_S)
        older_raw = _get(older_url, headers)
        if older_raw is None:
            continue
        try:
            older_data = json.loads(older_raw)
            results.extend(_extract_filings(older_data))
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Parseo de Form 4 XML
# ---------------------------------------------------------------------------

def _strip_namespaces(xml_text: str) -> str:
    """Elimina declaraciones xmlns del XML para simplificar el parsing."""
    # Eliminar atributos xmlns y xmlns:prefix
    xml_text = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", xml_text)
    # Eliminar prefijos de namespace en los tags
    xml_text = re.sub(r"<(/?)[\w-]+:", r"<\1", xml_text)
    return xml_text


def _elem_text(root: ET.Element, *paths: str) -> str | None:
    """
    Busca texto en un elemento XML recorriendo varias rutas posibles.

    Intenta múltiples variantes para manejar diferencias entre versiones
    del esquema XML de Form 4.
    """
    for path in paths:
        try:
            elem = root.find(path)
            if elem is not None and elem.text and elem.text.strip():
                return elem.text.strip()
        except Exception:
            pass
    return None


def _parse_form4_xml(xml_text: str, filing_date: pd.Timestamp) -> list[dict]:
    """
    Parsea el XML de un Form 4 y extrae las transacciones de insider.

    Extrae solo transacciones de tabla no-derivativa (acciones directas),
    ignorando derivados como opciones y warrants.

    Args:
        xml_text: Contenido XML del Form 4.
        filing_date: Fecha de declaración (del índice de submissions de la SEC).
                     Usada para el lag anti-leakage.

    Returns:
        Lista de dicts con: trade_date, filing_date, role,
        transaction_type (código SEC), shares_sold_pct.
    """
    try:
        clean = _strip_namespaces(xml_text)
        root = ET.fromstring(clean)
    except ET.ParseError as exc:
        logger.debug("ParseError en Form 4 XML: %s", exc)
        return []

    # Extraer rol del insider
    role = _extract_role_from_xml(root)

    records = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        record = _parse_single_transaction(txn, filing_date, role)
        if record is not None:
            records.append(record)

    return records


def _extract_role_from_xml(root: ET.Element) -> str:
    """
    Extrae el rol relevante del reportingOwnerRelationship.

    Preferencia: officerTitle > isDirector > isOfficer genérico.
    """
    # Buscar officerTitle primero
    title = _elem_text(
        root,
        ".//reportingOwnerRelationship/officerTitle",
        ".//officerTitle",
    )
    if title:
        return _extract_role_from_title(title)

    # Fallback: isDirector
    is_director = _elem_text(root, ".//reportingOwnerRelationship/isDirector", ".//isDirector")
    if is_director == "1":
        return "Director"

    # Fallback: isOfficer sin título
    is_officer = _elem_text(root, ".//reportingOwnerRelationship/isOfficer", ".//isOfficer")
    if is_officer == "1":
        return "Officer"

    return "Unknown"


def _extract_role_from_title(title: str) -> str:
    """
    Normaliza un título de insider a un rol de RELEVANT_ROLES.

    Ejemplos: "Chief Executive Officer" → "CEO", "Chief Financial Officer" → "CFO".
    """
    title_upper = title.upper()

    # Coincidencias exactas primero
    for role in RELEVANT_ROLES:
        if role in title_upper:
            return role

    # Variantes de texto completo
    patterns = {
        "CEO": ["CHIEF EXECUTIVE", "C.E.O"],
        "CFO": ["CHIEF FINANCIAL", "C.F.O"],
        "COO": ["CHIEF OPERATING", "C.O.O"],
        "CTO": ["CHIEF TECHNOLOGY", "CHIEF TECHNICAL", "C.T.O"],
        "President": ["PRESIDENT"],
        "Director": ["DIRECTOR"],
    }
    for role, patterns_list in patterns.items():
        for pat in patterns_list:
            if pat in title_upper:
                return role

    return title  # Retornar original si no hay coincidencia


def _parse_single_transaction(
    txn: ET.Element,
    filing_date: pd.Timestamp,
    role: str,
) -> dict | None:
    """
    Parsea una transacción individual del nodo nonDerivativeTransaction.

    Returns:
        Dict con campos del insider o None si no es una venta válida.
    """
    # Código de transacción (S = sale, P = purchase, etc.)
    code = _elem_text(txn, ".//transactionCode", "transactionCoding/transactionCode")
    if code not in _SALE_CODES:
        return None

    # Verificar que es una disposición (D = disposed) no una adquisición
    acquired_disposed = _elem_text(
        txn,
        ".//transactionAcquiredDisposedCode/value",
        ".//transactionAcquiredDisposedCode",
    )
    if acquired_disposed and acquired_disposed.upper() not in ("D", "DISPOSED"):
        return None  # Adquisición, no venta

    # Fecha de la transacción
    trade_date_raw = _elem_text(txn, ".//transactionDate/value", ".//transactionDate")
    try:
        trade_date = pd.Timestamp(trade_date_raw)
    except Exception:
        trade_date = filing_date  # Fallback a fecha de declaración

    # Número de acciones vendidas
    shares_raw = _elem_text(txn, ".//transactionShares/value", ".//transactionShares")
    try:
        shares_sold = float(shares_raw.replace(",", ""))
    except Exception:
        return None  # Sin información de acciones, ignorar

    if shares_sold <= 0:
        return None

    # Acciones tras la transacción (para calcular % vendido)
    shares_after_raw = _elem_text(
        txn,
        ".//sharesOwnedFollowingTransaction/value",
        ".//sharesOwnedFollowingTransaction",
    )
    try:
        shares_after = float(shares_after_raw.replace(",", ""))
    except Exception:
        shares_after = 0.0

    # % vendido respecto al total previo
    shares_before = shares_after + shares_sold
    if shares_before <= 0:
        shares_sold_pct = 0.0
    else:
        shares_sold_pct = shares_sold / shares_before

    return {
        "trade_date": trade_date,
        "filing_date": filing_date,
        "role": role,
        "transaction_type": code,   # "S" o "S-"
        "shares_sold_pct": round(shares_sold_pct, 6),
    }


# ---------------------------------------------------------------------------
# Descarga por ticker
# ---------------------------------------------------------------------------

def download_insider_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Descarga transacciones de insiders para un ticker desde SEC EDGAR.

    Returns:
        DataFrame con columnas: trade_date, filing_date, role,
        transaction_type, shares_sold_pct.
        Vacío si no hay datos o falla la descarga.
    """
    if config is None:
        config = load_config()

    try:
        headers = _get_headers(config)
    except ValueError as exc:
        logger.error(str(exc))
        return _empty_df()

    cik_map = _load_cik_map(config)
    cik = cik_map.get(ticker.upper())
    if cik is None:
        logger.warning("%s: ticker no encontrado en el mapa CIK de la SEC", ticker)
        return _empty_df()

    filings = _fetch_form4_filing_list(cik, headers, start_date, end_date)
    if not filings:
        logger.debug("%s: sin Form 4 en el rango %s → %s", ticker, start_date, end_date)
        return _empty_df()

    logger.debug("%s: %d Form 4 filings a parsear", ticker, len(filings))

    all_records: list[dict] = []
    for filing in filings:
        acc = filing["accession_number"].replace("-", "")
        doc = filing["primary_document"]
        url = _SEC_ARCHIVES_URL.format(cik=cik, accession_clean=acc, document=doc)

        time.sleep(_REQUEST_DELAY_S)
        raw = _get(url, headers)
        if raw is None:
            continue

        try:
            xml_text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue

        records = _parse_form4_xml(xml_text, filing["filing_date"])
        all_records.extend(records)

    if not all_records:
        return _empty_df()

    df = pd.DataFrame(all_records)

    # Deduplicar: mismo insider puede declarar la misma venta en múltiples filings
    df = df.drop_duplicates(subset=["trade_date", "role", "transaction_type", "shares_sold_pct"])

    # Filtrar al rango de fechas
    mask = (df["trade_date"] >= pd.Timestamp(start_date)) & (
        df["trade_date"] <= pd.Timestamp(end_date)
    )
    return df[mask].sort_values("trade_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Descarga bulk con caché
# ---------------------------------------------------------------------------

def download_all_insiders(
    tickers: list[str],
    config: dict | None = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Descarga y cachea transacciones de insiders para una lista de tickers.

    Solo descarga tickers de tipo 'equity' (los ETFs no tienen Form 4 relevante).
    Usa caché Parquet en data/cache/insiders/ para evitar re-descargar.

    Returns:
        {ticker: DataFrame} con transacciones de insiders.
    """
    if config is None:
        config = load_config()

    # Validar User-Agent antes de empezar
    try:
        _get_headers(config)
    except ValueError as exc:
        logger.error(str(exc))
        return {}

    cache_dir = Path(config["data"]["cache_dir"]) / "insiders"
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    equity_tickers = _filter_equity_tickers(tickers, config)
    logger.info(
        "Insiders (SEC EDGAR): %d tickers equity (de %d total)",
        len(equity_tickers), len(tickers),
    )

    result: dict[str, pd.DataFrame] = {}
    download_needed: list[str] = []

    # Cargar desde caché los que ya están descargados
    for ticker in equity_tickers:
        cache_file = cache_dir / f"{ticker}.parquet"
        if cache_file.exists() and not force_download:
            try:
                df = pd.read_parquet(cache_file)
                # Solo usar caché si tiene filas (descarga anterior fallida → re-descargar)
                if not df.empty or not force_download:
                    result[ticker] = df
                    continue
            except Exception:
                pass
        download_needed.append(ticker)

    if download_needed:
        logger.info(
            "Descargando Form 4 de SEC EDGAR para %d tickers...",
            len(download_needed),
        )
        for ticker in tqdm(download_needed, desc="Form 4", unit="ticker"):
            df = download_insider_ticker(ticker, start, end, config)
            cache_file = cache_dir / f"{ticker}.parquet"
            df.to_parquet(cache_file, index=False)
            result[ticker] = df
            n = len(df)
            if n > 0:
                logger.debug("%s: %d transacciones de insiders", ticker, n)
    else:
        logger.info("Insiders: todos los tickers en caché.")

    return result


def load_cached_insiders(
    tickers: list[str],
    config: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Carga transacciones de insiders desde caché Parquet sin descargar.

    Returns:
        {ticker: DataFrame} para tickers con caché disponible.
    """
    if config is None:
        config = load_config()

    cache_dir = Path(config["data"]["cache_dir"]) / "insiders"
    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        cache_file = cache_dir / f"{ticker}.parquet"
        if cache_file.exists():
            try:
                result[ticker] = pd.read_parquet(cache_file)
            except Exception as exc:
                logger.warning("Error cargando caché insiders %s: %s", ticker, exc)

    return result


# ---------------------------------------------------------------------------
# Construcción de la señal de alerta
# ---------------------------------------------------------------------------

def build_alert_series(
    insider_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    sell_threshold_pct: float,
    lookback_days: int,
    form4_lag_days: int,
    ticker: str = "",
) -> pd.Series:
    """
    Construye la serie binaria de alertas de insider para un ticker.

    Algoritmo por fecha de trading:
        1. Calcula la ventana de búsqueda: [fecha - lookback_days, fecha]
           sobre los datos ya desplazados por el lag de Form 4.
        2. Filtra ventas por roles relevantes que superen sell_threshold_pct.
        3. Si hay alguna venta relevante → alerta = 1, sino 0.

    El lag se aplica sobre las fechas de declaración (filing_date):
    una transacción en día T solo es visible a partir de T + form4_lag_days
    días hábiles. Esto evita usar información que el mercado no tendría.

    Args:
        insider_df: DataFrame con columnas [trade_date, filing_date, role,
                    transaction_type, shares_sold_pct].
        trading_dates: Días hábiles NYSE del período de validación.
        sell_threshold_pct: Umbral de venta (0.20 = 20% del paquete).
        lookback_days: Ventana de búsqueda en días naturales.
        form4_lag_days: Lag de Form 4 en días hábiles.
        ticker: Nombre del ticker (solo para logging).

    Returns:
        pd.Series binaria con índice = trading_dates.
    """
    alertas = pd.Series(0, index=trading_dates, dtype=int)

    if insider_df.empty:
        return alertas

    # Filtrar solo ventas de roles relevantes por encima del umbral
    sales = insider_df[
        insider_df["transaction_type"].isin(_SALE_CODES)
        & insider_df["role"].isin(RELEVANT_ROLES)
        & (insider_df["shares_sold_pct"] >= sell_threshold_pct)
    ].copy()

    if sales.empty:
        return alertas

    # Aplicar lag: la señal es visible form4_lag_days días hábiles después
    # del filing_date (cuando el Form 4 es público en EDGAR)
    sales["signal_date"] = sales["filing_date"].apply(
        lambda d: _add_business_days(d, form4_lag_days, trading_dates),
    )
    sales = sales.dropna(subset=["signal_date"])

    if sales.empty:
        return alertas

    lookback_td = pd.Timedelta(days=lookback_days)

    for fecha in trading_dates:
        window_start = fecha - lookback_td
        relevant = sales[
            (sales["signal_date"] >= window_start)
            & (sales["signal_date"] <= fecha)
        ]
        if not relevant.empty:
            alertas.loc[fecha] = 1

    n_alerts = int(alertas.sum())
    if n_alerts > 0:
        logger.debug("%s: %d días con alerta insider", ticker, n_alerts)

    return alertas


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["trade_date", "filing_date", "role", "transaction_type", "shares_sold_pct"],
    )


def _add_business_days(
    date: pd.Timestamp,
    n: int,
    trading_dates: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """
    Devuelve el día hábil N posiciones después de date en el calendario NYSE.

    Si date no está en trading_dates, usa el siguiente día hábil como base.
    Retorna None si no hay suficientes fechas en el calendario.
    """
    if pd.isna(date):
        return None

    future = trading_dates[trading_dates >= date]
    if future.empty:
        return None

    idx = trading_dates.get_loc(future[0])
    target_idx = idx + n
    if target_idx >= len(trading_dates):
        return None

    return trading_dates[target_idx]


def _filter_equity_tickers(tickers: list[str], config: dict) -> list[str]:
    """
    Filtra solo los tickers con asset_class == 'equity'.

    Los ETFs (TLT, GLD, EFA, etc.) no tienen directivos que declaren Form 4.
    """
    try:
        tickers_file = Path(config["universe"]["tickers_file"])
        df = pd.read_csv(tickers_file)
        equity_set = set(df[df["asset_class"] == "equity"]["ticker"].tolist())
        return [t for t in tickers if t in equity_set]
    except Exception as exc:
        logger.warning(
            "No se pudo leer asset_class del CSV para filtrar equities: %s. "
            "Descargando insiders para todos los tickers.",
            exc,
        )
        return tickers
