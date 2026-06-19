"""
Tests unitarios para El Cazador y el pipeline de insiders.

Cobertura:
    - Parsing de Form 4 XML de la SEC (fixtures estáticos)
    - Extracción de roles desde XML de reportingOwnerRelationship
    - Regla CEO vende >20% → alerta = 1
    - Regla Director vende 10% → alerta = 0 (umbral 20%)
    - Lag de Form 4: transacción T visible solo en T + form4_lag_days días hábiles
    - Ticker sin datos → serie de ceros
    - ETFs excluidos del filtro de equity
"""

import pandas as pd
import pytest

from data.insiders import (
    RELEVANT_ROLES,
    _empty_df,
    _extract_role_from_title,
    _parse_form4_xml,
    build_alert_series,
)


# ---------------------------------------------------------------------------
# XML fixtures de Form 4 (estructura real de SEC EDGAR)
# ---------------------------------------------------------------------------

# Form 4 con venta de CEO (25% de sus acciones)
_XML_CEO_SALE = """<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2021-01-04</periodOfReport>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Tim Cook</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2021-01-04</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>S</transactionCode>
        <equitySwapInvolved>0</equitySwapInvolved>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>130.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>30000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""

# Form 4 con venta de Director (5% de sus acciones)
_XML_DIRECTOR_SMALL_SALE = """<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2021-01-05</periodOfReport>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Jane Smith</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>0</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2021-01-05</value></transactionDate>
      <transactionCoding>
        <transactionCode>S</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>131.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>9000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""

# Form 4 con COMPRA (no es señal bajista)
_XML_CFO_PURCHASE = """<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2021-01-06</periodOfReport>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Luca Maestri</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <officerTitle>Chief Financial Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2021-01-06</value></transactionDate>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>132.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>55000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""

# Form 4 sin transacciones (solo derivados que ignoramos)
_XML_EMPTY_NONDERIVATIVE = """<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2021-01-07</periodOfReport>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Nobody</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>0</isOfficer>
      <isDirector>0</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
</ownershipDocument>
"""

# Form 4 con namespace XML (variante real encontrada en EDGAR)
_XML_WITH_NAMESPACE = """<?xml version="1.0"?>
<ownershipDocument xmlns="http://www.sec.gov/cgi-bin/viewer?action=view&cik=320193">
  <periodOfReport>2021-02-01</periodOfReport>
  <reportingOwner>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>President</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2021-02-01</value></transactionDate>
      <transactionCoding>
        <transactionCode>S</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>3000</value></transactionShares>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>7000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


# ---------------------------------------------------------------------------
# Tests: parsing de Form 4 XML
# ---------------------------------------------------------------------------

class TestParseForm4XML:
    _filing_date = pd.Timestamp("2021-01-06")

    def test_parses_ceo_sale_correctly(self):
        records = _parse_form4_xml(_XML_CEO_SALE, self._filing_date)
        assert len(records) == 1
        r = records[0]
        assert set(r.keys()) == {"trade_date", "filing_date", "role",
                                  "transaction_type", "shares_sold_pct"}

    def test_ceo_role_extracted_from_title(self):
        records = _parse_form4_xml(_XML_CEO_SALE, self._filing_date)
        assert records[0]["role"] == "CEO"

    def test_ceo_transaction_type_is_sec_code(self):
        """transaction_type debe ser código SEC ('S'), no texto de OpenInsider."""
        records = _parse_form4_xml(_XML_CEO_SALE, self._filing_date)
        assert records[0]["transaction_type"] == "S"

    def test_ceo_shares_sold_pct(self):
        """10000 vendidas / (10000 + 30000) = 0.25."""
        records = _parse_form4_xml(_XML_CEO_SALE, self._filing_date)
        assert records[0]["shares_sold_pct"] == pytest.approx(0.25, abs=0.001)

    def test_ceo_trade_date(self):
        records = _parse_form4_xml(_XML_CEO_SALE, self._filing_date)
        assert records[0]["trade_date"] == pd.Timestamp("2021-01-04")

    def test_ceo_filing_date_preserved(self):
        records = _parse_form4_xml(_XML_CEO_SALE, self._filing_date)
        assert records[0]["filing_date"] == self._filing_date

    def test_director_role_from_isDirector_flag(self):
        """Cuando no hay officerTitle pero isDirector=1 → rol = 'Director'."""
        records = _parse_form4_xml(_XML_DIRECTOR_SMALL_SALE, self._filing_date)
        assert len(records) == 1
        assert records[0]["role"] == "Director"

    def test_director_shares_sold_pct(self):
        """500 vendidas / (500 + 9000) = 0.0526..."""
        records = _parse_form4_xml(_XML_DIRECTOR_SMALL_SALE, self._filing_date)
        assert records[0]["shares_sold_pct"] == pytest.approx(500 / 9500, abs=0.001)

    def test_purchase_returns_no_records(self):
        """Compra (código P, acquired=A) no debe producir registros."""
        records = _parse_form4_xml(_XML_CFO_PURCHASE, self._filing_date)
        assert records == []

    def test_empty_nonderivative_table_returns_empty(self):
        records = _parse_form4_xml(_XML_EMPTY_NONDERIVATIVE, self._filing_date)
        assert records == []

    def test_namespace_xml_is_handled(self):
        """XML con xmlns declaration no debe fallar el parsing."""
        records = _parse_form4_xml(_XML_WITH_NAMESPACE, self._filing_date)
        assert len(records) == 1
        assert records[0]["role"] == "President"
        assert records[0]["transaction_type"] == "S"

    def test_invalid_xml_returns_empty(self):
        records = _parse_form4_xml("<not valid xml >>>", self._filing_date)
        assert records == []


# ---------------------------------------------------------------------------
# Tests: extracción de rol desde título
# ---------------------------------------------------------------------------

class TestExtractRoleFromTitle:
    @pytest.mark.parametrize("title,expected", [
        ("Chief Executive Officer", "CEO"),
        ("CEO", "CEO"),
        ("Chief Financial Officer", "CFO"),
        ("CFO", "CFO"),
        ("Chief Operating Officer", "COO"),
        ("Chief Technology Officer", "CTO"),
        ("President and CEO", "CEO"),   # CEO tiene preferencia sobre President
        ("Chairman of the Board", "Chairman of the Board"),  # sin coincidencia → retorna original; Chairman usa isDirector=1 en XML
        ("President", "President"),
        ("VP of Marketing", "VP of Marketing"),  # no es rol relevante
    ])
    def test_role_extraction(self, title: str, expected: str):
        assert _extract_role_from_title(title) == expected


# ---------------------------------------------------------------------------
# Tests: build_alert_series (usa códigos SEC "S")
# ---------------------------------------------------------------------------

def _trading_dates(start: str, end: str) -> pd.DatetimeIndex:
    """Genera fechas de trading (lunes a viernes, sin festivos)."""
    return pd.bdate_range(start=start, end=end)


class TestBuildAlertSeries:

    def test_ceo_sells_above_threshold_triggers_alert(self):
        """CEO vende 25% (> umbral 20%) → alerta = 1."""
        insider_df = pd.DataFrame({
            "trade_date": [pd.Timestamp("2021-01-04")],
            "filing_date": [pd.Timestamp("2021-01-04")],
            "role": ["CEO"],
            "transaction_type": ["S"],
            "shares_sold_pct": [0.25],
        })
        trading_dates = _trading_dates("2021-01-04", "2021-02-28")

        alerts = build_alert_series(
            insider_df=insider_df,
            trading_dates=trading_dates,
            sell_threshold_pct=0.20,
            lookback_days=30,
            form4_lag_days=0,
        )

        assert alerts.iloc[0] == 1
        assert alerts.sum() > 0

    def test_director_below_threshold_no_alert(self):
        """Director vende 10% (< umbral 20%) → sin alerta."""
        insider_df = pd.DataFrame({
            "trade_date": [pd.Timestamp("2021-01-04")],
            "filing_date": [pd.Timestamp("2021-01-04")],
            "role": ["Director"],
            "transaction_type": ["S"],
            "shares_sold_pct": [0.10],
        })
        trading_dates = _trading_dates("2021-01-04", "2021-02-28")

        alerts = build_alert_series(
            insider_df=insider_df,
            trading_dates=trading_dates,
            sell_threshold_pct=0.20,
            lookback_days=30,
            form4_lag_days=0,
        )
        assert alerts.sum() == 0

    def test_purchase_does_not_trigger_alert(self):
        """Una compra (código P) de insider no es señal bajista."""
        insider_df = pd.DataFrame({
            "trade_date": [pd.Timestamp("2021-01-04")],
            "filing_date": [pd.Timestamp("2021-01-04")],
            "role": ["CEO"],
            "transaction_type": ["P"],
            "shares_sold_pct": [0.30],
        })
        trading_dates = _trading_dates("2021-01-04", "2021-02-28")

        alerts = build_alert_series(
            insider_df=insider_df,
            trading_dates=trading_dates,
            sell_threshold_pct=0.20,
            lookback_days=30,
            form4_lag_days=0,
        )
        assert alerts.sum() == 0

    def test_form4_lag_2_business_days(self):
        """
        Transacción en 2021-01-04 (lunes) con lag=2 días hábiles.
        La señal debe ser visible a partir del 2021-01-06 (miércoles).
        En 2021-01-04 y 2021-01-05 no debe haber alerta.
        """
        insider_df = pd.DataFrame({
            "trade_date": [pd.Timestamp("2021-01-04")],
            "filing_date": [pd.Timestamp("2021-01-04")],
            "role": ["CEO"],
            "transaction_type": ["S"],
            "shares_sold_pct": [0.25],
        })
        trading_dates = pd.bdate_range("2021-01-04", "2021-01-06")

        alerts = build_alert_series(
            insider_df=insider_df,
            trading_dates=trading_dates,
            sell_threshold_pct=0.20,
            lookback_days=30,
            form4_lag_days=2,
        )

        assert alerts.loc[pd.Timestamp("2021-01-04")] == 0
        assert alerts.loc[pd.Timestamp("2021-01-05")] == 0
        assert alerts.loc[pd.Timestamp("2021-01-06")] == 1

    def test_empty_insider_df_all_zeros(self):
        """Sin datos de insiders → serie de ceros."""
        trading_dates = _trading_dates("2021-01-04", "2021-01-31")
        alerts = build_alert_series(
            insider_df=pd.DataFrame(),
            trading_dates=trading_dates,
            sell_threshold_pct=0.20,
            lookback_days=30,
            form4_lag_days=2,
        )
        assert (alerts == 0).all()
        assert len(alerts) == len(trading_dates)

    def test_alert_persists_within_lookback_window(self):
        """La alerta persiste durante la ventana lookback de 30 días."""
        insider_df = pd.DataFrame({
            "trade_date": [pd.Timestamp("2021-01-04")],
            "filing_date": [pd.Timestamp("2021-01-04")],
            "role": ["CFO"],
            "transaction_type": ["S"],
            "shares_sold_pct": [0.22],
        })
        trading_dates = _trading_dates("2021-01-04", "2021-03-01")

        alerts = build_alert_series(
            insider_df=insider_df,
            trading_dates=trading_dates,
            sell_threshold_pct=0.20,
            lookback_days=30,
            form4_lag_days=0,
        )

        assert alerts.sum() >= 15
        alerts_after = alerts[alerts.index > pd.Timestamp("2021-02-15")]
        assert alerts_after.sum() == 0

    def test_irrelevant_role_no_alert(self):
        """Un 'VP of Marketing' vendiendo no dispara la alerta (no está en RELEVANT_ROLES)."""
        insider_df = pd.DataFrame({
            "trade_date": [pd.Timestamp("2021-01-04")],
            "filing_date": [pd.Timestamp("2021-01-04")],
            "role": ["VP of Marketing"],
            "transaction_type": ["S"],
            "shares_sold_pct": [0.35],
        })
        trading_dates = _trading_dates("2021-01-04", "2021-01-31")

        alerts = build_alert_series(
            insider_df=insider_df,
            trading_dates=trading_dates,
            sell_threshold_pct=0.20,
            lookback_days=30,
            form4_lag_days=0,
        )
        assert alerts.sum() == 0

    def test_sale_minus_code_triggers_alert(self):
        """Código 'S-' (venta exenta) también dispara la alerta."""
        insider_df = pd.DataFrame({
            "trade_date": [pd.Timestamp("2021-01-04")],
            "filing_date": [pd.Timestamp("2021-01-04")],
            "role": ["CEO"],
            "transaction_type": ["S-"],
            "shares_sold_pct": [0.30],
        })
        trading_dates = _trading_dates("2021-01-04", "2021-01-31")

        alerts = build_alert_series(
            insider_df=insider_df,
            trading_dates=trading_dates,
            sell_threshold_pct=0.20,
            lookback_days=30,
            form4_lag_days=0,
        )
        assert alerts.sum() > 0


# ---------------------------------------------------------------------------
# Tests: Cazador agent
# ---------------------------------------------------------------------------

class TestCazadorAgent:
    def _make_cfg(self):
        return {
            "cazador": {
                "sell_threshold_pct": 0.20,
                "lookback_days": 30,
                "form4_lag_days": 2,
                "sec_user_agent": "Test User test@example.com",
            },
            "data": {
                "start_date": "2018-01-01",
                "end_date": "2025-01-01",
                "cache_dir": "data/cache",
            },
            "universe": {
                "tickers_file": "data/universe_2018-01-01.csv",
            },
        }

    def test_fit_is_noop(self):
        """El Cazador no se entrena; fit() no debe fallar."""
        from agents.cazador import Cazador
        agent = Cazador(self._make_cfg())
        agent.fit(pd.DataFrame())
        assert agent.is_fitted()

    def test_predict_ticker_returns_binary_series(self):
        """predict_ticker retorna Serie binaria 0/1 con datos de insider en formato SEC."""
        from agents.cazador import Cazador
        agent = Cazador(self._make_cfg())

        agent._insider_data["AAPL"] = pd.DataFrame({
            "trade_date": [pd.Timestamp("2021-01-04")],
            "filing_date": [pd.Timestamp("2021-01-04")],
            "role": ["CEO"],
            "transaction_type": ["S"],          # código SEC, no texto OpenInsider
            "shares_sold_pct": [0.25],
        })

        data = pd.DataFrame(
            index=_trading_dates("2021-01-04", "2021-01-31"),
            data={"close": 130.0},
        )
        preds = agent.predict_ticker(data, "AAPL")

        assert isinstance(preds, pd.Series)
        assert set(preds.unique()).issubset({0, 1})
        assert len(preds) == len(data)

    def test_predict_ticker_etf_no_data_all_zeros(self):
        """Para un ETF sin datos de insider → serie de ceros."""
        from agents.cazador import Cazador
        agent = Cazador(self._make_cfg())
        agent._insider_data["TLT"] = pd.DataFrame()

        data = pd.DataFrame(
            index=_trading_dates("2021-01-04", "2021-01-31"),
            data={"close": 100.0},
        )
        preds = agent.predict_ticker(data, "TLT")
        assert (preds == 0).all()
