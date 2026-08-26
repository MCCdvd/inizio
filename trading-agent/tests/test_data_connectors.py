"""
Unit tests for data_connectors module.

Covers:
- YahooConnector.fetch_bars (mocked yfinance)
- IBKRConnector initialization and fetch_bars (mocked ib-insync)
- ConnectorFactory routing
- Fallback behaviour in TradingEnvironmentWithVolumeProfile.load_data
"""
import sys
import types
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers – inject a minimal fake ib_insync so tests run without the package
# ---------------------------------------------------------------------------

def _make_fake_ib_insync():
    """Return a minimal ib_insync stub module."""
    mod = types.ModuleType("ib_insync")

    class FakeIB:
        def connect(self, *a, **kw): pass
        def disconnect(self): pass
        def qualifyContracts(self, *a): pass
        def reqHistoricalData(self, *a, **kw): return []

    class FakeStock:
        def __init__(self, *a, **kw): pass

    mod.IB = FakeIB
    mod.Stock = FakeStock
    mod.util = MagicMock()
    mod.util.df = lambda bars: pd.DataFrame()
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def add_src_to_path(monkeypatch):
    """Ensure trading-agent/src is importable."""
    import os
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    src = os.path.abspath(src)
    if src not in sys.path:
        monkeypatch.syspath_prepend(src)


@pytest.fixture()
def fake_ib_insync(monkeypatch):
    """Inject fake ib_insync into sys.modules."""
    fake = _make_fake_ib_insync()
    monkeypatch.setitem(sys.modules, "ib_insync", fake)
    return fake


# ---------------------------------------------------------------------------
# YahooConnector tests
# ---------------------------------------------------------------------------

class TestYahooConnector:
    def test_returns_arrays(self):
        from data_connectors import YahooConnector

        sample = pd.DataFrame({
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1000.0, 2000.0, 3000.0],
        })
        with patch("data_connectors.yf.download", return_value=sample):
            connector = YahooConnector()
            prices, volumes = connector.fetch_bars("AAPL", "2023-01-01", "2023-01-04")

        assert isinstance(prices, np.ndarray)
        assert isinstance(volumes, np.ndarray)
        assert prices.dtype == np.float64
        np.testing.assert_array_equal(prices, [100.0, 101.0, 102.0])
        np.testing.assert_array_equal(volumes, [1000.0, 2000.0, 3000.0])

    def test_empty_download_returns_empty_arrays(self):
        from data_connectors import YahooConnector

        with patch("data_connectors.yf.download", return_value=pd.DataFrame()):
            connector = YahooConnector()
            prices, volumes = connector.fetch_bars("AAPL", "2023-01-01", "2023-01-04")

        assert len(prices) == 0
        assert len(volumes) == 0

    def test_none_download_returns_empty_arrays(self):
        from data_connectors import YahooConnector

        with patch("data_connectors.yf.download", return_value=None):
            connector = YahooConnector()
            prices, volumes = connector.fetch_bars("AAPL", "2023-01-01", "2023-01-04")

        assert len(prices) == 0
        assert len(volumes) == 0


# ---------------------------------------------------------------------------
# IBKRConnector tests
# ---------------------------------------------------------------------------

class TestIBKRConnector:
    def test_init_defaults(self):
        from data_connectors import IBKRConnector

        c = IBKRConnector()
        assert c.host == "127.0.0.1"
        assert c.port == 7497
        assert c.client_id == 11
        assert c.timeframe == "1 day"

    def test_init_custom(self):
        from data_connectors import IBKRConnector

        c = IBKRConnector(host="10.0.0.1", port=4002, client_id=5, timeframe="1 hour")
        assert c.host == "10.0.0.1"
        assert c.port == 4002
        assert c.client_id == 5
        assert c.timeframe == "1 hour"

    def test_fetch_bars_returns_data(self, fake_ib_insync):
        from data_connectors import IBKRConnector

        sample_df = pd.DataFrame({
            "close": [150.0, 151.0],
            "volume": [5000.0, 6000.0],
        })
        fake_ib_insync.util.df = lambda bars: sample_df

        ib_instance = MagicMock()
        ib_instance.connect = MagicMock()
        ib_instance.disconnect = MagicMock()
        ib_instance.qualifyContracts = MagicMock()
        ib_instance.reqHistoricalData = MagicMock(return_value=["bar1", "bar2"])
        fake_ib_insync.IB = lambda: ib_instance

        connector = IBKRConnector()
        prices, volumes = connector.fetch_bars("AAPL", "2023-01-01", "2023-01-03")

        assert len(prices) == 2
        assert prices[0] == 150.0
        assert volumes[1] == 6000.0

    def test_fetch_bars_raises_on_bad_dates(self, fake_ib_insync):
        from data_connectors import IBKRConnector

        connector = IBKRConnector()
        with pytest.raises(ValueError, match="end_date"):
            connector.fetch_bars("AAPL", "2023-01-05", "2023-01-01")

    def test_fetch_bars_empty_response_returns_empty(self, fake_ib_insync):
        from data_connectors import IBKRConnector

        fake_ib_insync.util.df = lambda bars: pd.DataFrame()
        ib_instance = MagicMock()
        ib_instance.reqHistoricalData = MagicMock(return_value=[])
        fake_ib_insync.IB = lambda: ib_instance

        connector = IBKRConnector()
        prices, volumes = connector.fetch_bars("AAPL", "2023-01-01", "2023-01-03")

        assert len(prices) == 0
        assert len(volumes) == 0

    def test_disconnect_called_on_exception(self, fake_ib_insync):
        """ib.disconnect() must be called even when reqHistoricalData raises."""
        from data_connectors import IBKRConnector

        ib_instance = MagicMock()
        ib_instance.connect = MagicMock()
        ib_instance.disconnect = MagicMock()
        ib_instance.qualifyContracts = MagicMock()
        ib_instance.reqHistoricalData = MagicMock(side_effect=RuntimeError("timeout"))
        fake_ib_insync.IB = lambda: ib_instance

        connector = IBKRConnector()
        with pytest.raises(RuntimeError):
            connector.fetch_bars("AAPL", "2023-01-01", "2023-01-03")

        ib_instance.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# ConnectorFactory tests
# ---------------------------------------------------------------------------

class TestConnectorFactory:
    def test_default_is_yahoo(self):
        from data_connectors import ConnectorFactory, YahooConnector

        factory = ConnectorFactory()
        assert isinstance(factory.get_connector(), YahooConnector)

    def test_explicit_yahoo(self):
        from data_connectors import ConnectorFactory, YahooConnector

        factory = ConnectorFactory(data_source="yahoo")
        assert isinstance(factory.get_connector(), YahooConnector)

    def test_ibkr_routing(self):
        from data_connectors import ConnectorFactory, IBKRConnector

        factory = ConnectorFactory(
            data_source="ibkr",
            ibkr_host="10.0.0.1",
            ibkr_port=4002,
            ibkr_client_id=3,
            ibkr_timeframe="1 hour",
        )
        connector = factory.get_connector()
        assert isinstance(connector, IBKRConnector)
        assert connector.host == "10.0.0.1"
        assert connector.port == 4002
        assert connector.timeframe == "1 hour"

    def test_case_insensitive(self):
        from data_connectors import ConnectorFactory, IBKRConnector

        factory = ConnectorFactory(data_source="IBKR")
        assert isinstance(factory.get_connector(), IBKRConnector)


# ---------------------------------------------------------------------------
# Fallback behaviour in TradingEnvironmentWithVolumeProfile
# ---------------------------------------------------------------------------

class TestTradingAgentFallback:
    def test_yahoo_source(self):
        from trading_agent import TradingEnvironmentWithVolumeProfile

        sample = pd.DataFrame({
            "Close": [10.0, 11.0, 12.0],
            "Volume": [100.0, 200.0, 300.0],
        })
        with patch("data_connectors.yf.download", return_value=sample):
            env = TradingEnvironmentWithVolumeProfile("TEST", data_source="yahoo")
            prices, volumes = env.load_data("2023-01-01", "2023-01-04")

        np.testing.assert_array_equal(prices, [10.0, 11.0, 12.0])

    def test_ibkr_fails_fallback_to_yahoo(self, fake_ib_insync):
        """When IBKR raises an exception, load_data must fall back to Yahoo."""
        from trading_agent import TradingEnvironmentWithVolumeProfile

        # Make IBKR connection raise
        fake_ib_insync.IB = MagicMock(side_effect=ConnectionRefusedError("refused"))

        sample = pd.DataFrame({
            "Close": [20.0, 21.0],
            "Volume": [500.0, 600.0],
        })
        with patch("data_connectors.yf.download", return_value=sample):
            env = TradingEnvironmentWithVolumeProfile(
                "TEST", data_source="ibkr", ibkr_host="127.0.0.1", ibkr_port=7497
            )
            prices, volumes = env.load_data("2023-01-01", "2023-01-03")

        assert len(prices) == 2
        np.testing.assert_array_equal(prices, [20.0, 21.0])

    def test_ibkr_empty_fallback_to_yahoo(self, fake_ib_insync):
        """When IBKR returns empty bars, load_data must fall back to Yahoo."""
        from trading_agent import TradingEnvironmentWithVolumeProfile

        fake_ib_insync.util.df = lambda bars: pd.DataFrame()
        ib_instance = MagicMock()
        ib_instance.reqHistoricalData = MagicMock(return_value=[])
        fake_ib_insync.IB = lambda: ib_instance

        sample = pd.DataFrame({
            "Close": [30.0],
            "Volume": [700.0],
        })
        with patch("data_connectors.yf.download", return_value=sample):
            env = TradingEnvironmentWithVolumeProfile("TEST", data_source="ibkr")
            prices, _ = env.load_data("2023-01-01", "2023-01-02")

        assert len(prices) == 1
        assert prices[0] == 30.0

    def test_backward_compatible_default(self):
        """Omitting data_source should behave identically to data_source='yahoo'."""
        from trading_agent import TradingEnvironmentWithVolumeProfile

        env = TradingEnvironmentWithVolumeProfile("TEST")
        assert env.data_source == "yahoo"
