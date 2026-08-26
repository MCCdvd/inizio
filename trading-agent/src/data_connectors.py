"""
Data connectors for the trading agent.

Provides a unified interface to fetch historical OHLCV data from different
sources (Yahoo Finance, Interactive Brokers).
"""
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Abstract base class for market-data connectors.

    All concrete connectors must implement :meth:`fetch_bars`, which returns a
    ``(prices, volumes)`` tuple of 1-D float64 arrays aligned by date.
    """

    @abstractmethod
    def fetch_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch historical price and volume bars.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. ``"AAPL"``).
        start_date : str
            ISO-8601 start date, inclusive (``"YYYY-MM-DD"``).
        end_date : str
            ISO-8601 end date, exclusive (``"YYYY-MM-DD"``).

        Returns
        -------
        prices : np.ndarray
            1-D array of closing prices (float64).
        volumes : np.ndarray
            1-D array of volumes (float64).
        """


class YahooConnector(BaseConnector):
    """Fetch historical bars from Yahoo Finance via *yfinance*.

    Parameters
    ----------
    progress : bool, optional
        Whether to show the yfinance download progress bar.  Defaults to
        ``False``.
    """

    def __init__(self, progress: bool = False) -> None:
        self.progress = progress

    def fetch_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch bars from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        start_date : str
            ISO-8601 start date.
        end_date : str
            ISO-8601 end date.

        Returns
        -------
        prices : np.ndarray
            Closing prices.
        volumes : np.ndarray
            Trading volumes.
        """
        logger.debug("YahooConnector: downloading %s from %s to %s", symbol, start_date, end_date)
        data = yf.download(symbol, start=start_date, end=end_date, progress=self.progress)
        if data is None or data.empty:
            logger.warning("YahooConnector: no data returned for %s", symbol)
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        prices = np.ravel(np.asarray(data["Close"])).astype(np.float64)
        volumes = np.ravel(np.asarray(data["Volume"])).astype(np.float64)
        logger.debug("YahooConnector: fetched %d bars for %s", len(prices), symbol)
        return prices, volumes


class IBKRConnector(BaseConnector):
    """Fetch historical bars from Interactive Brokers via *ib-insync*.

    Parameters
    ----------
    host : str
        TWS / IB Gateway hostname.  Defaults to ``"127.0.0.1"``.
    port : int
        TWS / IB Gateway port.  Defaults to ``7497``.
    client_id : int
        Unique client identifier.  Defaults to ``11``.
    timeframe : str
        IB bar-size string, e.g. ``"1 day"``, ``"1 hour"``.
        Defaults to ``"1 day"``.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 11,
        timeframe: str = "1 day",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.client_id = int(client_id)
        self.timeframe = timeframe

    def fetch_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch bars from Interactive Brokers.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        start_date : str
            ISO-8601 start date.
        end_date : str
            ISO-8601 end date.

        Returns
        -------
        prices : np.ndarray
            Closing prices.
        volumes : np.ndarray
            Trading volumes.

        Raises
        ------
        ImportError
            If *ib-insync* is not installed.
        ValueError
            If ``end_date`` is not after ``start_date``.
        RuntimeError
            If the IB connection or data request fails.
        """
        from ib_insync import IB, Stock, util  # optional dependency

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if end <= start:
            raise ValueError(f"end_date ({end_date}) must be after start_date ({start_date})")

        days = max(1, int((end - start).days))
        duration_str = f"{days} D"

        logger.info(
            "IBKRConnector: connecting to %s:%s (clientId=%s)",
            self.host, self.port, self.client_id,
        )
        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id, readonly=True)

        try:
            contract = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(contract)

            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end.strftime("%Y%m%d %H:%M:%S"),
                durationStr=duration_str,
                barSizeSetting=self.timeframe,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
            )
        finally:
            ib.disconnect()
            logger.debug("IBKRConnector: disconnected")

        df = util.df(bars) if bars else pd.DataFrame()
        if df is None or df.empty:
            logger.warning("IBKRConnector: no data returned for %s", symbol)
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        close_col = "close" if "close" in df.columns else "Close"
        vol_col = "volume" if "volume" in df.columns else "Volume"
        prices = np.ravel(np.asarray(df[close_col])).astype(np.float64)
        volumes = np.ravel(np.asarray(df[vol_col])).astype(np.float64)
        logger.debug("IBKRConnector: fetched %d bars for %s", len(prices), symbol)
        return prices, volumes


class ConnectorFactory:
    """Route a ``data_source`` string to the appropriate connector instance.

    Parameters
    ----------
    data_source : str
        ``"yahoo"`` (default) or ``"ibkr"``.
    ibkr_host : str
        Passed to :class:`IBKRConnector` when ``data_source="ibkr"``.
    ibkr_port : int
        Passed to :class:`IBKRConnector` when ``data_source="ibkr"``.
    ibkr_client_id : int
        Passed to :class:`IBKRConnector` when ``data_source="ibkr"``.
    ibkr_timeframe : str
        Passed to :class:`IBKRConnector` when ``data_source="ibkr"``.
    """

    def __init__(
        self,
        data_source: str = "yahoo",
        ibkr_host: str = "127.0.0.1",
        ibkr_port: int = 7497,
        ibkr_client_id: int = 11,
        ibkr_timeframe: str = "1 day",
    ) -> None:
        self.data_source = str(data_source).lower()
        self.ibkr_host = ibkr_host
        self.ibkr_port = int(ibkr_port)
        self.ibkr_client_id = int(ibkr_client_id)
        self.ibkr_timeframe = ibkr_timeframe

    def get_connector(self) -> BaseConnector:
        """Return the connector matching *data_source*.

        Returns
        -------
        BaseConnector
            A :class:`YahooConnector` or :class:`IBKRConnector` instance.
        """
        if self.data_source == "ibkr":
            logger.info("ConnectorFactory: selecting IBKRConnector")
            return IBKRConnector(
                host=self.ibkr_host,
                port=self.ibkr_port,
                client_id=self.ibkr_client_id,
                timeframe=self.ibkr_timeframe,
            )
        logger.info("ConnectorFactory: selecting YahooConnector")
        return YahooConnector()
