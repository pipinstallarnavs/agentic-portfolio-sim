"""Data provider abstraction.

A ``DataProvider`` answers one question: "give me the full historical OHLCV
series for this ticker, from cache if possible." It does NOT know anything
about simulation time ``t`` or lookahead bias — that is the simulation
engine's responsibility (see simulation/state.py). Loading full history into
memory upfront is fine for a backtest; what matters is that the engine only
ever *slices* that history up to ``t`` before handing it to agents or tools.

Two implementations are provided:
  - YFinanceDataProvider: downloads from yfinance, then caches to disk.
    Subsequent calls read from the cache and never hit the network again
    for dates already covered.
  - LocalCacheDataProvider: reads only from the on-disk cache, raising if a
    ticker isn't cached. Used for fully offline/deterministic runs (e.g. CI,
    unit tests) so the simulator never silently depends on API availability.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from agentic_portfolio.data.cache import REQUIRED_COLUMNS, CacheStore

logger = logging.getLogger(__name__)


class DataProviderError(RuntimeError):
    """Raised when historical data cannot be obtained for a ticker."""


class DataProvider(ABC):
    """Abstract source of historical daily OHLCV data."""

    @abstractmethod
    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return a DataFrame indexed by date with columns:
        open, high, low, close, adj_close, volume.

        ``close`` is the raw close; ``adj_close`` is split/dividend adjusted
        and is what all return calculations must use.
        """

    def get_price_panel(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Convenience: adjusted-close panel, columns=tickers, index=date."""
        series = {}
        for ticker in tickers:
            hist = self.get_history(ticker, start, end)
            series[ticker] = hist["adj_close"]
        panel = pd.DataFrame(series)
        panel = panel.sort_index()
        return panel


class YFinanceDataProvider(DataProvider):
    """Downloads from yfinance on cache miss, then caches to disk."""

    def __init__(self, raw_dir: Path):
        self.cache = CacheStore(raw_dir)

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        cached = self.cache.load(ticker)
        if cached is not None and not cached.empty:
            cached_start, cached_end = cached.index.min(), cached.index.max()
            if cached_start <= pd.Timestamp(start) and cached_end >= pd.Timestamp(end) - pd.Timedelta(days=5):
                return cached.loc[start:end]
            logger.info("Cache for %s does not fully cover [%s, %s]; refreshing", ticker, start, end)

        df = self._download(ticker, start, end)
        self.cache.save(ticker, df)
        return df.loc[start:end]

    def _download(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        logger.info("Downloading %s from yfinance [%s, %s]", ticker, start, end)
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
        if raw is None or raw.empty:
            raise DataProviderError(f"yfinance returned no data for {ticker}")

        raw = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        missing = set(REQUIRED_COLUMNS) - set(raw.columns)
        if missing:
            raise DataProviderError(f"yfinance response for {ticker} missing columns {missing}")
        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "date"
        return raw[REQUIRED_COLUMNS].sort_index()


class LocalCacheDataProvider(DataProvider):
    """Reads only from the on-disk cache. Never touches the network."""

    def __init__(self, raw_dir: Path):
        self.cache = CacheStore(raw_dir)

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        cached = self.cache.load(ticker)
        if cached is None:
            raise DataProviderError(
                f"No cached data for {ticker} in {self.cache.raw_dir}. "
                "Run scripts/download_data.py first, or use the yfinance provider."
            )
        return cached.loc[start:end]


def get_data_provider(kind: str, raw_dir: Path) -> DataProvider:
    if kind == "yfinance":
        return YFinanceDataProvider(raw_dir)
    if kind == "local":
        return LocalCacheDataProvider(raw_dir)
    raise ValueError(f"Unknown data provider kind: {kind!r} (expected 'yfinance' or 'local')")
