"""Local on-disk cache for per-ticker historical price data.

Data is cached as one parquet file per ticker under ``data/raw``. Once a
ticker's history has been fetched and cached, the simulator never needs to
hit a live API again — providers should read through this cache first.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class CacheStore:
    """Reads and writes cached OHLCV history for a single ticker per file."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, ticker: str) -> Path:
        return self.raw_dir / f"{ticker.upper()}.parquet"

    def exists(self, ticker: str) -> bool:
        return self.path_for(ticker).exists()

    def load(self, ticker: str) -> pd.DataFrame | None:
        path = self.path_for(ticker)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def save(self, ticker: str, df: pd.DataFrame) -> None:
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Cannot cache {ticker}: missing columns {missing}")
        out = df[REQUIRED_COLUMNS].copy()
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        out.to_parquet(self.path_for(ticker))
        logger.info("Cached %d rows for %s at %s", len(out), ticker, self.path_for(ticker))
