#!/usr/bin/env python3
"""Download and cache historical price data for the configured universe.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --synthetic   # offline fallback, no network

Once run, the simulator never needs the network again — everything reads
from data/raw/<TICKER>.parquet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_portfolio.config import load_config
from agentic_portfolio.data.cache import CacheStore
from agentic_portfolio.data.provider import YFinanceDataProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("download_data")


def generate_synthetic_history(ticker: str, start: str, end: str, seed: int) -> pd.DataFrame:
    """Deterministic geometric Brownian motion series, seeded per ticker,
    used only when real market data can't be fetched (offline dev/CI).
    Clearly NOT real data — for pipeline development and tests only.
    """
    rng = np.random.default_rng(abs(hash((ticker, seed))) % (2**32))
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    annual_drift = rng.uniform(0.04, 0.14)
    annual_vol = rng.uniform(0.18, 0.40)
    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)
    shocks = rng.normal(daily_drift - 0.5 * daily_vol**2, daily_vol, size=n)
    log_prices = np.cumsum(shocks)
    start_price = rng.uniform(30, 400)
    close = start_price * np.exp(log_prices)

    daily_range = np.abs(rng.normal(0, daily_vol, size=n)) * close
    high = close + daily_range * 0.5
    low = close - daily_range * 0.5
    open_ = low + (high - low) * rng.uniform(0, 1, size=n)
    volume = rng.integers(1_000_000, 20_000_000, size=n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=dates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config YAML (default: config/default.yaml)")
    parser.add_argument("--synthetic", action="store_true", help="Generate deterministic synthetic data instead of hitting yfinance")
    parser.add_argument("--force", action="store_true", help="Re-download/regenerate even if already cached")
    args = parser.parse_args()

    config = load_config(args.config)
    tickers = list(config.universe) + [config.benchmark]
    cache = CacheStore(config.raw_dir)

    if args.synthetic:
        logger.warning("Generating SYNTHETIC price data (not real market data) for offline development/testing")
        for ticker in tickers:
            if cache.exists(ticker) and not args.force:
                logger.info("%s already cached, skipping (use --force to regenerate)", ticker)
                continue
            df = generate_synthetic_history(ticker, config.data.start_date, config.data.end_date, config.random_seed)
            cache.save(ticker, df)
        logger.info("Synthetic data ready for %d tickers in %s", len(tickers), config.raw_dir)
        return

    provider = YFinanceDataProvider(config.raw_dir)
    failures = []
    for ticker in tickers:
        if cache.exists(ticker) and not args.force:
            logger.info("%s already cached, skipping (use --force to re-download)", ticker)
            continue
        try:
            provider.get_history(ticker, config.data.start_date, config.data.end_date)
        except Exception as exc:  # noqa: BLE001 - report and continue with other tickers
            logger.error("Failed to download %s: %s", ticker, exc)
            failures.append(ticker)

    if failures:
        logger.error(
            "Failed to download %d/%d tickers: %s. "
            "Re-run with --synthetic for an offline fallback dataset.",
            len(failures), len(tickers), failures,
        )
        sys.exit(1)
    logger.info("Downloaded and cached %d tickers to %s", len(tickers), config.raw_dir)


if __name__ == "__main__":
    main()
