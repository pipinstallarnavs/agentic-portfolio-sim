"""Deterministic feature calculations over price/return series.

Every function here is pure and unit-testable in isolation: given the same
inputs, it always returns the same outputs. Agents (Phase 3+) call these
functions instead of doing arithmetic themselves — the LLM never computes a
number, it only reads numbers these functions produced.

Convention: all functions operate on data "as of" the last row of the input
series/frame. Callers (the simulation engine) are responsible for slicing
input data to information available at time t *before* calling these
functions — that is where lookahead bias is actually prevented, not here.

Annualization convention: 252 trading days/year, used consistently
throughout the codebase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """Period-over-period returns from a price series (use adjusted prices).

    method: "simple" -> P_t/P_{t-1} - 1, or "log" -> ln(P_t / P_{t-1}).
    First value is dropped (there is no prior price to compare against).
    """
    if method not in ("simple", "log"):
        raise ValueError(f"Unknown method: {method!r}")
    if method == "simple":
        returns = prices.pct_change()
    else:
        returns = np.log(prices / prices.shift(1))
    return returns.dropna()


def calculate_momentum(prices: pd.Series, lookback: int) -> float:
    """Total return over the trailing `lookback` periods, using only prices
    already in the series (i.e. through the series' last index value).

    Returns NaN if there isn't enough history.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    clean = prices.dropna()
    if len(clean) <= lookback:
        return float("nan")
    return float(clean.iloc[-1] / clean.iloc[-1 - lookback] - 1.0)


def calculate_realized_volatility(
    returns: pd.Series, window: int, annualize: bool = True
) -> float:
    """Realized (historical) volatility over the trailing `window` returns.

    Returns NaN if there isn't enough history.
    """
    if window <= 1:
        raise ValueError("window must be > 1")
    clean = returns.dropna()
    if len(clean) < window:
        return float("nan")
    vol = clean.iloc[-window:].std(ddof=1)
    if annualize:
        vol *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(vol)


def calculate_moving_average(prices: pd.Series, window: int) -> float:
    """Simple moving average of the trailing `window` prices."""
    if window <= 0:
        raise ValueError("window must be positive")
    clean = prices.dropna()
    if len(clean) < window:
        return float("nan")
    return float(clean.iloc[-window:].mean())


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series, window: int | None = None) -> float:
    """Rolling (or full-sample) beta of asset returns to market returns.

    beta = Cov(asset, market) / Var(market), computed over the trailing
    `window` periods of the aligned series (or all available periods if
    window is None).
    """
    aligned = pd.concat([asset_returns, market_returns], axis=1, join="inner").dropna()
    aligned.columns = ["asset", "market"]
    if window is not None:
        if len(aligned) < window:
            return float("nan")
        aligned = aligned.iloc[-window:]
    if len(aligned) < 2:
        return float("nan")
    market_var = aligned["market"].var(ddof=1)
    if np.isnan(market_var) or abs(market_var) < 1e-16:
        return float("nan")
    cov = aligned["asset"].cov(aligned["market"])
    return float(cov / market_var)


def calculate_correlation_matrix(returns_df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Pairwise correlation matrix of asset returns.

    Uses the trailing `window` rows if given, else all available rows.
    Rows with any NaN are dropped before computing correlations so all
    assets are compared over the same dates.
    """
    clean = returns_df.dropna(how="any")
    if window is not None:
        clean = clean.iloc[-window:]
    return clean.corr()
