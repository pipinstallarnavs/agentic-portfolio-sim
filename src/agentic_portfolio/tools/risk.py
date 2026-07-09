"""Deterministic risk calculations: VaR, expected shortfall, drawdown.

All functions take a return series or equity curve that the caller has
already restricted to information available at decision time — these
functions do not know about simulation time and perform no date filtering
themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) Value-at-Risk as a positive loss fraction.

    E.g. VaR_95 = 0.03 means: over the sample, the 5th percentile return
    was -3% — a 3% loss is the threshold the worst 5% of days breach.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    percentile = (1 - confidence) * 100
    var_return = np.percentile(clean, percentile)
    return float(max(-var_return, 0.0))


def calculate_expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR): average loss in the tail beyond historical VaR.

    Returned as a positive loss fraction, consistent with calculate_historical_var.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    percentile = (1 - confidence) * 100
    threshold = np.percentile(clean, percentile)
    tail = clean[clean <= threshold]
    if tail.empty:
        tail = clean.nsmallest(1)
    return float(max(-tail.mean(), 0.0))


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough decline of an equity/value curve, as a
    positive fraction (0.25 == a 25% drawdown).
    """
    clean = equity_curve.dropna()
    if clean.empty:
        return float("nan")
    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    return float(-drawdown.min())
