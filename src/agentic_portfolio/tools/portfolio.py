"""Deterministic portfolio-level calculations: returns, turnover, costs,
and constraint validation.

Weights are represented as ``dict[ticker, float]``. Cash is implicit: any
gross exposure less than 1.0 is assumed to be held as cash unless a
separate cash weight is tracked by the caller.
"""

from __future__ import annotations

from pydantic import BaseModel
from math import isfinite


def calculate_portfolio_return(weights: dict[str, float], asset_returns: dict[str, float]) -> float:
    """Weighted portfolio return for one period.

    Tickers in `weights` but missing from `asset_returns` contribute zero
    return (e.g. cash, or a delisted name with no return that period).
    """
    return sum(w * asset_returns.get(ticker, 0.0) for ticker, w in weights.items())


def calculate_portfolio_volatility(weights: dict[str, float], cov_matrix) -> float:
    """Annualized portfolio volatility from weights and a *daily* covariance
    matrix (as produced by ``returns_df.cov()``).

    cov_matrix: pandas DataFrame indexed/columned by ticker, in daily return units.
    """
    import numpy as np

    tickers = [t for t in weights if t in cov_matrix.index]
    if not tickers:
        return 0.0
    w = np.array([weights[t] for t in tickers])
    cov = cov_matrix.loc[tickers, tickers].values
    daily_var = float(w @ cov @ w)
    daily_var = max(daily_var, 0.0)
    return float(np.sqrt(daily_var) * np.sqrt(252))


def calculate_turnover(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    """One-way turnover: half the sum of absolute weight changes across the
    union of tickers. A full liquidation-and-rebuy of a 100%-invested
    portfolio has turnover 1.0.
    """
    tickers = set(old_weights) | set(new_weights)
    total_abs_change = sum(abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0)) for t in tickers)
    return total_abs_change / 2.0


def apply_transaction_costs(trade_values: dict[str, float], cost_bps: float) -> float:
    """Total transaction cost in dollars for a set of trades.

    trade_values: dict[ticker, signed dollar trade amount] (buy positive,
    sell negative). Cost is charged on notional traded (absolute value),
    symmetrically for buys and sells.
    """
    notional = sum(abs(v) for v in trade_values.values())
    return notional * (cost_bps / 10_000.0)


def cap_and_renormalize_weights(
    weights: dict[str, float], max_position_weight: float, target_gross: float, max_iter: int = 50
) -> dict[str, float]:
    """Scale `weights` to sum to `target_gross`, then iteratively clip any
    weight above `max_position_weight` and redistribute the excess
    proportionally among the uncapped names, until nothing exceeds the cap.

    If the cap alone can't fit `target_gross` (e.g. cap * n_names < target),
    every name is set to the cap and the remainder is left unallocated
    (effectively additional cash).
    """
    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0:
        return {t: 0.0 for t in weights}
    current = {t: w / total * target_gross for t, w in weights.items()}

    for _ in range(max_iter):
        capped = {t for t, w in current.items() if w > max_position_weight + 1e-12}
        if not capped:
            break
        capped_total = len(capped) * max_position_weight
        if capped_total >= target_gross:
            return {t: (max_position_weight if t in capped else 0.0) for t in current}
        remaining = target_gross - capped_total
        uncapped = [t for t in current if t not in capped]
        uncapped_total = sum(current[t] for t in uncapped)
        next_weights = dict(current)
        for t in capped:
            next_weights[t] = max_position_weight
        if uncapped_total > 0:
            for t in uncapped:
                next_weights[t] = current[t] / uncapped_total * remaining
        current = next_weights
    return current


class WeightValidationResult(BaseModel):
    is_valid: bool
    violations: list[str] = []
    gross_exposure: float
    max_position: float
    min_weight: float


def validate_weights(
    weights: dict[str, float],
    long_only: bool = True,
    max_position_weight: float = 0.20,
    max_gross_exposure: float = 1.00,
    tolerance: float = 1e-6,
) -> WeightValidationResult:
    """Check a proposed weight dict against hard portfolio constraints.

    This does not modify weights — it only reports violations. Modifying a
    proposal in response to violations is the Risk Agent's job (Phase 3+).
    """
    violations: list[str] = []
    values = list(weights.values())
    if any(not isfinite(w) for w in values):
        violations.append("Weights must be finite")
    min_weight = min(values) if values else 0.0
    gross_exposure = sum(abs(w) for w in values)
    max_position = max((abs(w) for w in values), default=0.0)

    if long_only and min_weight < -tolerance:
        violations.append(f"long_only violated: min weight {min_weight:.4f} < 0")

    for ticker, w in weights.items():
        if abs(w) > max_position_weight + tolerance:
            violations.append(
                f"max_position_weight violated for {ticker}: {w:.4f} > {max_position_weight:.4f}"
            )

    if gross_exposure > max_gross_exposure + tolerance:
        violations.append(
            f"max_gross_exposure violated: {gross_exposure:.4f} > {max_gross_exposure:.4f}"
        )

    return WeightValidationResult(
        is_valid=len(violations) == 0,
        violations=violations,
        gross_exposure=gross_exposure,
        max_position=max_position,
        min_weight=min_weight,
    )
