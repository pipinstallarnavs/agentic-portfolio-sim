"""Financial performance metrics computed from a strategy's equity curve.

These are evaluation-time calculations over a *completed* backtest — unlike
tools/features.py and tools/risk.py, there is no lookahead concern here
because nothing computed here feeds back into a trading decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_portfolio.simulation.state import SimulationResult
from agentic_portfolio.tools.features import TRADING_DAYS_PER_YEAR, calculate_returns
from agentic_portfolio.tools.risk import calculate_expected_shortfall, calculate_historical_var, calculate_max_drawdown


def annualized_return(equity_curve: pd.Series) -> float:
    clean = equity_curve.dropna()
    if len(clean) < 2:
        return float("nan")
    total_return = clean.iloc[-1] / clean.iloc[0] - 1.0
    n_days = len(clean) - 1
    years = n_days / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.dropna()
    if clean.empty or clean.std(ddof=1) == 0:
        return float("nan")
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = clean - daily_rf
    return float(excess.mean() / clean.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = clean - daily_rf
    downside = excess[excess < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return float("nan")
    downside_dev = downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(excess.mean() * TRADING_DAYS_PER_YEAR / downside_dev)


def calmar_ratio(equity_curve: pd.Series) -> float:
    ann_return = annualized_return(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    if max_dd == 0 or np.isnan(max_dd):
        return float("nan")
    return float(ann_return / max_dd)


def compute_financial_metrics(
    result: SimulationResult,
    var_confidence: float = 0.95,
    es_confidence: float = 0.95,
) -> dict[str, float]:
    """Full financial metric report for one completed simulation run."""
    equity = result.equity_series()
    returns = calculate_returns(equity)

    turnovers = [r.turnover for r in result.rebalances]
    costs = [r.transaction_cost for r in result.rebalances]

    return {
        "final_value": float(equity.iloc[-1]) if not equity.empty else float("nan"),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else float("nan"),
        "annualized_return": annualized_return(equity),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": calculate_max_drawdown(equity),
        "calmar_ratio": calmar_ratio(equity),
        f"historical_var_{int(var_confidence*100)}": calculate_historical_var(returns, var_confidence),
        f"expected_shortfall_{int(es_confidence*100)}": calculate_expected_shortfall(returns, es_confidence),
        "n_rebalances": float(len(result.rebalances)),
        "mean_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "total_transaction_costs": float(np.sum(costs)) if costs else 0.0,
        "total_transaction_costs_pct_of_initial": (
            float(np.sum(costs) / equity.iloc[0]) if costs and not equity.empty else 0.0
        ),
    }


def buy_and_hold_equity_curve(prices: pd.Series, initial_capital: float) -> pd.Series:
    """Equity curve for holding a single instrument (e.g. SPY) from the
    first available price, for use as a passive benchmark comparison.
    """
    clean = prices.dropna()
    shares = initial_capital / clean.iloc[0]
    return clean * shares
