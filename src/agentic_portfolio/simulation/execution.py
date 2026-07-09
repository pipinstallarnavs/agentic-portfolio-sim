"""Execution layer: turns target weights into trades at execution prices,
applying transaction costs.

Timing convention (see README "Simulation Timing"): a strategy decides
target weights using data through decision date t. Trades are assumed to
execute at the close of the *next* trading day, t+1. This module only deals
with the mechanics of one such execution — it is handed t+1's prices, not
data from earlier dates, so it cannot itself introduce lookahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from agentic_portfolio.simulation.state import Trade
from agentic_portfolio.tools.portfolio import apply_transaction_costs, calculate_turnover


@dataclass
class ExecutionResult:
    trades: list[Trade]
    new_shares: dict[str, float]
    cash_after: float
    transaction_cost: float
    turnover: float
    portfolio_value_before: float
    portfolio_value_after: float
    realized_weights: dict[str, float] = field(default_factory=dict)


def execute_rebalance(
    current_shares: dict[str, float],
    current_cash: float,
    target_weights: dict[str, float],
    execution_prices: dict[str, float],
    cost_bps: float,
) -> ExecutionResult:
    """Compute trades to move from the current holdings to target_weights,
    priced at execution_prices, net of transaction costs.

    Any ticker in target_weights must have a price in execution_prices.
    Tickers held but not in target_weights are treated as fully sold
    (target weight 0).
    """
    # This educational executor is long-only and unlevered. Unknown held
    # prices must fail: valuing a missing holding at zero loses real wealth.
    required = set(target_weights) | {t for t, n in current_shares.items() if n != 0}
    if required - set(execution_prices):
        raise ValueError(f"Missing execution prices for {required - set(execution_prices)}")
    if any(not isfinite(execution_prices[t]) or execution_prices[t] <= 0 for t in required):
        raise ValueError("Execution prices must be finite and positive")
    if any(not isfinite(w) or w < 0 for w in target_weights.values()) or sum(target_weights.values()) > 1 + 1e-10:
        raise ValueError("Target weights must be finite, nonnegative and sum to at most one")
    if (not isfinite(cost_bps) or not 0 <= cost_bps < 10_000 or
        not isfinite(current_cash) or current_cash < -1e-6 or
        any(not isfinite(n) or n < 0 for n in current_shares.values())):
        raise ValueError("Invalid cost, cash, or shares for an unlevered portfolio")
    held_value = sum(n * execution_prices[t] for t, n in current_shares.items() if n != 0)
    portfolio_value_before = current_cash + held_value
    if portfolio_value_before <= 0:
        raise ValueError("Portfolio value must be positive to execute a rebalance")

    # Target weights are fractions of AFTER-COST wealth W. Solve
    # W + cost_rate * sum(|target_weight*W - current_dollars|) = old_wealth.
    # Spending old_wealth first would create negative cash to pay the fee.
    # The left side is monotone for cost_rate<1 and gross<=1; bisection is
    # sufficient and easier to audit than an optimizer for this one scalar.
    names = set(current_shares) | set(target_weights)
    current_dollars = {t: current_shares.get(t, 0)*execution_prices[t]
                       for t in names if t in execution_prices}
    low, high = 0.0, portfolio_value_before
    for _ in range(70):
        wealth = (low+high)/2
        cost = cost_bps/10_000 * sum(abs(target_weights.get(t, 0)*wealth-current_dollars.get(t, 0)) for t in names)
        if wealth + cost > portfolio_value_before:
            high = wealth
        else:
            low = wealth
    investable_wealth = (low+high)/2

    all_tickers = set(current_shares) | set(target_weights) | set(execution_prices)
    old_weights = {
        t: current_shares.get(t, 0.0) * execution_prices.get(t, 0.0) / portfolio_value_before
        for t in all_tickers
        if t in execution_prices
    }
    # Include the cash leg explicitly: a move from 100% cash to 100% equities
    # is a full turnover event even though every individual ticker weight
    # only "increases" (there's no offsetting equity sale to net against).
    old_weights_with_cash = {**old_weights, "_CASH_": current_cash / portfolio_value_before}
    target_weights_with_cash = {
        **target_weights,
        "_CASH_": 1.0 - sum(target_weights.values()),
    }
    turnover = calculate_turnover(old_weights_with_cash, target_weights_with_cash)

    trade_dollars: dict[str, float] = {}
    new_shares: dict[str, float] = {}
    trades: list[Trade] = []
    for ticker in all_tickers:
        price = execution_prices.get(ticker)
        if price is None or price <= 0:
            # No tradable price this period (e.g. delisted) -> hold existing shares as-is.
            new_shares[ticker] = current_shares.get(ticker, 0.0)
            continue
        target_dollar = target_weights.get(ticker, 0.0) * investable_wealth
        current_dollar = current_shares.get(ticker, 0.0) * price
        delta_dollar = target_dollar - current_dollar
        if abs(delta_dollar) > 1e-9:
            trade_dollars[ticker] = delta_dollar
            trades.append(
                Trade(
                    ticker=ticker,
                    shares_delta=delta_dollar / price,
                    dollar_amount=delta_dollar,
                    price=price,
                )
            )
        new_shares[ticker] = target_dollar / price

    transaction_cost = apply_transaction_costs(trade_dollars, cost_bps)
    spent_on_targets = sum(target_weights.get(t, 0.0) * investable_wealth for t in target_weights)
    cash_after = portfolio_value_before - spent_on_targets - transaction_cost
    if -1e-6 < cash_after < 0:
        cash_after = 0.0  # floating-point residue, not borrowing
    portfolio_value_after = portfolio_value_before - transaction_cost

    realized_weights = {
        t: (new_shares.get(t, 0.0) * execution_prices[t]) / portfolio_value_after
        for t in execution_prices
        if new_shares.get(t, 0.0) != 0.0
    }

    return ExecutionResult(
        trades=trades,
        new_shares=new_shares,
        cash_after=cash_after,
        transaction_cost=transaction_cost,
        turnover=turnover,
        portfolio_value_before=portfolio_value_before,
        portfolio_value_after=portfolio_value_after,
        realized_weights=realized_weights,
    )
