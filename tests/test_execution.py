import pytest

from agentic_portfolio.simulation.execution import execute_rebalance
from agentic_portfolio.tools.portfolio import apply_transaction_costs


def test_apply_transaction_costs_known_value():
    trades = {"A": 10_000, "B": -5_000}
    cost = apply_transaction_costs(trades, cost_bps=10)
    assert cost == pytest.approx(15_000 * 0.0010)


def test_apply_transaction_costs_no_trades():
    assert apply_transaction_costs({}, cost_bps=10) == 0.0


def test_execute_rebalance_from_all_cash():
    result = execute_rebalance(
        current_shares={},
        current_cash=1_000_000,
        target_weights={"AAPL": 0.5, "MSFT": 0.5},
        execution_prices={"AAPL": 100.0, "MSFT": 200.0},
        cost_bps=0,
    )
    assert result.portfolio_value_before == pytest.approx(1_000_000)
    assert result.new_shares["AAPL"] == pytest.approx(5_000)
    assert result.new_shares["MSFT"] == pytest.approx(2_500)
    assert result.cash_after == pytest.approx(0.0, abs=1e-6)
    assert result.turnover == pytest.approx(1.0)


def test_execute_rebalance_conserves_value_net_of_costs():
    result = execute_rebalance(
        current_shares={},
        current_cash=1_000_000,
        target_weights={"AAPL": 0.5, "MSFT": 0.5},
        execution_prices={"AAPL": 100.0, "MSFT": 200.0},
        cost_bps=10,
    )
    assert result.portfolio_value_after == pytest.approx(
        result.portfolio_value_before - result.transaction_cost
    )
    reconstructed_value = result.cash_after + sum(
        result.new_shares[t] * p for t, p in {"AAPL": 100.0, "MSFT": 200.0}.items()
    )
    assert reconstructed_value == pytest.approx(result.portfolio_value_after)


def test_execute_rebalance_no_trade_needed_has_zero_turnover_and_cost():
    result = execute_rebalance(
        current_shares={"AAPL": 5_000},
        current_cash=0.0,
        target_weights={"AAPL": 1.0},
        execution_prices={"AAPL": 100.0},
        cost_bps=10,
    )
    assert result.turnover == pytest.approx(0.0, abs=1e-9)
    assert result.transaction_cost == pytest.approx(0.0, abs=1e-9)
    assert len(result.trades) == 0


def test_execute_rebalance_missing_price_raises():
    with pytest.raises(ValueError):
        execute_rebalance(
            current_shares={},
            current_cash=1000,
            target_weights={"AAPL": 1.0},
            execution_prices={},
            cost_bps=0,
        )


def test_execute_rebalance_partial_liquidation():
    # Held 100% AAPL, target is 50% AAPL / 50% cash -> half should be sold.
    result = execute_rebalance(
        current_shares={"AAPL": 10_000},
        current_cash=0.0,
        target_weights={"AAPL": 0.5},
        execution_prices={"AAPL": 100.0},
        cost_bps=0,
    )
    assert result.new_shares["AAPL"] == pytest.approx(5_000)
    assert result.cash_after == pytest.approx(500_000)
    assert result.turnover == pytest.approx(0.5)


def test_fully_invested_trade_pays_fees_without_borrowing():
    result = execute_rebalance({}, 1000, {'A': 1.0}, {'A': 100.}, 10)
    # W + .001*W = 1000 is an analytic answer independent of the bisection.
    assert result.portfolio_value_after == pytest.approx(1000/1.001)
    assert result.cash_after >= 0
    assert result.realized_weights['A'] == pytest.approx(1)


def test_rotation_pays_for_both_sides_and_conserves_value():
    result = execute_rebalance({'A': 10}, 0, {'B': 1}, {'A': 100., 'B': 50.}, 10)
    assert result.portfolio_value_after == pytest.approx(1000*(1-.001)/(1+.001))
    assert result.cash_after >= 0
    assert result.new_shares['A'] == 0


@pytest.mark.parametrize('prices', [{}, {'A': 0.}, {'A': float('nan')}])
def test_missing_or_invalid_held_price_fails(prices):
    with pytest.raises(ValueError):
        execute_rebalance({'A': 1}, 0, {}, prices, 10)


@pytest.mark.parametrize('weights', [{'A': -0.2}, {'A': 1.1}, {'A': float('nan')}])
def test_invalid_target_rejected(weights):
    with pytest.raises(ValueError):
        execute_rebalance({}, 1000, weights, {'A': 100}, 10)
