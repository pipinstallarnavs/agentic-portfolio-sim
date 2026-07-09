import numpy as np
import pandas as pd
import pytest

from agentic_portfolio.tools.features import (
    calculate_beta,
    calculate_correlation_matrix,
    calculate_momentum,
    calculate_moving_average,
    calculate_realized_volatility,
    calculate_returns,
)


def make_prices(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx)


def test_calculate_returns_simple():
    prices = make_prices([100, 110, 99])
    returns = calculate_returns(prices, method="simple")
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(99 / 110 - 1)


def test_calculate_returns_log():
    prices = make_prices([100, 110])
    returns = calculate_returns(prices, method="log")
    assert returns.iloc[0] == pytest.approx(np.log(1.10))


def test_calculate_returns_invalid_method():
    with pytest.raises(ValueError):
        calculate_returns(make_prices([100, 110]), method="bogus")


def test_calculate_momentum_basic():
    prices = make_prices([100] * 10 + [120])
    mom = calculate_momentum(prices, lookback=10)
    assert mom == pytest.approx(0.20)


def test_calculate_momentum_insufficient_history():
    prices = make_prices([100, 105])
    assert np.isnan(calculate_momentum(prices, lookback=10))


def test_calculate_realized_volatility_known_value():
    # Constant returns of 1% every day -> zero std -> zero vol.
    idx = pd.bdate_range("2024-01-01", periods=25)
    returns = pd.Series([0.01] * 24, index=idx[1:])
    vol = calculate_realized_volatility(returns, window=20, annualize=False)
    assert vol == pytest.approx(0.0, abs=1e-9)


def test_calculate_realized_volatility_matches_manual_std():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=30)
    returns = pd.Series(rng.normal(0, 0.02, size=30), index=idx)
    vol = calculate_realized_volatility(returns, window=20, annualize=False)
    expected = returns.iloc[-20:].std(ddof=1)
    assert vol == pytest.approx(expected)


def test_calculate_realized_volatility_insufficient_history():
    idx = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.Series([0.01] * 5, index=idx)
    assert np.isnan(calculate_realized_volatility(returns, window=20))


def test_calculate_moving_average():
    prices = make_prices([10, 20, 30, 40])
    ma = calculate_moving_average(prices, window=2)
    assert ma == pytest.approx(35.0)


def test_calculate_beta_perfect_correlation():
    idx = pd.bdate_range("2024-01-01", periods=20)
    market = pd.Series(np.linspace(0.001, 0.02, 20), index=idx)
    asset = market * 2.0  # exactly 2x the market's moves -> beta should be 2
    beta = calculate_beta(asset, market)
    assert beta == pytest.approx(2.0, rel=1e-6)


def test_calculate_beta_zero_market_variance():
    idx = pd.bdate_range("2024-01-01", periods=10)
    market = pd.Series([0.001] * 10, index=idx)
    asset = pd.Series(np.linspace(0.001, 0.02, 10), index=idx)
    assert np.isnan(calculate_beta(asset, market))


def test_calculate_correlation_matrix_diagonal_is_one():
    idx = pd.bdate_range("2024-01-01", periods=30)
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"A": rng.normal(0, 0.01, 30), "B": rng.normal(0, 0.01, 30)}, index=idx
    )
    corr = calculate_correlation_matrix(df)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["B", "B"] == pytest.approx(1.0)
