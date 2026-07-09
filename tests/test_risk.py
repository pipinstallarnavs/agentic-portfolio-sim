import numpy as np
import pandas as pd
import pytest

from agentic_portfolio.tools.portfolio import (
    cap_and_renormalize_weights,
    calculate_turnover,
    validate_weights,
)
from agentic_portfolio.tools.risk import (
    calculate_expected_shortfall,
    calculate_historical_var,
    calculate_max_drawdown,
)


def test_calculate_historical_var_known_distribution():
    # 100 evenly spaced returns from -0.10 to 0.09; 5th percentile is well defined.
    returns = pd.Series(np.linspace(-0.10, 0.09, 100))
    var_95 = calculate_historical_var(returns, confidence=0.95)
    expected = -np.percentile(returns, 5)
    assert var_95 == pytest.approx(expected)
    assert var_95 > 0  # reported as a positive loss magnitude


def test_calculate_historical_var_invalid_confidence():
    with pytest.raises(ValueError):
        calculate_historical_var(pd.Series([0.01, -0.01]), confidence=1.5)


def test_calculate_expected_shortfall_is_worse_than_var():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0, 0.02, 500))
    var_95 = calculate_historical_var(returns, 0.95)
    es_95 = calculate_expected_shortfall(returns, 0.95)
    assert es_95 >= var_95


def test_calculate_max_drawdown_known_path():
    equity = pd.Series([100, 120, 90, 95, 130])
    dd = calculate_max_drawdown(equity)
    assert dd == pytest.approx((120 - 90) / 120)


def test_calculate_max_drawdown_monotonic_increase_is_zero():
    equity = pd.Series([100, 110, 120, 130])
    assert calculate_max_drawdown(equity) == pytest.approx(0.0)


def test_calculate_turnover_full_rebalance():
    old = {"A": 1.0}
    new = {"B": 1.0}
    assert calculate_turnover(old, new) == pytest.approx(1.0)


def test_calculate_turnover_no_change():
    weights = {"A": 0.5, "B": 0.5}
    assert calculate_turnover(weights, weights) == pytest.approx(0.0)


def test_calculate_turnover_partial():
    old = {"A": 0.5, "B": 0.5}
    new = {"A": 0.6, "B": 0.4}
    assert calculate_turnover(old, new) == pytest.approx(0.1)


def test_validate_weights_passes():
    result = validate_weights({"A": 0.15, "B": 0.15}, max_position_weight=0.20, max_gross_exposure=1.0)
    assert result.is_valid
    assert result.violations == []


def test_validate_weights_max_position_violation():
    result = validate_weights({"A": 0.5, "B": 0.1}, max_position_weight=0.20)
    assert not result.is_valid
    assert any("max_position_weight" in v for v in result.violations)


def test_validate_weights_gross_exposure_violation():
    result = validate_weights({"A": 0.6, "B": 0.6}, max_position_weight=1.0, max_gross_exposure=1.0)
    assert not result.is_valid
    assert any("max_gross_exposure" in v for v in result.violations)


def test_validate_weights_long_only_violation():
    result = validate_weights({"A": -0.1, "B": 0.2}, long_only=True, max_position_weight=1.0)
    assert not result.is_valid
    assert any("long_only" in v for v in result.violations)


def test_cap_and_renormalize_sums_to_target():
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    capped = cap_and_renormalize_weights(weights, max_position_weight=0.35, target_gross=1.0)
    assert sum(capped.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(w <= 0.35 + 1e-9 for w in capped.values())


def test_cap_and_renormalize_no_cap_needed():
    weights = {"A": 0.2, "B": 0.2}
    capped = cap_and_renormalize_weights(weights, max_position_weight=0.5, target_gross=0.4)
    assert capped["A"] == pytest.approx(0.2)
    assert capped["B"] == pytest.approx(0.2)


def test_cap_and_renormalize_cap_times_n_below_target():
    # 2 names capped at 0.2 can supply at most 0.4 gross; target of 1.0 is unreachable.
    weights = {"A": 0.5, "B": 0.5}
    capped = cap_and_renormalize_weights(weights, max_position_weight=0.2, target_gross=1.0)
    assert capped["A"] == pytest.approx(0.2)
    assert capped["B"] == pytest.approx(0.2)
