"""
Unit tests for DeribitOptionsScanner.

Run with:  pytest tests/test_scanner.py -v
"""
import pytest
import numpy as np
from unittest.mock import patch
from scanner import DeribitOptionsScanner, OptionFilters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def scanner():
    return DeribitOptionsScanner()


# ---------------------------------------------------------------------------
# calculate_dte
# ---------------------------------------------------------------------------

def test_calculate_dte_future(scanner):
    """Expiration in the future → positive DTE."""
    from datetime import datetime, timedelta
    future = datetime.now() + timedelta(days=30)
    ts = int(future.timestamp() * 1000)
    dte = scanner.calculate_dte(ts)
    assert dte >= 29  # allow ±1 day rounding


def test_calculate_dte_past(scanner):
    """Expiration in the past → negative or zero DTE."""
    from datetime import datetime, timedelta
    past = datetime.now() - timedelta(days=5)
    ts = int(past.timestamp() * 1000)
    dte = scanner.calculate_dte(ts)
    assert dte <= 0


# ---------------------------------------------------------------------------
# _calculate_iv_rank
# ---------------------------------------------------------------------------

def test_iv_rank_insufficient_history(scanner):
    """Returns 50.0 (neutral) when fewer than 30 data points."""
    scanner.iv_history['BTC-TEST'] = [0.5] * 10
    assert scanner._calculate_iv_rank('BTC-TEST', 0.6) == 50.0


def test_iv_rank_returns_neutral_for_unknown(scanner):
    """Returns 50.0 when instrument has no history at all."""
    assert scanner._calculate_iv_rank('UNKNOWN', 0.9) == 50.0


def test_iv_rank_bounds(scanner):
    """IV rank is always in [0, 100]."""
    scanner.iv_history['BTC-A'] = list(np.linspace(0.2, 0.8, 50))
    assert scanner._calculate_iv_rank('BTC-A', 0.0) == 0.0
    assert scanner._calculate_iv_rank('BTC-A', 1.0) == 100.0


def test_iv_rank_midpoint(scanner):
    """Current IV equal to midpoint of history → rank ≈ 50."""
    scanner.iv_history['BTC-B'] = list(np.linspace(0.0, 1.0, 100))
    rank = scanner._calculate_iv_rank('BTC-B', 0.5)
    assert 48 <= rank <= 52


def test_iv_rank_flat_history(scanner):
    """Flat history (min == max) → returns neutral 50.0."""
    scanner.iv_history['BTC-FLAT'] = [0.5] * 50
    assert scanner._calculate_iv_rank('BTC-FLAT', 0.5) == 50.0


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------

def test_calculate_metrics_liquidity_zero_strike(scanner):
    """Strike == 0 → liquidity_score == 0 (no division by zero)."""
    row = {'strike': 0, 'volume': 10, 'open_interest': 100,
           'iv': 0.5, 'delta': 0.5, 'gamma': 0.01, 'theta': -5, 'vega': 0.2,
           'instrument_name': 'BTC-X', 'option_type': 'call'}
    metrics = scanner.calculate_metrics(row, 50000)
    assert metrics['liquidity_score'] == 0


def test_calculate_metrics_vol_oi_ratio_zero_oi(scanner):
    """Open interest == 0 → vol_oi_ratio == 0 (no division by zero)."""
    row = {'strike': 50000, 'volume': 5, 'open_interest': 0,
           'iv': 0.5, 'delta': 0.5, 'gamma': 0.01, 'theta': -5, 'vega': 0.2,
           'instrument_name': 'BTC-Y', 'option_type': 'put'}
    metrics = scanner.calculate_metrics(row, 50000)
    assert metrics['vol_oi_ratio'] == 0


def test_calculate_metrics_moneyness(scanner):
    """Call ITM when strike < underlying_price."""
    row = {'strike': 40000, 'volume': 5, 'open_interest': 20,
           'iv': 0.6, 'delta': 0.7, 'gamma': 0.01, 'theta': -10, 'vega': 0.3,
           'instrument_name': 'BTC-Z', 'option_type': 'call'}
    metrics = scanner.calculate_metrics(row, 50000)
    assert metrics['in_the_money'] is True
    assert metrics['moneyness'] == pytest.approx(-0.2)


def test_calculate_metrics_zero_underlying(scanner):
    """underlying_price == 0 → moneyness == 0, no division by zero."""
    row = {'strike': 50000, 'volume': 5, 'open_interest': 20,
           'iv': 0.6, 'delta': 0.7, 'gamma': 0.01, 'theta': -10, 'vega': 0.3,
           'instrument_name': 'BTC-W', 'option_type': 'call'}
    metrics = scanner.calculate_metrics(row, 0)
    assert metrics['moneyness'] == 0
    assert metrics['in_the_money'] is False


def test_calculate_metrics_long_short_risk_call(scanner):
    row = {
        'strike': 50000, 'volume': 5, 'open_interest': 20,
        'iv': 0.6, 'delta': 0.7, 'gamma': 0.01, 'theta': -10, 'vega': 0.3,
        'instrument_name': 'BTC-CALL', 'option_type': 'call', 'mark_price': 0.02
    }
    metrics = scanner.calculate_metrics(row, 50000)
    assert metrics['premium_quote'] == pytest.approx(1000)
    assert metrics['long_max_loss'] == pytest.approx(1000)
    assert metrics['short_max_profit'] == pytest.approx(1000)
    assert metrics['long_max_profit'] == float('inf')
    assert metrics['short_max_loss'] == float('inf')


def test_calculate_metrics_long_short_risk_put(scanner):
    row = {
        'strike': 45000, 'volume': 5, 'open_interest': 20,
        'iv': 0.6, 'delta': -0.3, 'gamma': 0.01, 'theta': -10, 'vega': 0.3,
        'instrument_name': 'BTC-PUT', 'option_type': 'put', 'mark_price': 0.01
    }
    metrics = scanner.calculate_metrics(row, 50000)
    assert metrics['premium_quote'] == pytest.approx(500)
    assert metrics['long_max_profit'] == pytest.approx(44500)
    assert metrics['short_max_loss'] == pytest.approx(44500)


# ---------------------------------------------------------------------------
# _apply_filters
# ---------------------------------------------------------------------------

def make_row(**overrides):
    base = {
        'iv': 0.5, 'delta': 0.3, 'dte': 30,
        'volume': 5.0, 'open_interest': 50.0, 'liquidity_score': 1.0
    }
    base.update(overrides)
    return base


DEFAULT_FILTERS = OptionFilters(
    iv_min=0.2, iv_max=1.0,
    delta_min=-0.5, delta_max=0.5,
    dte_min=7, dte_max=90,
    min_volume=1.0, min_open_interest=10.0,
    min_liquidity_score=0.0
)


def test_apply_filters_passes(scanner):
    assert scanner._apply_filters(make_row(), DEFAULT_FILTERS, 50000) is True


def test_apply_filters_iv_too_low(scanner):
    assert scanner._apply_filters(make_row(iv=0.1), DEFAULT_FILTERS, 50000) is False


def test_apply_filters_iv_too_high(scanner):
    assert scanner._apply_filters(make_row(iv=1.5), DEFAULT_FILTERS, 50000) is False


def test_apply_filters_delta_out_of_range(scanner):
    assert scanner._apply_filters(make_row(delta=0.8), DEFAULT_FILTERS, 50000) is False


def test_apply_filters_no_delta(scanner):
    """Missing delta → rejected."""
    row = make_row()
    del row['delta']
    assert scanner._apply_filters(row, DEFAULT_FILTERS, 50000) is False


def test_apply_filters_dte_too_low(scanner):
    assert scanner._apply_filters(make_row(dte=3), DEFAULT_FILTERS, 50000) is False


def test_apply_filters_dte_too_high(scanner):
    assert scanner._apply_filters(make_row(dte=200), DEFAULT_FILTERS, 50000) is False


def test_apply_filters_volume_too_low(scanner):
    assert scanner._apply_filters(make_row(volume=0.5), DEFAULT_FILTERS, 50000) is False


def test_apply_filters_oi_too_low(scanner):
    assert scanner._apply_filters(make_row(open_interest=5.0), DEFAULT_FILTERS, 50000) is False


def test_apply_filters_underlying_price_range(scanner):
    filters = OptionFilters(underlying_price_range=(40000, 60000))
    assert scanner._apply_filters(make_row(), filters, 50000) is True
    assert scanner._apply_filters(make_row(), filters, 70000) is False
    assert scanner._apply_filters(make_row(), filters, 30000) is False


def test_parse_instrument_market_inverse(scanner):
    parsed = scanner._parse_instrument_market("BTC-28JUN24-70000-C")
    assert parsed["pair_type"] == "inverse"
    assert parsed["quote_currency"] == "USD"


def test_parse_instrument_market_non_inverse(scanner):
    parsed = scanner._parse_instrument_market("BTC_USDC-28JUN24-70000-C")
    assert parsed["pair_type"] == "non_inverse"
    assert parsed["quote_currency"] == "USDC"


# ---------------------------------------------------------------------------
# get_instruments — mocked
# ---------------------------------------------------------------------------

def test_get_instruments_returns_list(scanner):
    with patch.object(scanner, '_get', return_value=[{'name': 'BTC-1'}]) as mock_get:
        result = scanner.get_instruments('BTC')
        mock_get.assert_called_once()
        assert isinstance(result, list)
        assert result[0]['name'] == 'BTC-1'


def test_get_instruments_non_list_returns_empty(scanner):
    with patch.object(scanner, '_get', return_value={'unexpected': True}):
        result = scanner.get_instruments('BTC')
        assert result == []


def test_get_supported_option_currencies(scanner):
    mocked = [
        {"currency": "BTC", "kind": "option"},
        {"currency": "ETH", "kind": "option"},
        {"currency": "BTC", "kind": "future"},
        {"currency": "XRP", "kind": "option"},
    ]
    with patch.object(scanner, '_get', return_value=mocked):
        result = scanner.get_supported_option_currencies()
        assert result == ["BTC", "ETH", "XRP"]


# ---------------------------------------------------------------------------
# _get_index_price — mocked
# ---------------------------------------------------------------------------

def test_get_index_price(scanner):
    with patch.object(scanner, '_get', return_value={'index_price': 65000.0}):
        price = scanner._get_index_price('BTC')
        assert price == 65000.0


def test_get_index_price_empty_response(scanner):
    with patch.object(scanner, '_get', return_value={}):
        price = scanner._get_index_price('BTC')
        assert price == 0


# ---------------------------------------------------------------------------
# export_to_csv — mocked filesystem
# ---------------------------------------------------------------------------

def test_export_to_csv(tmp_path, scanner):
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    out = str(tmp_path / "test.csv")
    scanner.export_to_csv(df, out)
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == ['a', 'b']
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# OptionFilters defaults
# ---------------------------------------------------------------------------

def test_option_filters_defaults():
    f = OptionFilters()
    assert f.currency == "BTC"
    assert f.iv_min == 0.0
    assert f.iv_max == 2.0
    assert f.exclude_perpetual is True
    assert f.underlying_price_range == (0, float('inf'))


def test_option_filters_independent_defaults():
    """Each OptionFilters instance has its own underlying_price_range tuple."""
    f1 = OptionFilters()
    f2 = OptionFilters()
    assert f1.underlying_price_range is not f2.underlying_price_range
