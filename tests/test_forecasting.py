from datetime import date

import numpy as np
import pandas as pd
import pytest

from inventory_ai.forecasting import abc_classes, classify_pattern, forecast_warehouse, ses, tsb


def test_ses_constant_series():
    level, rmse = ses(np.full((2, 60), 3.0))
    assert level == pytest.approx([3.0, 3.0]) and rmse == pytest.approx([0.0, 0.0])


def test_tsb_matches_rate_of_regular_intermittent_demand():
    y = np.zeros((1, 400))
    y[0, ::5] = 10  # 10 units every 5th day = 2/day
    mean, _ = tsb(y)
    assert mean[0] == pytest.approx(2.0, rel=0.1)


def test_tsb_decays_when_item_stops_selling():
    y = np.zeros((1, 200))
    y[0, :100:4] = 4
    mean, _ = tsb(y)
    assert mean[0] < 0.2  # was ~1/day, nothing sold for 100 days


def test_patterns():
    y = np.zeros((4, 100))
    y[0] = 5  # smooth
    y[1, ::2] = np.tile([1, 30], 25)  # erratic? no: ADI=2 -> lumpy
    y[2, ::7] = 3  # intermittent
    pattern, adi, _ = classify_pattern(y)
    assert list(pattern) == ["smooth", "lumpy", "intermittent", "none"]
    assert adi[0] == 1.0


def test_abc_classes():
    v = pd.Series({"a": 800, "b": 100, "c": 60, "d": 40, "e": 0})
    abc = abc_classes(v)
    assert abc.to_dict() == {"a": "A", "b": "B", "c": "B", "d": "C"}


def test_forecast_warehouse_uses_tsb_for_slow_movers():
    days = pd.date_range("2026-01-01", periods=120)
    rows = [{"item_code": "FAST", "day": d.date(), "qty": 4.0} for d in days]
    rows += [{"item_code": "SLOW", "day": d.date(), "qty": 2.0} for d in days[::10]]
    fc = forecast_warehouse(pd.DataFrame(rows), date(2026, 4, 30), 120).set_index("item_code")
    assert fc.loc["FAST", "method"] == "SES" and fc.loc["FAST", "daily_mean"] == pytest.approx(4.0)
    assert fc.loc["SLOW", "method"] == "TSB" and fc.loc["SLOW", "daily_mean"] == pytest.approx(0.2, rel=0.3)
    assert fc.loc["SLOW", "daily_std"] >= np.sqrt(fc.loc["SLOW", "daily_mean"]) - 1e-9


def test_sale_days_counts_days_with_sales():
    rows = [{"item_code": "A", "day": date(2026, 4, d), "qty": 1.0} for d in (1, 5, 9)]
    fc = forecast_warehouse(pd.DataFrame(rows), date(2026, 4, 30), 60)
    assert fc.loc[0, "sale_days"] == 3
