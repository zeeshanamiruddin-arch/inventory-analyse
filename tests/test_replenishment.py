import pandas as pd
import pytest

from inventory_ai.config import Settings
from inventory_ai.replenishment import Inputs, plan, z_value

WH = pd.DataFrame({"name": ["CPD - SP", "P01 - A - SP", "P02 - B - SP", "Slow Moving - SP"],
                   "category": ["Central", "Branch", "Branch", "Slow moving"]})


def inputs(fc, stock, open_po=None, transit=None, items=None):
    return Inputs(
        forecasts=pd.DataFrame(fc, columns=["item_code", "warehouse", "daily_mean", "daily_std", "abc"]),
        stock=pd.DataFrame(stock, columns=["item_code", "warehouse", "qty", "reserved", "valuation_rate"]),
        in_transit=pd.DataFrame(transit or [], columns=["item_code", "destination", "qty"]),
        open_po=pd.DataFrame(open_po or [], columns=["item_code", "po_no", "remaining_qty"]),
        items=pd.DataFrame(items or [["X", 10.0, "S1", "Toyota"]], columns=["code", "unit_cost", "supplier_code", "parent_group"]),
        supplier_lead=pd.Series({"S1": 30.0}),
        warehouses=WH,
    )


def test_z_values():
    assert z_value(0.95) == pytest.approx(1.645, abs=1e-3)


def test_branch_short_is_filled_from_central_surplus_not_bought():
    cfg = Settings(min_transfer_value=0)
    inp = inputs(
        fc=[["X", "P01 - A - SP", 2.0, 1.0, "A"], ["X", "CPD - SP", 1.0, 1.0, "A"]],
        stock=[["X", "P01 - A - SP", 0, 0, 10], ["X", "CPD - SP", 500, 0, 10]],
    )
    transfers, purchases = plan(inp, cfg)
    assert len(transfers) == 1
    t = transfers.iloc[0]
    assert (t.from_warehouse, t.to_warehouse) == ("CPD - SP", "P01 - A - SP")
    assert t.qty > 2.0 * (cfg.transfer_lead_days + cfg.review_period_days)  # covers lead time + cycle
    assert purchases.empty  # company holds plenty


def test_branch_with_no_demand_gives_away_its_stock():
    cfg = Settings(min_transfer_value=0)
    inp = inputs(
        fc=[["X", "P01 - A - SP", 1.0, 0.5, "B"]],
        stock=[["X", "P01 - A - SP", 0, 0, 10], ["X", "P02 - B - SP", 40, 0, 10]],
    )
    transfers, _ = plan(inp, cfg)
    assert list(transfers["from_warehouse"]) == ["P02 - B - SP"]


def test_purchase_counts_open_orders_and_transit():
    cfg = Settings()
    fc = [["X", "P01 - A - SP", 3.0, 1.5, "A"]]
    short = plan(inputs(fc, [["X", "P01 - A - SP", 10, 0, 10]]), cfg)[1]
    assert len(short) == 1 and short.iloc[0]["supplier_code"] == "S1"
    covered = plan(inputs(fc, [["X", "P01 - A - SP", 10, 0, 10]], open_po=[["X", "PO1", 500]]), cfg)[1]
    assert covered.empty
    via_transit = plan(inputs(fc, [["X", "P01 - A - SP", 10, 0, 10]], transit=[["X", "P01 - A - SP", 500]]), cfg)[1]
    assert via_transit.empty


def test_non_selling_stores_are_not_counted():
    cfg = Settings()
    fc = [["X", "P01 - A - SP", 3.0, 1.5, "A"]]
    p = plan(inputs(fc, [["X", "P01 - A - SP", 0, 0, 10], ["X", "Slow Moving - SP", 999, 0, 10]]), cfg)[1]
    assert len(p) == 1


def test_new_item_gets_brand_supplier_marked_as_guess():
    cfg = Settings()
    inp = inputs([["NEW", "P01 - A - SP", 3.0, 1.5, "A"]], [["NEW", "P01 - A - SP", 0, 0, 10]],
                 items=[["NEW", 10.0, None, "Toyota"], ["OLD1", 5.0, "S1", "Toyota"], ["OLD2", 5.0, "S1", "Toyota"],
                        ["OTHER", 5.0, "S9", "Kia"]])
    p = plan(inp, cfg)[1]
    assert p.iloc[0]["supplier_code"] == "S1" and "guessed" in p.iloc[0]["reason"]


def test_branch_is_only_topped_up_for_its_range():
    cfg = Settings(min_transfer_value=0)
    inp = inputs(
        fc=[["X", "P01 - A - SP", 0.5, 0.5, "A"], ["X", "CPD - SP", 1.0, 1.0, "A"]],
        stock=[["X", "P01 - A - SP", 0, 0, 10], ["X", "CPD - SP", 500, 0, 10]],
    )
    inp.forecasts["in_range"] = [False, True]  # P01 sold it as a one-off special order
    transfers, purchases = plan(inp, cfg)
    assert transfers.empty
    # ...but that demand still counts when deciding what the company must buy.
    inp.stock.loc[inp.stock["warehouse"] == "CPD - SP", "qty"] = 0
    assert plan(inp, cfg)[1].iloc[0]["qty"] > 0


def test_stock_outside_a_branch_range_is_offered_to_others():
    cfg = Settings(min_transfer_value=0)
    inp = inputs(
        fc=[["X", "P01 - A - SP", 1.0, 0.5, "B"], ["X", "P02 - B - SP", 0.05, 0.2, "B"]],
        stock=[["X", "P01 - A - SP", 0, 0, 10], ["X", "P02 - B - SP", 8, 0, 10]],
    )
    inp.forecasts["in_range"] = [True, False]
    transfers, _ = plan(inp, cfg)
    t = transfers.iloc[0]
    assert (t.from_warehouse, t.qty) == ("P02 - B - SP", 8) and "range" in t.reason
