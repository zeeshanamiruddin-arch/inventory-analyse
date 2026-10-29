"""Find things a person should look at.

* demand_spike      - last week's sales far above the item's normal level
* demand_drop       - a steady seller that suddenly stopped selling while in stock
* dead_stock        - stock with no sale at that warehouse for DEAD_STOCK_DAYS
* unexplained_loss  - stock fell between two stock imports by more than sales
                      and transfers explain (possible shrinkage, theft or
                      unrecorded movement)
* negative_stock    - the ERP shows a negative balance
* below_cost_sales  - items sold under cost in the last 30 days
* dead_on_order     - open purchase orders for items nobody is buying
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import Settings

COLUMNS = ["kind", "severity", "item_code", "warehouse", "value", "message"]
DEAD_STOCK_MIN_VALUE = 500.0
BELOW_COST_MIN_LOSS = 200.0


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def demand_changes(demand: pd.DataFrame, stock: pd.DataFrame, as_of: date, cfg: Settings) -> pd.DataFrame:
    """demand: item_code, warehouse, day, qty over at least the history window."""
    if demand.empty:
        return _empty()
    w = cfg.spike_window_days
    recent_start = as_of - timedelta(days=w - 1)
    base_start = as_of - timedelta(days=cfg.history_days - 1)
    base_days = (recent_start - base_start).days
    d = demand.assign(day=pd.to_datetime(demand["day"]).dt.date)
    recent = d[d["day"] >= recent_start].groupby(["item_code", "warehouse"])["qty"].sum().rename("recent")
    base = d[(d["day"] >= base_start) & (d["day"] < recent_start)]
    agg = base.assign(sq=base["qty"] ** 2).groupby(["item_code", "warehouse"]).agg(s=("qty", "sum"), s2=("sq", "sum"))
    week = base.assign(wk=(pd.to_datetime(base["day"]) - pd.Timestamp(base_start)).dt.days // w)
    best = week.groupby(["item_code", "warehouse", "wk"])["qty"].sum().groupby(level=[0, 1]).max().rename("best_week")
    df = agg.join(best).join(recent, how="outer").fillna(0.0)
    df["mean"] = df["s"] / base_days
    df["std"] = np.sqrt(np.maximum(df["s2"] / base_days - df["mean"] ** 2, 0))
    exp = df["mean"] * w
    sd = np.maximum(df["std"] * np.sqrt(w), np.sqrt(np.maximum(exp, 1)))
    out = []
    # A spike must be statistically unusual AND beat the best week of the whole baseline.
    spike = df[(df["recent"] >= 5) & (df["recent"] > exp + cfg.spike_sigma * sd) & (df["recent"] > 1.5 * df["best_week"])]
    for (item, wh), r in spike.iterrows():
        ratio = r["recent"] / exp[(item, wh)] if exp[(item, wh)] > 0 else np.inf
        out.append({
            "kind": "demand_spike", "severity": "high" if ratio >= 10 and r["recent"] >= 20 else "medium", "item_code": item,
            "warehouse": wh, "value": float(r["recent"]),
            "message": (f"Sold {r['recent']:g} in the last {w} days vs a normal {exp[(item, wh)]:.1f}. "
                        "Check for a bulk order, a pricing error or a new customer before restocking."),
        })
    on_hand = stock.set_index(["item_code", "warehouse"])["qty"] if not stock.empty else pd.Series(dtype=float)
    drop = df[(df["mean"] >= 1.0) & (df["recent"] == 0)]
    for (item, wh), r in drop.iterrows():
        qty = float(on_hand.get((item, wh), 0.0))
        if qty <= 0:
            continue  # out of stock explains it; replenishment handles that
        out.append({
            "kind": "demand_drop", "severity": "medium", "item_code": item, "warehouse": wh, "value": qty,
            "message": (f"Normally sells {r['mean']:.1f}/day but nothing in the last {w} days, with {qty:g} on hand. "
                        "Check the stock is findable and correctly located."),
        })
    return pd.DataFrame(out, columns=COLUMNS)


def dead_stock(stock: pd.DataFrame, last_sale: pd.DataFrame, sellable: set[str], as_of: date, cfg: Settings) -> pd.DataFrame:
    s = stock[(stock["qty"] > 0) & stock["warehouse"].isin(sellable)].merge(last_sale, on=["item_code", "warehouse"], how="left")
    cutoff = as_of - timedelta(days=cfg.dead_stock_days)
    dead = s[s["last_sale"].isna() | (pd.to_datetime(s["last_sale"]).dt.date < cutoff)].copy()
    dead["value"] = dead["qty"] * dead["valuation_rate"]
    dead = dead[dead["value"] >= DEAD_STOCK_MIN_VALUE].sort_values("value", ascending=False)
    msgs = [
        f"{r.qty:g} units worth {r.value:,.0f} SAR, " + (
            "no sale on record here." if pd.isna(r.last_sale) else f"last sold here {r.last_sale}.")
        + " Consider transferring to a branch that sells it, or clearance."
        for r in dead.itertuples()
    ]
    return pd.DataFrame({
        "kind": "dead_stock", "severity": np.where(dead["value"] >= 10_000, "high", "low"),
        "item_code": dead["item_code"], "warehouse": dead["warehouse"], "value": dead["value"].round(2), "message": msgs,
    }, columns=COLUMNS)


def unexplained_loss(prev: pd.DataFrame, cur: pd.DataFrame, sales: pd.DataFrame, transfers: pd.DataFrame,
                     prev_date: date, cur_date: date, cfg: Settings) -> pd.DataFrame:
    """Stock that disappeared between two imports. Receipts can only add stock, so a loss is never caused by them."""
    key = ["item_code", "warehouse"]
    p = prev.set_index(key)["qty"].rename("prev")
    c = cur.set_index(key)[["qty", "valuation_rate"]]
    sold = sales.groupby(key)["qty"].sum().rename("sold") if not sales.empty else pd.Series(dtype=float, name="sold")
    t_out = transfers.groupby(["item_code", "source"])["qty"].sum().rename("out") if not transfers.empty else pd.Series(dtype=float, name="out")
    t_out.index.names = key
    arrived = transfers[~transfers["in_transit"]] if not transfers.empty else transfers
    t_in = arrived.groupby(["item_code", "destination"])["qty"].sum().rename("received") if not arrived.empty else pd.Series(dtype=float, name="received")
    t_in.index.names = key
    df = pd.concat([p, c, sold, t_out, t_in], axis=1).fillna(0.0)
    df = df[df["prev"] > 0]
    df["expected"] = df["prev"] - df["sold"] - df["out"] + df["received"]
    df["missing"] = df["expected"] - df["qty"]
    df["value"] = df["missing"] * df["valuation_rate"]
    bad = df[(df["missing"] >= 1) & (df["value"] >= cfg.shrinkage_min_value)].sort_values("value", ascending=False)
    rows = [{
        "kind": "unexplained_loss", "severity": "high" if r.value >= 5_000 else "medium",
        "item_code": item, "warehouse": wh, "value": round(float(r.value), 2),
        "message": (f"{r.missing:g} units ({r.value:,.0f} SAR) missing between {prev_date} and {cur_date}: "
                    f"had {r.prev:g}, sold {r.sold:g}, sent out {r.out:g}, received by transfer {r.received:g}, "
                    f"expected {r.expected:g}, counted {r.qty:g}. Recount and check for unrecorded movements."),
    } for (item, wh), r in zip(bad.index, bad.itertuples(index=False))]
    return pd.DataFrame(rows, columns=COLUMNS)


def negative_stock(stock: pd.DataFrame) -> pd.DataFrame:
    neg = stock[stock["qty"] < 0]
    return pd.DataFrame({
        "kind": "negative_stock", "severity": "medium", "item_code": neg["item_code"], "warehouse": neg["warehouse"],
        "value": neg["qty"],
        "message": [f"ERP balance is {q:g}. Something was issued before it was received; fix the posting." for q in neg["qty"]],
    }, columns=COLUMNS)


def below_cost_sales(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales[(sales["qty"] > 0) & (sales["net_amount"] < sales["cost"])]
    if s.empty:
        return _empty()
    g = s.groupby(["item_code", "warehouse"]).agg(lines=("qty", "size"), loss=("cost", "sum"), net=("net_amount", "sum"))
    g["loss"] = g["loss"] - g["net"]
    g = g[g["loss"] >= BELOW_COST_MIN_LOSS].sort_values("loss", ascending=False)
    return pd.DataFrame([{
        "kind": "below_cost_sales", "severity": "medium", "item_code": item, "warehouse": wh, "value": round(float(r.loss), 2),
        "message": f"{int(r.lines)} sale lines under cost in the last 30 days, {r.loss:,.0f} SAR below cost. Check prices and discounts.",
    } for (item, wh), r in g.iterrows()], columns=COLUMNS)


def dead_on_order(open_po: pd.DataFrame, company_mean: pd.Series, unit_cost: pd.Series) -> pd.DataFrame:
    po = open_po[open_po["remaining_qty"] > 0].groupby("item_code").agg(qty=("remaining_qty", "sum"), pos=("po_no", "nunique"))
    po = po[~po.index.isin(company_mean[company_mean > 0].index)]
    po["value"] = po["qty"] * po.index.map(unit_cost).fillna(0.0)
    po = po.sort_values("value", ascending=False)
    return pd.DataFrame([{
        "kind": "dead_on_order", "severity": "high" if r.value >= 10_000 else "medium", "item_code": item,
        "warehouse": None, "value": round(float(r.value), 2),
        "message": f"{r.qty:g} units on {int(r.pos)} open PO(s) but no demand forecast anywhere. Consider delaying or cancelling.",
    } for item, r in po.iterrows()], columns=COLUMNS)
