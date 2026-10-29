"""Turn forecasts and stock into transfer and purchase suggestions.

Two steps, both producing suggestions that wait for human approval:

1. Rebalance (transfers). A branch whose stock position (on hand - reserved
   + in transit to it) is below its reorder point is topped up to its
   order-up-to level from stock that other warehouses do not need: first the
   central store (CPD), then branches holding more than DONOR_KEEP_DAYS of
   their own demand, or stock they do not sell at all.

2. Buy (purchases), at company level per item. Company position = sellable
   stock + stock in transit + open purchase orders. Below the reorder point
   for the supplier's lead time, a purchase tops it up to the order-up-to
   level. Transfers move stock around but do not change company position, so
   the two steps never double count.

Reorder point  = daily mean x (lead time + review period) + safety stock
Safety stock   = z(service level) x daily std x sqrt(lead time + review period)
Order-up-to    = reorder point + daily mean x order cycle
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import Settings
from .warehouses import BRANCH, CENTRAL

_Z: dict[float, float] = {}


def z_value(service_level: float) -> float:
    if service_level not in _Z:
        _Z[service_level] = NormalDist().inv_cdf(service_level)
    return _Z[service_level]


def policy(mean: np.ndarray, std: np.ndarray, z: np.ndarray, lead: np.ndarray, review: float, cycle: float):
    horizon = lead + review
    ss = z * std * np.sqrt(horizon)
    rop = mean * horizon + ss
    out = rop + mean * cycle
    return ss, rop, out


@dataclass
class Inputs:
    forecasts: pd.DataFrame  # item_code, warehouse, daily_mean, daily_std, abc[, in_range]
    stock: pd.DataFrame  # item_code, warehouse, qty, reserved, valuation_rate
    in_transit: pd.DataFrame  # item_code, destination, qty
    open_po: pd.DataFrame  # item_code, remaining_qty
    items: pd.DataFrame  # code, unit_cost, supplier_code[, parent_group]
    supplier_lead: pd.Series  # supplier_code -> days
    warehouses: pd.DataFrame  # name, category


def _grid(inp: Inputs) -> pd.DataFrame:
    wh = inp.warehouses.set_index("name")["category"]
    sell = set(wh[wh.isin([CENTRAL, BRANCH])].index)
    fc = inp.forecasts[inp.forecasts["warehouse"].isin(sell)]
    st = inp.stock[inp.stock["warehouse"].isin(sell)]
    if "in_range" not in fc.columns:
        fc = fc.assign(in_range=True)
    g = pd.merge(
        fc[["item_code", "warehouse", "daily_mean", "daily_std", "abc", "in_range"]],
        st[["item_code", "warehouse", "qty", "reserved", "valuation_rate"]],
        on=["item_code", "warehouse"], how="outer",
    )
    tr = inp.in_transit.groupby(["item_code", "destination"], as_index=False)["qty"].sum()
    tr = tr.rename(columns={"destination": "warehouse", "qty": "inbound"})
    g = g.merge(tr[tr["warehouse"].isin(sell)], on=["item_code", "warehouse"], how="outer")
    g["category"] = g["warehouse"].map(wh)
    for c in ("daily_mean", "daily_std", "qty", "reserved", "valuation_rate", "inbound"):
        g[c] = g[c].fillna(0.0)
    abc = inp.forecasts.groupby("item_code")["abc"].first()
    g["abc"] = g["abc"].fillna(g["item_code"].map(abc)).fillna("C")
    g["available"] = (g["qty"] - g["reserved"]).clip(lower=0)
    # Stock with no forecast (no sales here) is not part of this warehouse's range, except at central.
    g["in_range"] = g["in_range"].astype("boolean").fillna(False).astype(bool) | (g["category"] == CENTRAL)
    return g


def _unit_cost(inp: Inputs, g: pd.DataFrame) -> pd.Series:
    cost = inp.items.set_index("code")["unit_cost"].dropna()
    fallback = g[g["valuation_rate"] > 0].groupby("item_code")["valuation_rate"].median()
    return cost.combine_first(fallback)


def plan_transfers(g: pd.DataFrame, cfg: Settings, unit_cost: pd.Series) -> pd.DataFrame:
    z = g["abc"].map(lambda a: z_value(cfg.service_level(a))).to_numpy()
    mean, std = g["daily_mean"].to_numpy(), g["daily_std"].to_numpy()
    lead = np.full(len(g), float(cfg.transfer_lead_days))
    _, rop, out = policy(mean, std, z, lead, cfg.review_period_days, cfg.order_cycle_days)
    g = g.assign(rop=rop, order_up_to=out, position=g["available"] + g["inbound"])

    branch = g["category"] == BRANCH
    need = np.where(branch & g["in_range"] & (g["position"] < g["rop"]) & (g["daily_mean"] > 0),
                    np.ceil(g["order_up_to"] - g["position"]), 0)
    keep = np.where(
        g["category"] == CENTRAL,
        g["daily_mean"] * cfg.order_cycle_days + g["rop"],
        # A branch keeps what it needs of its range; anything outside its range can go.
        np.where(g["in_range"], np.maximum(g["daily_mean"] * cfg.donor_keep_days, g["order_up_to"]), 0),
    )
    surplus = np.floor(np.maximum(g["available"] - keep, 0))
    g = g.assign(need=need, surplus=surplus)

    takers = g[g["need"] > 0]
    givers = g[(g["surplus"] > 0) & g["item_code"].isin(takers["item_code"])]
    if givers.empty:
        return pd.DataFrame(columns=["item_code", "from_warehouse", "to_warehouse", "qty", "value", "reason"])
    givers = givers.assign(central=(givers["category"] == CENTRAL).astype(int)).sort_values(
        ["item_code", "central", "surplus"], ascending=[True, False, False])
    takers = takers[takers["item_code"].isin(givers["item_code"])].sort_values(
        ["item_code", "daily_mean"], ascending=[True, False])

    moves = []
    give_by_item = {k: v for k, v in givers.groupby("item_code", sort=False)}
    for item, tk in takers.groupby("item_code", sort=False):
        donors = [[r.warehouse, r.surplus, r.daily_mean, r.category, not r.in_range] for r in give_by_item[item].itertuples()]
        cost = float(unit_cost.get(item, 0.0) or 0.0)
        for t in tk.itertuples():
            left = t.need
            for d in donors:
                if left <= 0:
                    break
                if d[1] <= 0 or d[0] == t.warehouse:
                    continue
                q = min(left, d[1])
                d[1] -= q
                left -= q
                src_note = "central store surplus" if d[3] == CENTRAL else (
                    "not in that branch's range" if d[4] else f"{d[2]:.2f}/day there, above {cfg.donor_keep_days} days of cover")
                moves.append({
                    "item_code": item, "from_warehouse": d[0], "to_warehouse": t.warehouse, "qty": float(q),
                    "value": round(q * cost, 2),
                    "reason": (f"{t.warehouse}: position {t.position:g} is below reorder point {t.rop:.1f} "
                               f"(forecast {t.daily_mean:.2f}/day, class {t.abc}); top up to {t.order_up_to:.0f}. "
                               f"Source: {src_note}."),
                })
    mv = pd.DataFrame(moves)
    if mv.empty:
        return pd.DataFrame(columns=["item_code", "from_warehouse", "to_warehouse", "qty", "value", "reason"])
    return mv[mv["value"] >= cfg.min_transfer_value].reset_index(drop=True)


def plan_purchases(g: pd.DataFrame, inp: Inputs, cfg: Settings, unit_cost: pd.Series) -> pd.DataFrame:
    by = g.assign(var=g["daily_std"] ** 2).groupby("item_code").agg(
        daily_mean=("daily_mean", "sum"),
        var=("var", "sum"),
        available=("available", "sum"),
        inbound=("inbound", "sum"),
        abc=("abc", "first"),
    )
    by = by[by["daily_mean"] > 0]
    if by.empty:
        return pd.DataFrame(columns=["item_code", "supplier_code", "qty", "value", "reason"])
    on_order = inp.open_po.groupby("item_code")["remaining_qty"].sum()
    by["on_order"] = by.index.map(on_order).fillna(0.0)
    items = inp.items.set_index("code")
    by["supplier_code"] = pd.Series(by.index.map(items["supplier_code"]), index=by.index, dtype=object)
    by["supplier_guessed"] = False
    if "parent_group" in items.columns:
        # Never ordered before: suggest the supplier most used for the same brand.
        known = items.dropna(subset=["supplier_code", "parent_group"])
        usual = known.groupby("parent_group")["supplier_code"].agg(lambda s: s.value_counts().index[0])
        guess = by.index.map(items["parent_group"]).map(usual)
        miss = by["supplier_code"].isna() & pd.notna(guess)
        by.loc[miss, "supplier_code"] = guess[miss.to_numpy()]
        by.loc[miss, "supplier_guessed"] = True
    by["lead"] = by["supplier_code"].map(inp.supplier_lead).fillna(cfg.default_supplier_lead_days).astype(float)
    z = by["abc"].map(lambda a: z_value(cfg.service_level(a))).to_numpy()
    ss, rop, out = policy(by["daily_mean"].to_numpy(), np.sqrt(by["var"].to_numpy()), z, by["lead"].to_numpy(),
                          cfg.review_period_days, cfg.order_cycle_days)
    by["safety"], by["rop"], by["order_up_to"] = ss, rop, out
    by["position"] = by["available"] + by["inbound"] + by["on_order"]
    buy = by[by["position"] < by["rop"]].copy()
    buy["qty"] = np.ceil(buy["order_up_to"] - buy["position"])
    buy["unit_cost"] = buy.index.map(unit_cost).fillna(0.0)
    buy["value"] = (buy["qty"] * buy["unit_cost"]).round(2)
    buy["reason"] = [
        (f"Company position {r.position:g} (stock {r.available:g} + in transit {r.inbound:g} + on order {r.on_order:g}) "
         f"is below reorder point {r.rop:.1f} for a {r.lead:.0f}-day supplier lead time "
         f"(forecast {r.daily_mean:.2f}/day, safety stock {r.safety:.1f}, class {r.abc}). "
         f"Order up to {r.order_up_to:.0f}." + (" Supplier guessed from the brand: check it." if r.supplier_guessed else ""))
        for r in buy.itertuples()
    ]
    buy = buy.reset_index()[["item_code", "supplier_code", "qty", "value", "reason"]]
    return buy[buy["qty"] > 0].reset_index(drop=True)


def plan(inp: Inputs, cfg: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = _grid(inp)
    cost = _unit_cost(inp, g)
    return plan_transfers(g, cfg, cost), plan_purchases(g, inp, cfg, cost)


def item_policy_table(inp: Inputs, cfg: Settings) -> pd.DataFrame:
    """Per item and warehouse: position, reorder point and order-up-to level (for the API and agent)."""
    g = _grid(inp)
    z = g["abc"].map(lambda a: z_value(cfg.service_level(a))).to_numpy()
    ss, rop, out = policy(g["daily_mean"].to_numpy(), g["daily_std"].to_numpy(), z,
                          np.full(len(g), float(cfg.transfer_lead_days)), cfg.review_period_days, cfg.order_cycle_days)
    g = g.assign(safety_stock=ss, reorder_point=rop, order_up_to=out, position=g["available"] + g["inbound"])
    g["days_of_cover"] = np.where(g["daily_mean"] > 0, g["position"] / g["daily_mean"].where(g["daily_mean"] > 0, 1), math.inf)
    return g
