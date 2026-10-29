"""The nightly run: forecast -> replenishment suggestions -> alerts.

Works one warehouse at a time so memory stays flat at 200k SKUs x 20 branches.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from . import anomalies
from .config import Settings, get_settings
from .forecasting import abc_classes, forecast_warehouse
from .importers.loader import bulk_insert, latest_sales_date, latest_stock_date, previous_snapshot_date, records
from .models import (
    Alert,
    DailyDemand,
    Forecast,
    Item,
    PurchaseOrderLine,
    Recommendation,
    SalesLine,
    StockLevel,
    StockSnapshot,
    StockTransfer,
    Supplier,
    Warehouse,
)
from .replenishment import Inputs, plan

log = logging.getLogger(__name__)


def _df(session: Session, stmt, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(session.execute(stmt).all(), columns=columns)


def load_inputs(session: Session, forecasts: pd.DataFrame | None = None) -> Inputs:
    if forecasts is None:
        forecasts = _df(session, select(Forecast.item_code, Forecast.warehouse, Forecast.daily_mean, Forecast.daily_std,
                                        Forecast.abc, Forecast.in_range),
                        ["item_code", "warehouse", "daily_mean", "daily_std", "abc", "in_range"])
    return Inputs(
        forecasts=forecasts,
        stock=_df(session, select(StockLevel.item_code, StockLevel.warehouse, StockLevel.qty, StockLevel.reserved, StockLevel.valuation_rate),
                  ["item_code", "warehouse", "qty", "reserved", "valuation_rate"]),
        in_transit=_df(session, select(StockTransfer.item_code, StockTransfer.destination, StockTransfer.qty).where(StockTransfer.in_transit.is_(True)),
                       ["item_code", "destination", "qty"]),
        open_po=_df(session, select(PurchaseOrderLine.item_code, PurchaseOrderLine.po_no, PurchaseOrderLine.remaining_qty).where(PurchaseOrderLine.remaining_qty > 0),
                    ["item_code", "po_no", "remaining_qty"]),
        items=_df(session, select(Item.code, Item.unit_cost, Item.supplier_code, Item.parent_group),
                  ["code", "unit_cost", "supplier_code", "parent_group"]),
        supplier_lead=pd.Series(dict(session.execute(select(Supplier.code, Supplier.lead_time_days).where(Supplier.lead_time_days.is_not(None))).all()), dtype=float),
        warehouses=_df(session, select(Warehouse.name, Warehouse.category), ["name", "category"]),
    )


def run_forecasts(session: Session, as_of: date, cfg: Settings) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    start = as_of - timedelta(days=cfg.history_days - 1)
    value = _df(session, select(SalesLine.item_code, func.sum(SalesLine.net_amount))
                .where(SalesLine.posting_date.between(start, as_of)).group_by(SalesLine.item_code), ["item_code", "value"])
    abc = abc_classes(value.set_index("item_code")["value"]) if len(value) else pd.Series(dtype=object)

    stock = _df(session, select(StockLevel.item_code, StockLevel.warehouse, StockLevel.qty), ["item_code", "warehouse", "qty"])
    sellable = list(session.scalars(select(Warehouse.name).where(Warehouse.sellable.is_(True)).order_by(Warehouse.name)))
    category = dict(session.execute(select(Warehouse.name, Warehouse.category)).all())
    frames, last_sales, alerts = [], [], []
    for wh in sellable:
        demand = _df(session, select(DailyDemand.item_code, DailyDemand.day, DailyDemand.qty)
                     .where(DailyDemand.warehouse == wh, DailyDemand.day.between(start, as_of)), ["item_code", "day", "qty"])
        last = _df(session, select(DailyDemand.item_code, func.max(DailyDemand.day)).where(DailyDemand.warehouse == wh)
                   .group_by(DailyDemand.item_code), ["item_code", "last_sale"])
        last["warehouse"] = wh
        last_sales.append(last)
        fc = forecast_warehouse(demand, as_of, cfg.history_days)
        if len(fc):
            fc["warehouse"] = wh
            fc["abc"] = fc["item_code"].map(abc).fillna("C")
            # Central keeps everything it sells; a branch only its regular sellers.
            fc["in_range"] = True if category.get(wh) != "Branch" else fc["sale_days"] >= cfg.branch_range_min_sale_days
            fc = fc.merge(last[["item_code", "last_sale"]], on="item_code", how="left")
            frames.append(fc)
        alerts.append(anomalies.demand_changes(demand.assign(warehouse=wh), stock[stock["warehouse"] == wh], as_of, cfg))
    cols = ["item_code", "warehouse", "daily_mean", "daily_std", "method", "pattern", "abc", "xyz", "last_sale",
            "sale_days", "in_range"]
    forecasts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    forecasts["run_date"] = as_of
    session.execute(delete(Forecast))
    bulk_insert(session, Forecast, records(forecasts, cols + ["run_date"]))
    last_sale = pd.concat(last_sales, ignore_index=True) if last_sales else pd.DataFrame(columns=["item_code", "last_sale", "warehouse"])
    return forecasts, last_sale, alerts


SPIKE_NOTE = " WARNING: driven by an unexplained demand spike (see Alerts); confirm the demand is real before approving."


def flag_spikes(transfers: pd.DataFrame, purchases: pd.DataFrame, demand_alerts: list[pd.DataFrame]) -> None:
    spikes = [a[a["kind"] == "demand_spike"] for a in demand_alerts if len(a)]
    if not spikes:
        return
    sp = pd.concat(spikes)
    pairs = set(zip(sp["item_code"], sp["warehouse"]))
    items = set(sp["item_code"])
    if len(transfers):
        hit = [(i, w) in pairs for i, w in zip(transfers["item_code"], transfers["to_warehouse"])]
        transfers.loc[hit, "reason"] += SPIKE_NOTE
    if len(purchases):
        hit = purchases["item_code"].isin(items)
        purchases.loc[hit, "reason"] += SPIKE_NOTE


def save_recommendations(session: Session, transfers: pd.DataFrame, purchases: pd.DataFrame, as_of: date) -> tuple[int, int]:
    # Yesterday's open system suggestions are replaced by today's; agent drafts and decisions are kept.
    session.execute(update(Recommendation).where(Recommendation.status == "pending", Recommendation.created_by == "system")
                    .values(status="superseded"))
    rows = [{"run_date": as_of, "kind": "transfer", "item_code": r.item_code, "from_warehouse": r.from_warehouse,
             "to_warehouse": r.to_warehouse, "supplier_code": None, "qty": float(r.qty), "value": float(r.value),
             "reason": r.reason, "status": "pending", "created_by": "system"} for r in transfers.itertuples()]
    rows += [{"run_date": as_of, "kind": "purchase", "item_code": r.item_code, "from_warehouse": None, "to_warehouse": None,
              "supplier_code": None if pd.isna(r.supplier_code) else r.supplier_code, "qty": float(r.qty),
              "value": float(r.value), "reason": r.reason, "status": "pending", "created_by": "system"}
             for r in purchases.itertuples()]
    bulk_insert(session, Recommendation, rows)
    return len(transfers), len(purchases)


def run_alerts(session: Session, as_of: date, inp: Inputs, forecasts: pd.DataFrame, last_sale: pd.DataFrame,
               demand_alerts: list[pd.DataFrame], cfg: Settings) -> pd.DataFrame:
    sellable = set(inp.warehouses.loc[inp.warehouses["category"].isin(["Central", "Branch"]), "name"])
    parts = list(demand_alerts)
    parts.append(anomalies.dead_stock(inp.stock, last_sale, sellable, as_of, cfg))
    parts.append(anomalies.negative_stock(inp.stock))

    month_ago = as_of - timedelta(days=29)
    sales30 = _df(session, select(SalesLine.item_code, SalesLine.warehouse, SalesLine.qty, SalesLine.net_amount, SalesLine.cost)
                  .where(SalesLine.posting_date.between(month_ago, as_of)), ["item_code", "warehouse", "qty", "net_amount", "cost"])
    parts.append(anomalies.below_cost_sales(sales30))

    company_mean = forecasts.groupby("item_code")["daily_mean"].sum() if len(forecasts) else pd.Series(dtype=float)
    unit_cost = inp.items.set_index("code")["unit_cost"]
    parts.append(anomalies.dead_on_order(inp.open_po, company_mean, unit_cost))

    cur_date = latest_stock_date(session)
    prev_date = previous_snapshot_date(session, cur_date) if cur_date else None
    if prev_date:
        snap = lambda d: _df(session, select(StockSnapshot.item_code, StockSnapshot.warehouse, StockSnapshot.qty, StockSnapshot.valuation_rate)
                             .where(StockSnapshot.as_of == d), ["item_code", "warehouse", "qty", "valuation_rate"])
        between = lambda col: col > prev_date
        sales = _df(session, select(SalesLine.item_code, SalesLine.warehouse, SalesLine.qty)
                    .where(between(SalesLine.posting_date), SalesLine.posting_date <= cur_date), ["item_code", "warehouse", "qty"])
        transfers = _df(session, select(StockTransfer.item_code, StockTransfer.source, StockTransfer.destination, StockTransfer.qty, StockTransfer.in_transit)
                        .where(between(StockTransfer.posting_date), StockTransfer.posting_date <= cur_date),
                        ["item_code", "source", "destination", "qty", "in_transit"])
        parts.append(anomalies.unexplained_loss(snap(prev_date), snap(cur_date), sales, transfers, prev_date, cur_date, cfg))

    parts = [p for p in parts if len(p)]
    alerts = pd.concat(parts, ignore_index=True) if parts else anomalies._empty()
    session.execute(delete(Alert).where(Alert.acknowledged.is_(False)))
    alerts["run_date"] = as_of
    bulk_insert(session, Alert, records(alerts, anomalies.COLUMNS + ["run_date"]))
    return alerts


def run_nightly(session: Session, as_of: date | None = None, cfg: Settings | None = None) -> dict[str, Any]:
    cfg = cfg or get_settings()
    t0 = time.time()
    as_of = as_of or latest_sales_date(session) or date.today()
    forecasts, last_sale, demand_alerts = run_forecasts(session, as_of, cfg)
    log.info("forecasts: %d series in %.1fs", len(forecasts), time.time() - t0)
    inp = load_inputs(session, forecasts[["item_code", "warehouse", "daily_mean", "daily_std", "abc", "in_range"]])
    transfers, purchases = plan(inp, cfg)
    flag_spikes(transfers, purchases, demand_alerts)
    n_t, n_p = save_recommendations(session, transfers, purchases, as_of)
    alerts = run_alerts(session, as_of, inp, forecasts, last_sale, demand_alerts, cfg)
    ranged = forecasts[forecasts["in_range"].astype(bool) & forecasts["warehouse"].map(
        dict(zip(inp.warehouses["name"], inp.warehouses["category"]))).eq("Branch")]
    return {
        "as_of": as_of.isoformat(),
        "series_forecast": int(len(forecasts)),
        "avg_branch_range": int(round(ranged.groupby("warehouse").size().mean())) if len(ranged) else 0,
        "transfer_suggestions": n_t,
        "purchase_suggestions": n_p,
        "purchase_value": round(float(purchases["value"].sum()) if len(purchases) else 0.0, 2),
        "alerts": alerts["kind"].value_counts().to_dict() if len(alerts) else {},
        "seconds": round(time.time() - t0, 1),
    }
