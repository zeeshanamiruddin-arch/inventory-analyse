"""Read-only questions about the inventory, shared by the API and the AI agent,
plus the two draft actions the agent may take (they only create pending
suggestions; a person still has to approve them)."""
from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Select, case, func, or_, select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .models import (
    Alert,
    Forecast,
    ImportLog,
    Item,
    PurchaseOrderLine,
    Recommendation,
    StockLevel,
    StockTransfer,
    Supplier,
    Warehouse,
)
from .replenishment import policy, z_value
from .warehouses import BRANCH, CENTRAL


def _rows(session: Session, stmt: Select) -> list[dict[str, Any]]:
    return [dict(r._mapping) for r in session.execute(stmt)]


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 3)
    if isinstance(v, date):
        return v.isoformat()
    return v


def clean(rows: list[dict]) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in rows]


def resolve_warehouse(session: Session, text: str | None) -> str | None:
    """Accept a full warehouse name or a branch label like 'Jeddah Safa'."""
    if not text:
        return None
    exact = session.scalar(select(Warehouse.name).where(or_(Warehouse.name == text, Warehouse.label == text)))
    if exact:
        return exact
    like = list(session.scalars(select(Warehouse.name).where(Warehouse.name.ilike(f"%{text}%")).limit(2)))
    return like[0] if len(like) == 1 else None


def summary(session: Session) -> dict[str, Any]:
    sellable = select(Warehouse.name).where(Warehouse.sellable.is_(True))
    stock_value = session.scalar(select(func.coalesce(func.sum(StockLevel.value), 0)).where(StockLevel.warehouse.in_(sellable)))
    pending = session.execute(select(Recommendation.kind, func.count(), func.coalesce(func.sum(Recommendation.value), 0))
                              .where(Recommendation.status == "pending").group_by(Recommendation.kind)).all()
    alerts = session.execute(select(Alert.kind, func.count(), func.coalesce(func.sum(Alert.value), 0))
                             .where(Alert.acknowledged.is_(False)).group_by(Alert.kind)).all()
    out_of_stock = session.scalar(
        select(func.count()).select_from(Forecast)
        .join(Warehouse, Warehouse.name == Forecast.warehouse)
        .outerjoin(StockLevel, (StockLevel.item_code == Forecast.item_code) & (StockLevel.warehouse == Forecast.warehouse))
        .where(Warehouse.category == BRANCH, Forecast.in_range.is_(True), func.coalesce(StockLevel.qty, 0) <= 0))
    branch_range = (select(func.count().label("n")).select_from(Forecast).join(Warehouse, Warehouse.name == Forecast.warehouse)
                    .where(Warehouse.category == BRANCH, Forecast.in_range.is_(True)).group_by(Forecast.warehouse).subquery())
    imports = session.execute(select(ImportLog.kind, func.max(ImportLog.date_to), func.max(ImportLog.imported_at))
                              .group_by(ImportLog.kind)).all()
    return clean([{
        "stock_value_sellable": float(stock_value or 0),
        "items": session.scalar(select(func.count()).select_from(Item)),
        "warehouses": session.scalar(select(func.count()).select_from(Warehouse).where(Warehouse.sellable.is_(True))),
        "forecast_run": session.scalar(select(func.max(Forecast.run_date))),
        "avg_branch_range": session.scalar(select(func.avg(branch_range.c.n))),
        "branch_lines_out_of_stock": out_of_stock,
        "pending": {k: {"count": n, "value": float(v)} for k, n, v in pending},
        "alerts": {k: {"count": n, "value": float(v)} for k, n, v in alerts},
        "last_imports": {k: {"data_to": d.isoformat() if d else None, "imported_at": t.isoformat() if t else None} for k, d, t in imports},
    }])[0]


def search_items(session: Session, text: str, limit: int = 20) -> list[dict]:
    pat = f"%{text}%"
    q = (select(Item.code, Item.name, Item.item_group, Item.parent_group, Item.unit_cost)
         .where(or_(Item.code.ilike(pat), Item.name.ilike(pat))).order_by(Item.code).limit(limit))
    return clean(_rows(session, q))


def item_status(session: Session, code: str, cfg: Settings | None = None) -> dict[str, Any] | None:
    cfg = cfg or get_settings()
    item = session.get(Item, code)
    if item is None:
        return None
    stock = pd.DataFrame(_rows(session, select(StockLevel.warehouse, StockLevel.qty, StockLevel.reserved, StockLevel.value)
                               .where(StockLevel.item_code == code)), columns=["warehouse", "qty", "reserved", "value"])
    fc = pd.DataFrame(_rows(session, select(Forecast.warehouse, Forecast.daily_mean, Forecast.daily_std, Forecast.pattern,
                                            Forecast.abc, Forecast.last_sale, Forecast.in_range).where(Forecast.item_code == code)),
                      columns=["warehouse", "daily_mean", "daily_std", "pattern", "abc", "last_sale", "in_range"])
    transit = pd.DataFrame(_rows(session, select(StockTransfer.destination.label("warehouse"), func.sum(StockTransfer.qty).label("inbound"))
                                 .where(StockTransfer.item_code == code, StockTransfer.in_transit.is_(True)).group_by(StockTransfer.destination)),
                           columns=["warehouse", "inbound"])
    cats = dict(session.execute(select(Warehouse.name, Warehouse.category)).all())
    df = stock.merge(fc, on="warehouse", how="outer").merge(transit, on="warehouse", how="outer")
    for c in ("qty", "reserved", "value", "daily_mean", "daily_std", "inbound"):
        df[c] = df[c].fillna(0.0)
    df["category"] = df["warehouse"].map(cats)
    df["in_range"] = df["in_range"].astype("boolean").fillna(False).astype(bool) | (df["category"] == CENTRAL)
    df["abc"] = df["abc"].fillna(fc["abc"].iloc[0] if len(fc) else "C")
    z = df["abc"].map(lambda a: z_value(cfg.service_level(a)))
    _, rop, out = policy(df["daily_mean"].to_numpy(), df["daily_std"].to_numpy(), z.to_numpy(),
                         float(cfg.transfer_lead_days), cfg.review_period_days, cfg.order_cycle_days)
    df["reorder_point"] = np.where(df["in_range"], rop, 0.0)
    df["order_up_to"] = np.where(df["in_range"], out, 0.0)
    df["position"] = (df["qty"] - df["reserved"]).clip(lower=0) + df["inbound"]
    df["days_of_cover"] = [p / m if m > 0 else None for p, m in zip(df["position"], df["daily_mean"])]
    df = df.sort_values(["category", "warehouse"])
    open_po = _rows(session, select(PurchaseOrderLine.po_no, PurchaseOrderLine.supplier_code, PurchaseOrderLine.po_date,
                                    PurchaseOrderLine.remaining_qty, PurchaseOrderLine.warehouse)
                    .where(PurchaseOrderLine.item_code == code, PurchaseOrderLine.remaining_qty > 0))
    recs = _rows(session, select(Recommendation.id, Recommendation.kind, Recommendation.from_warehouse, Recommendation.to_warehouse,
                                 Recommendation.qty, Recommendation.value, Recommendation.status, Recommendation.created_by)
                 .where(Recommendation.item_code == code, Recommendation.status == "pending"))
    alerts = _rows(session, select(Alert.kind, Alert.severity, Alert.warehouse, Alert.message)
                   .where(Alert.item_code == code, Alert.acknowledged.is_(False)))
    return {
        "item": clean([{"code": item.code, "name": item.name, "group": item.item_group, "brand": item.parent_group,
                        "unit_cost": item.unit_cost, "supplier": item.supplier_code}])[0],
        "company": clean([{"on_hand": float(df.loc[df["category"].isin([CENTRAL, BRANCH]), "qty"].sum()),
                           "daily_demand": float(df["daily_mean"].sum()),
                           "on_order": float(sum(p["remaining_qty"] for p in open_po))}])[0],
        "warehouses": clean(df[["warehouse", "category", "in_range", "qty", "reserved", "inbound", "daily_mean", "pattern", "abc",
                                "last_sale", "reorder_point", "order_up_to", "position", "days_of_cover"]].to_dict("records")),
        "open_purchase_orders": clean(open_po),
        "pending_suggestions": clean(recs),
        "alerts": clean(alerts),
    }


def list_recommendations(session: Session, kind: str | None = None, status: str = "pending", warehouse: str | None = None,
                         supplier: str | None = None, item: str | None = None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    q = select(Recommendation, Item.name.label("item_name")).outerjoin(Item, Item.code == Recommendation.item_code)
    if status != "all":
        q = q.where(Recommendation.status == status)
    if kind:
        q = q.where(Recommendation.kind == kind)
    if warehouse:
        q = q.where(or_(Recommendation.to_warehouse == warehouse, Recommendation.from_warehouse == warehouse))
    if supplier:
        q = q.where(Recommendation.supplier_code == supplier)
    if item:
        q = q.where(Recommendation.item_code == item)
    sub = q.subquery()
    agg = session.execute(select(func.count(), func.coalesce(func.sum(sub.c.value), 0))).one()
    rows = session.execute(q.order_by(Recommendation.value.desc(), Recommendation.id).limit(limit).offset(offset)).all()
    out = [{
        "id": r.id, "kind": r.kind, "item_code": r.item_code, "item_name": name, "from_warehouse": r.from_warehouse,
        "to_warehouse": r.to_warehouse, "supplier_code": r.supplier_code, "qty": r.qty, "value": r.value, "reason": r.reason,
        "status": r.status, "created_by": r.created_by, "run_date": r.run_date, "decided_by": r.decided_by,
        "decided_at": r.decided_at.isoformat() if r.decided_at else None, "decision_note": r.decision_note,
    } for r, name in rows]
    return {"total": agg[0], "total_value": round(float(agg[1]), 2), "rows": clean(out)}


def list_alerts(session: Session, kind: str | None = None, severity: str | None = None, warehouse: str | None = None,
                item: str | None = None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    q = select(Alert).where(Alert.acknowledged.is_(False))
    if kind:
        q = q.where(Alert.kind == kind)
    if severity:
        q = q.where(Alert.severity == severity)
    if warehouse:
        q = q.where(Alert.warehouse == warehouse)
    if item:
        q = q.where(Alert.item_code == item)
    n = session.scalar(select(func.count()).select_from(q.subquery()))
    sev = case((Alert.severity == "high", 0), (Alert.severity == "medium", 1), else_=2)
    rows = session.scalars(q.order_by(sev, Alert.value.desc()).limit(limit).offset(offset))
    return {"total": n, "rows": clean([{
        "id": a.id, "kind": a.kind, "severity": a.severity, "item_code": a.item_code, "warehouse": a.warehouse,
        "value": a.value, "message": a.message, "run_date": a.run_date} for a in rows])}


def stockout_risk(session: Session, warehouse: str | None = None, within_days: float = 14, limit: int = 50) -> list[dict]:
    """Selling lines whose stock (plus anything in transit) runs out within N days at the forecast rate."""
    inbound = (select(StockTransfer.item_code, StockTransfer.destination, func.sum(StockTransfer.qty).label("inbound"))
               .where(StockTransfer.in_transit.is_(True)).group_by(StockTransfer.item_code, StockTransfer.destination).subquery())
    on_hand = func.coalesce(StockLevel.qty, 0) - func.coalesce(StockLevel.reserved, 0)
    position = case((on_hand > 0, on_hand), else_=0) + func.coalesce(inbound.c.inbound, 0)
    q = (select(Forecast.item_code, Item.name, Forecast.warehouse, Forecast.daily_mean, Forecast.abc,
                position.label("position"), (position / Forecast.daily_mean).label("days_of_cover"))
         .join(Item, Item.code == Forecast.item_code)
         .outerjoin(StockLevel, (StockLevel.item_code == Forecast.item_code) & (StockLevel.warehouse == Forecast.warehouse))
         .outerjoin(inbound, (inbound.c.item_code == Forecast.item_code) & (inbound.c.destination == Forecast.warehouse))
         .where(Forecast.in_range.is_(True), Forecast.daily_mean >= 0.05, position < Forecast.daily_mean * within_days))
    if warehouse:
        q = q.where(Forecast.warehouse == warehouse)
    q = q.order_by(Forecast.abc, (Forecast.daily_mean * within_days - position).desc()).limit(limit)
    return clean(_rows(session, q))


def warehouse_summary(session: Session, warehouse: str) -> dict[str, Any] | None:
    wh = session.get(Warehouse, warehouse)
    if wh is None:
        return None
    stock = session.execute(select(func.count(), func.coalesce(func.sum(StockLevel.value), 0))
                            .where(StockLevel.warehouse == warehouse, StockLevel.qty > 0)).one()
    sales_rate = session.scalar(select(func.coalesce(func.sum(Forecast.daily_mean), 0)).where(Forecast.warehouse == warehouse))
    range_size = session.scalar(select(func.count()).select_from(Forecast).where(Forecast.warehouse == warehouse, Forecast.in_range.is_(True)))
    pend_in = session.execute(select(func.count(), func.coalesce(func.sum(Recommendation.value), 0))
                              .where(Recommendation.status == "pending", Recommendation.to_warehouse == warehouse)).one()
    pend_out = session.execute(select(func.count(), func.coalesce(func.sum(Recommendation.value), 0))
                               .where(Recommendation.status == "pending", Recommendation.from_warehouse == warehouse)).one()
    alerts = dict(session.execute(select(Alert.kind, func.count()).where(Alert.warehouse == warehouse, Alert.acknowledged.is_(False))
                                  .group_by(Alert.kind)).all())
    return clean([{
        "warehouse": wh.name, "label": wh.label, "category": wh.category, "lines_in_stock": stock[0],
        "stock_value": float(stock[1]), "forecast_units_per_day": float(sales_rate or 0), "items_in_range": range_size,
        "pending_transfers_in": {"count": pend_in[0], "value": float(pend_in[1])},
        "pending_transfers_out": {"count": pend_out[0], "value": float(pend_out[1])},
        "alerts": alerts,
        "stockout_risk_14d": len(stockout_risk(session, warehouse, 14, 500)),
    }])[0]


def list_warehouses(session: Session) -> list[dict]:
    return _rows(session, select(Warehouse.name, Warehouse.label, Warehouse.category, Warehouse.sellable).order_by(Warehouse.category, Warehouse.name))


# ------------------------------------------------------------- draft actions ---

class DraftError(ValueError):
    pass


def draft_transfer(session: Session, item_code: str, from_wh: str, to_wh: str, qty: float, reason: str,
                   created_by: str = "agent") -> Recommendation:
    if qty <= 0:
        raise DraftError("qty must be positive")
    if session.get(Item, item_code) is None:
        raise DraftError(f"unknown item {item_code}")
    src, dst = resolve_warehouse(session, from_wh), resolve_warehouse(session, to_wh)
    if not src or not dst:
        raise DraftError(f"unknown warehouse: {from_wh if not src else to_wh}")
    if src == dst:
        raise DraftError("source and destination are the same warehouse")
    lvl = session.get(StockLevel, (item_code, src))
    available = (lvl.qty - lvl.reserved) if lvl else 0.0
    if available < qty:
        raise DraftError(f"{src} has only {available:g} available")
    rate = lvl.valuation_rate if lvl else 0.0
    rec = Recommendation(run_date=date.today(), kind="transfer", item_code=item_code, from_warehouse=src, to_warehouse=dst,
                         qty=qty, value=round(qty * rate, 2), reason=reason, status="pending", created_by=created_by)
    session.add(rec)
    session.flush()
    return rec


def draft_purchase(session: Session, item_code: str, qty: float, reason: str, supplier_code: str | None = None,
                   created_by: str = "agent") -> Recommendation:
    if qty <= 0:
        raise DraftError("qty must be positive")
    item = session.get(Item, item_code)
    if item is None:
        raise DraftError(f"unknown item {item_code}")
    supplier = supplier_code or item.supplier_code
    if supplier and session.get(Supplier, supplier) is None:
        raise DraftError(f"unknown supplier {supplier}")
    rec = Recommendation(run_date=date.today(), kind="purchase", item_code=item_code, supplier_code=supplier, qty=qty,
                         value=round(qty * (item.unit_cost or 0.0), 2), reason=reason, status="pending", created_by=created_by)
    session.add(rec)
    session.flush()
    return rec
