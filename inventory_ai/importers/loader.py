"""Write parsed ERP exports into the database.

The exports are cumulative (year to date), so each load replaces the rows
inside the file's date range instead of appending duplicates. The stock
balance replaces the current stock and is also kept as a dated snapshot.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import and_, delete, func, insert, select, update
from sqlalchemy.orm import Session

from ..models import (
    DailyDemand,
    ImportLog,
    Item,
    PurchaseOrderLine,
    SalesLine,
    StockLevel,
    StockSnapshot,
    StockTransfer,
    Supplier,
    Warehouse,
)
from ..warehouses import classify
from .erp_excel import ParsedFile

CHUNK = 20_000


def _py(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.date()
    return v


def records(df: pd.DataFrame, columns: Iterable[str]) -> list[dict]:
    cols = list(columns)
    return [{c: _py(v) for c, v in zip(cols, row)} for row in df[cols].itertuples(index=False, name=None)]


def bulk_insert(session: Session, model, rows: list[dict]) -> None:
    if not rows:
        return
    if session.get_bind().dialect.name == "postgresql" and len(rows) > 5_000:
        _copy(session, model, rows)
        return
    for i in range(0, len(rows), CHUNK):
        session.execute(insert(model), rows[i : i + CHUNK])


def _copy(session: Session, model, rows: list[dict]) -> None:
    """PostgreSQL COPY, an order of magnitude faster than INSERT for millions of rows."""
    table = model.__table__
    cols = list(rows[0].keys())
    # Columns with server/Python defaults that the rows leave out are filled in here.
    for c in table.columns:
        if c.name not in cols and c.default is not None and not c.primary_key and not callable(c.default.arg):
            cols.append(c.name)
    defaults = {c.name: c.default.arg for c in table.columns if c.default is not None and not callable(c.default.arg)}
    raw = session.connection().connection.driver_connection
    with raw.cursor() as cur:
        with cur.copy(f'COPY {table.name} ({", ".join(cols)}) FROM STDIN') as cp:
            for r in rows:
                cp.write_row(tuple(r.get(c, defaults.get(c)) for c in cols))


# ------------------------------------------------------------ dimensions ---

def ensure_warehouses(session: Session, names: Iterable[str]) -> None:
    existing = set(session.scalars(select(Warehouse.name)))
    new = []
    for n in sorted({n for n in names if n} - existing):
        info = classify(n)
        new.append({"name": n, "label": info.label, "category": info.category, "sellable": info.sellable})
    bulk_insert(session, Warehouse, new)


def upsert_items(session: Session, df: pd.DataFrame) -> None:
    """Insert unknown items; fill in name/group/uom/parent group where the new file has them."""
    fields = [c for c in ("item_name", "item_group", "parent_group", "uom") if c in df.columns]
    latest = df.drop_duplicates("item_code", keep="last").set_index("item_code")[fields]
    existing = {
        r.code: r
        for r in session.execute(select(Item.code, Item.name, Item.item_group, Item.parent_group, Item.uom))
    }
    new, changed = [], []
    for code, row in latest.iterrows():
        vals = {
            "name": row.get("item_name") or None,
            "item_group": row.get("item_group") or None,
            "parent_group": row.get("parent_group") or None,
            "uom": row.get("uom") or None,
        }
        cur = existing.get(code)
        if cur is None:
            new.append({"code": code, **vals})
            continue
        merged = {k: v or getattr(cur, k) for k, v in vals.items()}
        if any(merged[k] != getattr(cur, k) for k in merged):
            changed.append({"code": code, **merged})
    bulk_insert(session, Item, new)
    for i in range(0, len(changed), CHUNK):
        session.execute(update(Item), changed[i : i + CHUNK])


# ------------------------------------------------------------------ facts ---

def load_sales(session: Session, pf: ParsedFile) -> int:
    df = pf.frame
    upsert_items(session, df)
    ensure_warehouses(session, df["warehouse"].unique())
    if pf.date_from and pf.date_to:
        session.execute(delete(SalesLine).where(SalesLine.posting_date.between(pf.date_from, pf.date_to)))
    cols = ["invoice_no", "posting_date", "warehouse", "item_code", "qty", "net_amount", "cost",
            "customer_no", "customer_name", "sales_person", "transaction_type"]
    bulk_insert(session, SalesLine, records(df, cols))
    if pf.date_from and pf.date_to:
        rebuild_daily_demand(session, pf.date_from, pf.date_to)
    return len(df)


def rebuild_daily_demand(session: Session, start: date, end: date) -> None:
    session.execute(delete(DailyDemand).where(DailyDemand.day.between(start, end)))
    total = func.sum(SalesLine.qty)
    agg = (
        select(SalesLine.item_code, SalesLine.warehouse, SalesLine.posting_date, total)
        .where(SalesLine.posting_date.between(start, end))
        .group_by(SalesLine.item_code, SalesLine.warehouse, SalesLine.posting_date)
        .having(total > 0)
    )
    session.execute(insert(DailyDemand).from_select(["item_code", "warehouse", "day", "qty"], agg))


def load_stock(session: Session, pf: ParsedFile) -> int:
    df = pf.frame
    as_of = pf.date_to or date.today()
    upsert_items(session, df)
    ensure_warehouses(session, df["warehouse"].unique())
    # One row per item and warehouse (exports can split a balance across batches).
    agg = df.groupby(["item_code", "warehouse"], as_index=False).agg(
        qty=("qty", "sum"), value=("value", "sum"), reserved=("reserved", "sum"))
    agg["valuation_rate"] = np.where(agg["qty"] != 0, agg["value"] / agg["qty"].where(agg["qty"] != 0, 1), 0.0)
    agg["as_of"] = as_of
    session.execute(delete(StockLevel))
    bulk_insert(session, StockLevel, records(agg, ["item_code", "warehouse", "qty", "value", "valuation_rate", "reserved", "as_of"]))
    session.execute(delete(StockSnapshot).where(StockSnapshot.as_of == as_of))
    bulk_insert(session, StockSnapshot, records(agg, ["as_of", "item_code", "warehouse", "qty", "valuation_rate"]))

    # Company-wide unit cost per item, weighted by quantity on hand.
    pos = agg[agg["qty"] > 0]
    cost = (pos.groupby("item_code")["value"].sum() / pos.groupby("item_code")["qty"].sum()).dropna()
    rows = [{"code": c, "unit_cost": float(v)} for c, v in cost.items() if v > 0]
    for i in range(0, len(rows), CHUNK):
        session.execute(update(Item), rows[i : i + CHUNK])
    return len(agg)


def load_purchase(session: Session, pf: ParsedFile) -> int:
    df = pf.frame
    upsert_items(session, df)
    ensure_warehouses(session, df["warehouse"].replace("", np.nan).dropna().unique())
    existing = set(session.scalars(select(Supplier.code)))
    sup = df.drop_duplicates("supplier_code")[["supplier_code", "supplier_name"]]
    bulk_insert(session, Supplier, [
        {"code": r.supplier_code, "name": r.supplier_name} for r in sup.itertuples() if r.supplier_code not in existing
    ])
    if pf.date_from and pf.date_to:
        session.execute(delete(PurchaseOrderLine).where(PurchaseOrderLine.po_date.between(pf.date_from, pf.date_to)))
    cols = ["po_no", "supplier_code", "po_date", "item_code", "warehouse", "qty", "received_qty", "cancelled_qty",
            "remaining_qty", "status", "currency", "price_sar", "last_receipt"]
    bulk_insert(session, PurchaseOrderLine, records(df, cols))
    session.flush()
    refresh_supplier_stats(session)
    return len(df)


def refresh_supplier_stats(session: Session) -> None:
    """Learn each supplier's lead time and each item's latest supplier from PO history."""
    rows = session.execute(
        select(PurchaseOrderLine.supplier_code, PurchaseOrderLine.item_code, PurchaseOrderLine.po_date,
               PurchaseOrderLine.last_receipt)
    ).all()
    if not rows:
        return
    po = pd.DataFrame(rows, columns=["supplier_code", "item_code", "po_date", "last_receipt"])
    done = po.dropna(subset=["last_receipt"])
    if len(done):
        lead = (pd.to_datetime(done["last_receipt"]) - pd.to_datetime(done["po_date"])).dt.days
        med = lead[lead >= 0].groupby(done["supplier_code"]).median()
        session.execute(update(Supplier), [{"code": c, "lead_time_days": float(v)} for c, v in med.items()])
    last = po.sort_values("po_date").drop_duplicates("item_code", keep="last")
    known = set(session.scalars(select(Item.code)))
    ups = [{"code": r.item_code, "supplier_code": r.supplier_code} for r in last.itertuples() if r.item_code in known]
    for i in range(0, len(ups), CHUNK):
        session.execute(update(Item), ups[i : i + CHUNK])


def load_transfers(session: Session, pf: ParsedFile) -> int:
    df = pf.frame
    ensure_warehouses(session, pd.concat([df["source"], df["destination"]]).unique())
    if pf.date_from and pf.date_to:
        session.execute(delete(StockTransfer).where(StockTransfer.posting_date.between(pf.date_from, pf.date_to)))
    cols = ["entry_id", "posting_date", "item_code", "source", "destination", "qty", "value", "in_transit"]
    bulk_insert(session, StockTransfer, records(df, cols))
    return len(df)


LOADERS = {"sales": load_sales, "stock": load_stock, "purchase": load_purchase, "transfer": load_transfers}


def load(session: Session, pf: ParsedFile) -> ImportLog:
    n = LOADERS[pf.kind](session, pf)
    log = ImportLog(file_name=pf.file_name, kind=pf.kind, rows_loaded=n, date_from=pf.date_from,
                    date_to=pf.date_to, report=json.dumps(pf.report, default=str))
    session.add(log)
    session.flush()
    return log


def latest_stock_date(session: Session) -> date | None:
    return session.scalar(select(func.max(StockLevel.as_of)))


def latest_sales_date(session: Session) -> date | None:
    return session.scalar(select(func.max(SalesLine.posting_date)))


def previous_snapshot_date(session: Session, before: date) -> date | None:
    return session.scalar(select(func.max(StockSnapshot.as_of)).where(and_(StockSnapshot.as_of < before)))
