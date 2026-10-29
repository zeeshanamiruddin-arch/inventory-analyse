"""Database tables.

Raw ERP data (sales lines, stock snapshots, purchase order lines, stock
transfers) is stored as imported. The nightly run derives daily demand,
forecasts, recommendations and alerts from it. Recommendations are never
acted on automatically: each one waits for a person to approve or reject it.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Item(Base):
    __tablename__ = "items"
    code: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    item_group: Mapped[str | None] = mapped_column(String(128), index=True)
    parent_group: Mapped[str | None] = mapped_column(String(128), index=True)  # brand / parent item group
    uom: Mapped[str | None] = mapped_column(String(32))
    unit_cost: Mapped[float | None] = mapped_column(Float)
    supplier_code: Mapped[str | None] = mapped_column(String(64), index=True)  # last supplier ordered from


class Warehouse(Base):
    __tablename__ = "warehouses"
    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(32), index=True)
    sellable: Mapped[bool] = mapped_column(Boolean, default=False)


class Supplier(Base):
    __tablename__ = "suppliers"
    code: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    lead_time_days: Mapped[float | None] = mapped_column(Float)  # learned: median PO date -> last receipt


class SalesLine(Base):
    __tablename__ = "sales_lines"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_no: Mapped[str] = mapped_column(String(64))
    posting_date: Mapped[date] = mapped_column(Date)
    warehouse: Mapped[str] = mapped_column(String(128))
    item_code: Mapped[str] = mapped_column(String(64))
    qty: Mapped[float] = mapped_column(Float)
    net_amount: Mapped[float] = mapped_column(Float)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    customer_no: Mapped[str | None] = mapped_column(String(64))
    customer_name: Mapped[str | None] = mapped_column(String(255))
    sales_person: Mapped[str | None] = mapped_column(String(128))
    transaction_type: Mapped[str | None] = mapped_column(String(32))
    __table_args__ = (
        Index("ix_sales_item_wh_date", "item_code", "warehouse", "posting_date"),
        Index("ix_sales_date", "posting_date"),
    )


class StockLevel(Base):
    """Latest stock balance per item and warehouse."""

    __tablename__ = "stock_levels"
    item_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    warehouse: Mapped[str] = mapped_column(String(128), primary_key=True)
    qty: Mapped[float] = mapped_column(Float, default=0.0)
    value: Mapped[float] = mapped_column(Float, default=0.0)
    valuation_rate: Mapped[float] = mapped_column(Float, default=0.0)
    reserved: Mapped[float] = mapped_column(Float, default=0.0)
    as_of: Mapped[date] = mapped_column(Date)


class StockSnapshot(Base):
    """Every stock balance ever imported, used to find unexplained stock changes."""

    __tablename__ = "stock_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of: Mapped[date] = mapped_column(Date)
    item_code: Mapped[str] = mapped_column(String(64))
    warehouse: Mapped[str] = mapped_column(String(128))
    qty: Mapped[float] = mapped_column(Float)
    valuation_rate: Mapped[float] = mapped_column(Float, default=0.0)
    __table_args__ = (Index("ix_snap_asof_item_wh", "as_of", "item_code", "warehouse"),)


class PurchaseOrderLine(Base):
    __tablename__ = "purchase_order_lines"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    po_no: Mapped[str] = mapped_column(String(64), index=True)
    supplier_code: Mapped[str] = mapped_column(String(64), index=True)
    po_date: Mapped[date] = mapped_column(Date)
    item_code: Mapped[str] = mapped_column(String(64), index=True)
    warehouse: Mapped[str | None] = mapped_column(String(128))
    qty: Mapped[float] = mapped_column(Float)
    received_qty: Mapped[float] = mapped_column(Float, default=0.0)
    cancelled_qty: Mapped[float] = mapped_column(Float, default=0.0)
    remaining_qty: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str | None] = mapped_column(String(32))
    currency: Mapped[str] = mapped_column(String(8), default="SAR")
    price_sar: Mapped[float] = mapped_column(Float, default=0.0)
    last_receipt: Mapped[date | None] = mapped_column(Date)


class StockTransfer(Base):
    """Submitted material transfers between warehouses (one row per item line)."""

    __tablename__ = "stock_transfers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entry_id: Mapped[str] = mapped_column(String(64), index=True)
    posting_date: Mapped[date] = mapped_column(Date, index=True)
    item_code: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(128))
    destination: Mapped[str] = mapped_column(String(128))
    qty: Mapped[float] = mapped_column(Float)
    value: Mapped[float] = mapped_column(Float, default=0.0)
    in_transit: Mapped[bool] = mapped_column(Boolean, default=False)  # dispatched, not yet received


class DailyDemand(Base):
    __tablename__ = "daily_demand"
    item_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    warehouse: Mapped[str] = mapped_column(String(128), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    qty: Mapped[float] = mapped_column(Float)  # units sold net of returns, floored at 0
    __table_args__ = (Index("ix_demand_wh_day", "warehouse", "day"),)


class Forecast(Base):
    __tablename__ = "forecasts"
    item_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    warehouse: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_date: Mapped[date] = mapped_column(Date)
    daily_mean: Mapped[float] = mapped_column(Float)
    daily_std: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(16))  # SES or TSB
    pattern: Mapped[str] = mapped_column(String(16))  # smooth / erratic / intermittent / lumpy / none
    abc: Mapped[str] = mapped_column(String(1))
    xyz: Mapped[str] = mapped_column(String(1))
    last_sale: Mapped[date | None] = mapped_column(Date)
    sale_days: Mapped[int] = mapped_column(Integer, default=0)  # days with a sale in the history window
    in_range: Mapped[bool] = mapped_column(Boolean, default=True, index=True)  # stocked at this warehouse


class Recommendation(Base):
    """A suggested purchase or transfer. Nothing happens until a person approves it."""

    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[date] = mapped_column(Date, index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)  # purchase | transfer
    item_code: Mapped[str] = mapped_column(String(64), index=True)
    from_warehouse: Mapped[str | None] = mapped_column(String(128))
    to_warehouse: Mapped[str | None] = mapped_column(String(128), index=True)
    supplier_code: Mapped[str | None] = mapped_column(String(64), index=True)
    qty: Mapped[float] = mapped_column(Float)
    value: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending|approved|rejected|superseded
    created_by: Mapped[str] = mapped_column(String(64), default="system")  # system | agent
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    decided_by: Mapped[str | None] = mapped_column(String(128))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime)
    decision_note: Mapped[str | None] = mapped_column(Text)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[date] = mapped_column(Date, index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str] = mapped_column(String(8))  # high | medium | low
    item_code: Mapped[str | None] = mapped_column(String(64), index=True)
    warehouse: Mapped[str | None] = mapped_column(String(128))
    value: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)


class ImportLog(Base):
    __tablename__ = "import_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    file_name: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(16))
    rows_loaded: Mapped[int] = mapped_column(Integer)
    date_from: Mapped[date | None] = mapped_column(Date)
    date_to: Mapped[date | None] = mapped_column(Date)
    report: Mapped[str] = mapped_column(Text)  # JSON data-quality report
    __table_args__ = (UniqueConstraint("imported_at", "file_name", name="uq_import_once"),)
