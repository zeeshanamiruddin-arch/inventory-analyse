"""Read the four ERP exports (xlsx, xls or csv) into clean DataFrames.

Supported exports, recognised from their header row:

* Spare Part Item Sales Detailed  -> kind "sales"
* Stock Balance                   -> kind "stock"
* Item-Wise Purchase History      -> kind "purchase"
* Stock Entry                     -> kind "transfer"

Every parser returns a ``ParsedFile`` with a data-quality report so a bad
row never disappears silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..warehouses import is_transit

SIGNATURES: dict[str, list[str]] = {
    "sales": ["Invoice Number", "Item Code", "Net Line Sales"],
    "stock": ["Item", "Warehouse", "Balance Qty", "Balance Value"],
    "purchase": ["PO No.", "Item", "Remaining Qty"],
    "transfer": ["ID", "Stock Entry Type", "Qty (Stock Entry Detail)"],
}

REQUIRED: dict[str, list[str]] = {
    "sales": ["Warehouse", "Invoice Number", "Posting Date", "Item Code", "Qty", "Net Line Sales", "Total Cost"],
    "stock": ["Item", "Warehouse", "Balance Qty", "Balance Value"],
    "purchase": ["PO No.", "PO Date", "Item", "PO Qty"],
    "transfer": ["ID", "docstatus", "Stock Entry Type", "Posting Date", "Qty (Stock Entry Detail)", "Item Code (Stock Entry Detail)"],
}

CLOSED_PO_STATUSES = {"closed", "cancelled", "completed", "to bill"}
EXCEL_EPOCH = datetime(1899, 12, 30)


class ImportError_(ValueError):
    """Raised when a file is not one of the supported exports or is missing columns."""


@dataclass
class ParsedFile:
    kind: str
    file_name: str
    frame: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)
    date_from: date | None = None
    date_to: date | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def norm(h: Any) -> str:
    return re.sub(r"\s+", " ", str("" if h is None or (isinstance(h, float) and np.isnan(h)) else h)).strip().lower()


# ---------------------------------------------------------------- reading ---

def read_rows(path: str | Path) -> pd.DataFrame:
    """Read the first sheet with no header interpretation (all cells as objects)."""
    p = Path(path)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p, header=None, dtype=object, keep_default_na=False, na_values=[""])
    return pd.read_excel(p, header=None, dtype=object, engine=None)


def find_header_row(raw: pd.DataFrame, must: Iterable[str], scan: int = 15) -> int:
    need = [norm(m) for m in must]
    for i in range(min(len(raw), scan)):
        cells = {norm(v) for v in raw.iloc[i].tolist()}
        if all(n in cells for n in need):
            return i
    return -1


def detect(raw: pd.DataFrame) -> str | None:
    for kind, sig in SIGNATURES.items():
        if find_header_row(raw, sig) >= 0:
            return kind
    return None


def _frame(raw: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, list[str]]:
    hr = find_header_row(raw, SIGNATURES[kind])
    header = [norm(h) for h in raw.iloc[hr].tolist()]
    df = raw.iloc[hr + 1 :].copy()
    # Keep the first occurrence of duplicated column names (exports sometimes repeat a date column).
    seen: set[str] = set()
    keep = []
    for j, h in enumerate(header):
        if h and h not in seen:
            seen.add(h)
            keep.append(j)
    df = df.iloc[:, keep]
    df.columns = [header[j] for j in keep]
    df = df.dropna(how="all").reset_index(drop=True)
    missing = [c for c in REQUIRED[kind] if norm(c) not in df.columns]
    if missing:
        raise ImportError_(f"{kind} export is missing required columns: {', '.join(missing)}")
    return df, header


def col(df: pd.DataFrame, name: str, default: Any = None) -> pd.Series:
    n = norm(name)
    if n in df.columns:
        return df[n]
    return pd.Series([default] * len(df), index=df.index, dtype=object)


def to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        s = s.astype(str).str.replace(",", "", regex=False).replace({"None": None, "nan": None, "": None})
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)


def to_str(s: pd.Series) -> pd.Series:
    out = s.astype(object).where(s.notna(), "")
    return out.map(lambda v: str(v).strip())


def make_date_parser(samples: Iterable[Any]) -> Callable[[Any], date | None]:
    """Decide dd-mm vs mm-dd for 'a-b-yyyy' strings from the data (defaults to dd-mm, the export's format)."""
    dm = md = 0
    for s in samples:
        if not isinstance(s, str):
            continue
        m = re.match(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
        if m:
            dm += int(m.group(1)) > 12
            md += int(m.group(2)) > 12
    day_first = dm >= md

    def parse(v: Any) -> date | None:
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            return None
        if isinstance(v, pd.Timestamp):
            return None if pd.isna(v) else v.date()
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        if isinstance(v, (int, float, np.integer, np.floating)):
            return (EXCEL_EPOCH + timedelta(days=float(v))).date()
        s = str(v).strip()
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
        if m:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            d, mo = (a, b) if day_first else (b, a)
            try:
                return date(int(m.group(3)), mo, d)
            except ValueError:
                return None
        try:
            return pd.Timestamp(s).date()
        except (ValueError, TypeError):
            return None

    return parse


def _dates(s: pd.Series) -> pd.Series:
    parse = make_date_parser(s.head(3000).tolist())
    return s.map(parse)


def dates_from_name(name: str) -> tuple[date, date] | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", name or "")
    if not m:
        return None
    return date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))


# ---------------------------------------------------------------- parsers ---

def parse_sales(raw: pd.DataFrame, file_name: str) -> ParsedFile:
    df, header = _frame(raw, "sales")
    pos = to_str(col(df, "Pos Profile"))
    inv = to_str(col(df, "Invoice Number"))
    code = to_str(col(df, "Item Code"))
    rep: dict[str, Any] = {"rows_read": len(df)}

    summary = pos.str.fullmatch(r"(?i)(grand )?total") | ((inv == "") & (code == ""))
    rep["summary_rows_skipped"] = int(summary.sum())

    wh = to_str(col(df, "Warehouse"))
    # Fill a missing warehouse from the POS profile it was sold through.
    pos_wh = {p: w for p, w in zip(pos, wh) if p and w}
    fill = (wh == "") & pos.map(lambda p: p in pos_wh)
    wh = wh.where(~fill, pos.map(pos_wh))
    rep["warehouse_filled_from_pos"] = int(fill.sum())

    out = pd.DataFrame(
        {
            "invoice_no": inv,
            "posting_date": _dates(col(df, "Posting Date")),
            "warehouse": wh.replace("", "UNASSIGNED"),
            "item_code": code,
            "item_name": to_str(col(df, "Item Name")),
            "item_group": to_str(col(df, "Item Group")),
            "parent_group": to_str(col(df, "Parent Item Group")),
            "qty": to_num(col(df, "Qty")),
            "net_amount": to_num(col(df, "Net Line Sales")),
            "cost": to_num(col(df, "Total Cost")),
            "customer_no": to_str(col(df, "Customer Number")),
            "customer_name": to_str(col(df, "Customer Name")),
            "sales_person": to_str(col(df, "Sales Person")),
            "transaction_type": to_str(col(df, "Transaction Type")).replace("", "UNKNOWN"),
        }
    )
    out = out[~summary]
    bad_date = out["posting_date"].isna()
    rep["bad_date_rows"] = int(bad_date.sum())
    out = out[~bad_date & (out["item_code"] != "")]
    rep["return_lines"] = int((out["qty"] < 0).sum())
    rep["zero_cost_lines"] = int(((out["cost"] == 0) & (out["qty"] > 0)).sum())
    below = (out["qty"] > 0) & (out["net_amount"] < out["cost"])
    rep["below_cost_lines"] = int(below.sum())
    rep["below_cost_loss"] = round(float((out.loc[below, "cost"] - out.loc[below, "net_amount"]).sum()), 2)
    rep["unassigned_warehouse_lines"] = int((out["warehouse"] == "UNASSIGNED").sum())
    rep["rows_loaded"] = len(out)
    return _finish("sales", file_name, out.reset_index(drop=True), rep, "posting_date")


def parse_stock(raw: pd.DataFrame, file_name: str) -> ParsedFile:
    df, _ = _frame(raw, "stock")
    out = pd.DataFrame(
        {
            "item_code": to_str(col(df, "Item")),
            "item_name": to_str(col(df, "Item Name")),
            "item_group": to_str(col(df, "Item Group")),
            "uom": to_str(col(df, "Stock UOM")),
            "warehouse": to_str(col(df, "Warehouse")),
            "qty": to_num(col(df, "Balance Qty")),
            "value": to_num(col(df, "Balance Value")),
            "valuation_rate": to_num(col(df, "Valuation Rate")),
            "reserved": to_num(col(df, "Reserved Stock")),
        }
    )
    rep: dict[str, Any] = {"rows_read": len(out)}
    blank = (out["item_code"] == "") | (out["warehouse"] == "")
    rep["summary_rows_skipped"] = int(blank.sum())
    out = out[~blank]
    no_rate = (out["valuation_rate"] == 0) & (out["qty"] != 0)
    out.loc[no_rate, "valuation_rate"] = (out.loc[no_rate, "value"] / out.loc[no_rate, "qty"]).fillna(0.0)
    rep["negative_stock_lines"] = int((out["qty"] < 0).sum())
    rep["zero_valuation_lines"] = int(((out["valuation_rate"] == 0) & (out["qty"] > 0)).sum())
    rep["rows_loaded"] = len(out)
    pf = ParsedFile("stock", file_name, out.reset_index(drop=True), rep)
    rng = dates_from_name(file_name)
    pf.date_to = rng[1] if rng else date.today()
    pf.date_from = pf.date_to
    return pf


def parse_purchase(raw: pd.DataFrame, file_name: str, usd_to_sar: float = 3.75) -> ParsedFile:
    df, _ = _frame(raw, "purchase")
    qty = to_num(col(df, "PO Qty"))
    received = to_num(col(df, "Received Qty"))
    cancelled = to_num(col(df, "Cancelled Qty"))
    has_remaining = norm("Remaining Qty") in df.columns
    remaining = to_num(col(df, "Remaining Qty")) if has_remaining else (qty - received - cancelled)
    status = to_str(col(df, "Status"))
    currency = to_str(col(df, "Currency")).replace("", "SAR").str.upper()
    price = to_num(col(df, "PO Price"))
    price_sar = np.where(currency == "USD", price * usd_to_sar, price)
    supplier = to_str(col(df, "Supplier"))
    supplier_name = to_str(col(df, "Supplier Name"))
    supplier = supplier.where(supplier != "", supplier_name).replace("", "UNKNOWN")

    out = pd.DataFrame(
        {
            "po_no": to_str(col(df, "PO No.")),
            "supplier_code": supplier,
            "supplier_name": supplier_name.where(supplier_name != "", supplier),
            "po_date": _dates(col(df, "PO Date")),
            "item_code": to_str(col(df, "Item")),
            "item_name": to_str(col(df, "Item Name")),
            "warehouse": to_str(col(df, "Warehouse")),
            "qty": qty,
            "received_qty": received,
            "cancelled_qty": cancelled,
            "remaining_qty": remaining.clip(lower=0),
            "status": status,
            "currency": currency,
            "price_sar": price_sar,
            "last_receipt": _dates(col(df, "Last Receipt")),
        }
    )
    rep: dict[str, Any] = {"rows_read": len(out)}
    closed = out["status"].str.lower().isin(CLOSED_PO_STATUSES)
    rep["closed_lines_with_remaining"] = int((closed & (out["remaining_qty"] > 0)).sum())
    out.loc[closed, "remaining_qty"] = 0.0
    bad = out["po_date"].isna() | (out["po_no"] == "") | (out["item_code"] == "")
    rep["bad_rows_skipped"] = int(bad.sum())
    out = out[~bad]
    rep["usd_lines"] = int((out["currency"] == "USD").sum())
    rep["rows_loaded"] = len(out)
    return _finish("purchase", file_name, out.reset_index(drop=True), rep, "po_date")


def parse_stock_entry(raw: pd.DataFrame, file_name: str) -> ParsedFile:
    df, _ = _frame(raw, "transfer")
    entry = to_str(col(df, "ID"))
    docstatus = to_num(col(df, "docstatus"))
    etype = to_str(col(df, "Stock Entry Type")).replace("", "UNKNOWN")
    src = to_str(col(df, "Source Warehouse (Stock Entry Detail)"))
    src = src.where(src != "", to_str(col(df, "Default Source Warehouse")))
    tgt = to_str(col(df, "Target Warehouse (Stock Entry Detail)"))
    tgt = tgt.where(tgt != "", to_str(col(df, "Default Target Warehouse")))
    final = to_str(col(df, "Final Target Warehouse"))
    qty = to_num(col(df, "Qty (Stock Entry Detail)"))
    rate = to_num(col(df, "Valuation Rate (Stock Entry Detail)"))
    per_transferred = to_num(col(df, "Per Transferred"))
    day = _dates(col(df, "Posting Date"))

    rep: dict[str, Any] = {"rows_read": len(df)}
    keep = entry != ""
    rep["cancelled_rows"] = int((keep & (docstatus == 2)).sum())
    drafts = keep & (docstatus == 0)
    rep["draft_rows"] = int(drafts.sum())
    rep["draft_entries"] = int(entry[drafts].nunique())
    keep &= docstatus == 1
    rep["bad_date_rows"] = int((keep & day.isna()).sum())
    keep &= day.notna()
    other = keep & (etype != "Material Transfer")
    rep["other_entry_types"] = etype[other].value_counts().to_dict()
    keep &= etype == "Material Transfer"
    arrivals = keep & src.map(is_transit)
    rep["arrival_legs_skipped"] = int(arrivals.sum())
    keep &= ~arrivals

    via = tgt.map(is_transit)
    dest = tgt.where(~via | (final == ""), final)
    out = pd.DataFrame(
        {
            "entry_id": entry,
            "posting_date": day,
            "item_code": to_str(col(df, "Item Code (Stock Entry Detail)")),
            "source": src.replace("", "UNKNOWN"),
            "destination": dest.replace("", "UNKNOWN"),
            "qty": qty,
            "value": (qty * rate).round(2),
            "in_transit": via & (per_transferred < 100),
        }
    )[keep]
    rep["in_transit_lines"] = int(out["in_transit"].sum())
    rep["rows_loaded"] = len(out)
    return _finish("transfer", file_name, out.reset_index(drop=True), rep, "posting_date")


def _finish(kind: str, file_name: str, out: pd.DataFrame, rep: dict, date_col: str) -> ParsedFile:
    pf = ParsedFile(kind, file_name, out, rep)
    if len(out):
        pf.date_from = min(out[date_col])
        pf.date_to = max(out[date_col])
    rng = dates_from_name(file_name)
    if rng:  # the export's own date range wins: it also covers days with no rows
        pf.date_from, pf.date_to = rng
    return pf


def parse_file(path: str | Path, file_name: str | None = None, usd_to_sar: float = 3.75) -> ParsedFile:
    name = file_name or Path(path).name
    raw = read_rows(path)
    kind = detect(raw)
    if kind == "sales":
        return parse_sales(raw, name)
    if kind == "stock":
        return parse_stock(raw, name)
    if kind == "purchase":
        return parse_purchase(raw, name, usd_to_sar)
    if kind == "transfer":
        return parse_stock_entry(raw, name)
    raise ImportError_(
        f'"{name}" is not one of the supported ERP exports (Sales Detailed, Stock Balance, '
        "Item-Wise Purchase History, Stock Entry): header row not recognised."
    )
