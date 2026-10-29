"""Human approval of suggested purchases and transfers.

Only a named person can approve or reject. The system and the AI agent can
create suggestions (status "pending") but there is deliberately no code path
that lets either of them change a suggestion's status.
"""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Item, Recommendation, Supplier, Warehouse

DECISIONS = {"approve": "approved", "reject": "rejected"}


class ApprovalError(ValueError):
    pass


def decide(session: Session, rec_id: int, decision: str, decided_by: str, note: str | None = None,
           qty: float | None = None) -> Recommendation:
    if decision not in DECISIONS:
        raise ApprovalError("decision must be 'approve' or 'reject'")
    who = (decided_by or "").strip()
    if not who or who.lower() in {"system", "agent", "claude", "ai"}:
        raise ApprovalError("a person's name is required to approve or reject")
    rec = session.get(Recommendation, rec_id)
    if rec is None:
        raise ApprovalError(f"suggestion {rec_id} not found")
    if rec.status != "pending":
        raise ApprovalError(f"suggestion {rec_id} is already {rec.status}")
    if qty is not None:
        if qty <= 0:
            raise ApprovalError("quantity must be positive; reject the suggestion instead")
        if rec.qty:
            rec.value = round(rec.value * qty / rec.qty, 2)
        rec.qty = qty
    rec.status = DECISIONS[decision]
    rec.decided_by = who
    rec.decided_at = datetime.now()
    rec.decision_note = note
    session.flush()
    return rec


def approved_export(session: Session, since: datetime | None = None) -> bytes:
    """Approved suggestions as an Excel workbook ready to key into the ERP:
    one sheet of purchase lines grouped by supplier, one of transfers by route."""
    q = select(Recommendation).where(Recommendation.status == "approved")
    if since:
        q = q.where(Recommendation.decided_at >= since)
    recs = list(session.scalars(q))
    names = dict(session.execute(select(Item.code, Item.name)).all())
    sup = dict(session.execute(select(Supplier.code, Supplier.name)).all())
    label = dict(session.execute(select(Warehouse.name, Warehouse.label)).all())
    buys = pd.DataFrame([{
        "Supplier": r.supplier_code or "(choose supplier)", "Supplier Name": sup.get(r.supplier_code, ""),
        "Item": r.item_code, "Item Name": names.get(r.item_code, ""), "Qty": r.qty, "Est. Value SAR": r.value,
        "Approved By": r.decided_by, "Approved At": r.decided_at, "Note": r.decision_note or "", "Suggestion #": r.id,
    } for r in recs if r.kind == "purchase"])
    moves = pd.DataFrame([{
        "From": r.from_warehouse, "From Branch": label.get(r.from_warehouse, ""), "To": r.to_warehouse,
        "To Branch": label.get(r.to_warehouse, ""), "Item": r.item_code, "Item Name": names.get(r.item_code, ""),
        "Qty": r.qty, "Est. Value SAR": r.value, "Approved By": r.decided_by, "Approved At": r.decided_at,
        "Suggestion #": r.id,
    } for r in recs if r.kind == "transfer"])
    if len(buys):
        buys = buys.sort_values(["Supplier", "Item"])
    if len(moves):
        moves = moves.sort_values(["From", "To", "Item"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        (buys if len(buys) else pd.DataFrame(columns=["Supplier", "Item", "Qty"])).to_excel(xw, sheet_name="Purchase orders", index=False)
        (moves if len(moves) else pd.DataFrame(columns=["From", "To", "Item", "Qty"])).to_excel(xw, sheet_name="Transfers", index=False)
    return buf.getvalue()
