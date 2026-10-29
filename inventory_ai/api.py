"""HTTP API and dashboard.  Start with:  python -m inventory_ai serve"""
from __future__ import annotations

import os
import secrets
import tempfile
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import anthropic
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import approvals, db, queries
from .config import get_settings
from .importers import ImportError_, load, parse_file
from .models import Alert
from .pipeline import run_nightly

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Inventory AI", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def _token_guard(request: Request, call_next):
    """If APP_TOKEN is set, every /api call must send it (header X-Token)."""
    token = os.environ.get("APP_TOKEN")
    if token and request.url.path.startswith("/api/"):
        sent = request.headers.get("x-token") or request.query_params.get("token") or ""
        if not secrets.compare_digest(sent, token):
            return JSONResponse({"detail": "missing or wrong token"}, status_code=401)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/summary")
def get_summary(s: Session = Depends(db.get_session)) -> dict[str, Any]:
    return queries.summary(s)


# ------------------------------------------------------------------ import ---

@app.post("/api/import")
def import_files(files: list[UploadFile] = File(...), s: Session = Depends(db.get_session)) -> dict[str, Any]:
    cfg = get_settings()
    parsed, errors = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = Path(f.filename or "upload.xlsx").name
            path = Path(tmp) / name
            path.write_bytes(f.file.read())
            try:
                parsed.append(parse_file(path, name, cfg.usd_to_sar))
            except ImportError_ as e:
                errors.append({"file": name, "error": str(e)})
            except Exception as e:  # unreadable file
                errors.append({"file": name, "error": f"could not read file: {e}"})
    order = {"stock": 0, "purchase": 1, "transfer": 2, "sales": 3}
    loaded = []
    for pf in sorted(parsed, key=lambda p: (order[p.kind], str(p.date_to))):
        entry = load(s, pf)
        loaded.append({"file": pf.file_name, "kind": pf.kind, "rows": entry.rows_loaded, "from": str(pf.date_from),
                       "to": str(pf.date_to), "report": pf.report})
    return {"loaded": loaded, "errors": errors}


@app.post("/api/run")
def run(s: Session = Depends(db.get_session)) -> dict[str, Any]:
    return run_nightly(s)


# ------------------------------------------------------------ suggestions ---

@app.get("/api/suggestions")
def suggestions(kind: Literal["purchase", "transfer"] | None = None, status: str = "pending", warehouse: str | None = None,
                supplier: str | None = None, item: str | None = None, limit: int = 50, offset: int = 0,
                s: Session = Depends(db.get_session)) -> dict[str, Any]:
    return queries.list_recommendations(s, kind, status, warehouse, supplier, item, min(limit, 500), offset)


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    decided_by: str = Field(min_length=2, description="name of the person deciding")
    note: str | None = None
    qty: float | None = Field(default=None, description="approve a different quantity")


class BulkDecision(BaseModel):
    ids: list[int]
    decision: Literal["approve", "reject"]
    decided_by: str = Field(min_length=2)
    note: str | None = None


@app.post("/api/suggestions/{rec_id}/decision")
def decide(rec_id: int, body: Decision, s: Session = Depends(db.get_session)) -> dict[str, Any]:
    try:
        rec = approvals.decide(s, rec_id, body.decision, body.decided_by, body.note, body.qty)
    except approvals.ApprovalError as e:
        raise HTTPException(400, str(e))
    return {"id": rec.id, "status": rec.status, "qty": rec.qty, "value": rec.value, "decided_by": rec.decided_by}


@app.post("/api/suggestions/bulk-decision")
def bulk_decide(body: BulkDecision, s: Session = Depends(db.get_session)) -> dict[str, Any]:
    done, failed = [], []
    for rid in body.ids:
        try:
            approvals.decide(s, rid, body.decision, body.decided_by, body.note)
            done.append(rid)
        except approvals.ApprovalError as e:
            failed.append({"id": rid, "error": str(e)})
    return {"done": done, "failed": failed}


@app.get("/api/approved.xlsx")
def approved_xlsx(since: datetime | None = None, s: Session = Depends(db.get_session)) -> Response:
    data = approvals.approved_export(s, since)
    name = f"approved_{datetime.now():%Y-%m-%d_%H%M}.xlsx"
    return Response(data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ---------------------------------------------------------------- alerts ---

@app.get("/api/alerts")
def alerts(kind: str | None = None, severity: str | None = None, warehouse: str | None = None, item: str | None = None,
           limit: int = 50, offset: int = 0, s: Session = Depends(db.get_session)) -> dict[str, Any]:
    return queries.list_alerts(s, kind, severity, warehouse, item, min(limit, 500), offset)


@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge(alert_id: int, s: Session = Depends(db.get_session)) -> dict[str, Any]:
    a = s.get(Alert, alert_id)
    if a is None:
        raise HTTPException(404, "alert not found")
    a.acknowledged = True
    return {"id": a.id, "acknowledged": True}


# ----------------------------------------------------------- items / whs ---

@app.get("/api/items/search")
def item_search(q: str, limit: int = 20, s: Session = Depends(db.get_session)) -> list[dict]:
    return queries.search_items(s, q, min(limit, 100))


@app.get("/api/items/{code}")
def item(code: str, s: Session = Depends(db.get_session)) -> dict[str, Any]:
    res = queries.item_status(s, code)
    if res is None:
        raise HTTPException(404, "item not found")
    return res


@app.get("/api/warehouses")
def warehouses(s: Session = Depends(db.get_session)) -> list[dict]:
    return queries.list_warehouses(s)


@app.get("/api/warehouses/{name}/summary")
def warehouse_summary(name: str, s: Session = Depends(db.get_session)) -> dict[str, Any]:
    res = queries.warehouse_summary(s, queries.resolve_warehouse(s, name) or name)
    if res is None:
        raise HTTPException(404, "warehouse not found")
    return res


@app.get("/api/stockout-risk")
def stockout(warehouse: str | None = None, days: float = 14, limit: int = 50, s: Session = Depends(db.get_session)) -> list[dict]:
    return queries.stockout_risk(s, warehouse, days, min(limit, 500))


# ----------------------------------------------------------------- agent ---

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


_conversations: "OrderedDict[str, list[Any]]" = OrderedDict()
MAX_CONVERSATIONS = 200


@app.post("/api/agent")
def ask_agent(body: Question, s: Session = Depends(db.get_session)) -> dict[str, Any]:
    from .agent import InventoryAgent

    cid = body.conversation_id if body.conversation_id in _conversations else uuid.uuid4().hex
    try:
        reply = InventoryAgent(s).ask(body.question, _conversations.get(cid))
    except anthropic.AuthenticationError:
        raise HTTPException(503, "The AI assistant is not configured: set ANTHROPIC_API_KEY on the server.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "The AI assistant is busy. Try again in a minute.")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the AI service.")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"AI service error ({e.status_code}).")
    _conversations[cid] = reply.messages
    _conversations.move_to_end(cid)
    while len(_conversations) > MAX_CONVERSATIONS:
        _conversations.popitem(last=False)
    return {"answer": reply.text, "conversation_id": cid, "drafted_suggestions": reply.drafted}
