"""Generate sample exports, import them through the API, run the nightly job, approve, export."""
import io
from argparse import Namespace
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import generate_sample_data
from inventory_ai import api as api_module
from inventory_ai import db
from inventory_ai.agent import TOOLS, InventoryAgent
from inventory_ai.models import Recommendation


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    out = tmp_path_factory.mktemp("sample")
    files = generate_sample_data.make(Namespace(skus=300, branches=4, days=150, end="2026-10-28", seed=3, csv=True, demand_scale=1.0, out=str(out)))
    return list(files.values())


@pytest.fixture()
def client(database):
    with TestClient(api_module.app) as c:
        yield c


def upload(client, paths):
    files = [("files", (p.name, p.read_bytes(), "text/csv")) for p in paths]
    return client.post("/api/import", files=files).json()


def test_full_flow(client, sample):
    res = upload(client, sample)
    assert not res["errors"]
    assert {x["kind"] for x in res["loaded"]} == {"sales", "stock", "purchase", "transfer"}

    run = client.post("/api/run").json()
    assert run["series_forecast"] > 0 and run["purchase_suggestions"] > 0
    # The generator plants these problems; the alerts must find them.
    assert {"dead_stock", "unexplained_loss", "negative_stock", "dead_on_order", "demand_spike"} <= set(run["alerts"])
    spikes = client.get("/api/alerts", params={"kind": "demand_spike"}).json()["rows"]
    assert spikes[0]["item_code"] == "SP-000001"

    summary = client.get("/api/summary").json()
    assert summary["pending"]["purchase"]["count"] == run["purchase_suggestions"]

    rows = client.get("/api/suggestions", params={"kind": "purchase", "limit": 3}).json()["rows"]
    rid = rows[0]["id"]
    bad = client.post(f"/api/suggestions/{rid}/decision", json={"decision": "approve", "decided_by": "agent"})
    assert bad.status_code == 400
    ok = client.post(f"/api/suggestions/{rid}/decision", json={"decision": "approve", "decided_by": "Ahmed Ali"})
    assert ok.json()["status"] == "approved"
    bulk = client.post("/api/suggestions/bulk-decision",
                       json={"ids": [rows[1]["id"], rid], "decision": "reject", "decided_by": "Ahmed Ali"}).json()
    assert bulk["done"] == [rows[1]["id"]] and bulk["failed"][0]["id"] == rid

    x = client.get("/api/approved.xlsx")
    sheet = pd.read_excel(io.BytesIO(x.content), sheet_name="Purchase orders")
    assert list(sheet["Suggestion #"]) == [rid]

    item = client.get(f"/api/items/{rows[0]['item_code']}").json()
    assert item["warehouses"] and "reorder_point" in item["warehouses"][0]

    # Re-importing the same files replaces rather than duplicates.
    before = client.get("/api/summary").json()["items"]
    upload(client, sample)
    assert client.get("/api/summary").json()["items"] == before
    # A second nightly run supersedes yesterday's open system suggestions but keeps decisions.
    client.post("/api/run")
    with db.session_scope() as s:
        assert s.get(Recommendation, rid).status == "approved"
        assert s.query(Recommendation).filter_by(status="superseded").count() > 0


class FakeClient:
    """Plays back scripted model turns and records the requests."""

    def __init__(self, turns):
        self.turns, self.requests = list(turns), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self.turns.pop(0)


def turn(stop, *blocks):
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


def tool_use(id_, name, args):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


def text(t):
    return SimpleNamespace(type="text", text=t)


def test_agent_drafts_only_pending_suggestions(client, sample):
    upload(client, sample)
    client.post("/api/run")
    with db.session_scope() as s:
        fake = FakeClient([
            turn("tool_use", tool_use("t1", "get_item_status", {"item_code": "SP-000001"}),
                 tool_use("t2", "search_items", {"query": "SP-00000", "limit": 3})),
            turn("tool_use", tool_use("t3", "draft_purchase", {"item_code": "SP-000001", "qty": 12, "reason": "spike"}),
                 tool_use("t4", "get_item_status", {"item_code": "NOPE"})),
            turn("end_turn", text("Drafted a purchase of 12 for approval.")),
        ])
        reply = InventoryAgent(s, client=fake).ask("Order more SP-000001")
        assert reply.text.startswith("Drafted") and len(reply.drafted) == 1
        rec = s.get(Recommendation, reply.drafted[0])
        assert (rec.status, rec.created_by, rec.qty) == ("pending", "agent", 12)
        # Tool results go back in one user message, errors flagged, and every request carries the safety settings.
        # messages: question, turn 1, results 1, turn 2, results 2, final turn
        results = fake.requests[2]["messages"][4]["content"]
        assert [r["is_error"] for r in results] == [False, True]
        assert all(r["fallbacks"] == "default" and r["thinking"] == {"type": "adaptive"} for r in fake.requests)


def test_agent_has_no_way_to_approve():
    names = {t["name"] for t in TOOLS}
    assert not any(w in n for n in names for w in ("approve", "reject", "decide", "place", "submit"))


def test_spike_driven_suggestions_carry_a_warning(client, sample):
    upload(client, sample)
    client.post("/api/run")
    recs = client.get("/api/suggestions", params={"item": "SP-000001", "status": "pending"}).json()["rows"]
    assert recs and all("demand spike" in r["reason"] for r in recs)
