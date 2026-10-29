"""Claude-powered inventory assistant.

The agent answers questions by calling read-only tools over the inventory
database, and can *draft* purchases or transfers. Drafts are saved as
pending suggestions; it has no tool to approve, reject or place anything, so
every order still needs a person's approval in the dashboard.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
from sqlalchemy.orm import Session

from . import queries
from .config import Settings, get_settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the inventory assistant for an automotive spare-parts business in Saudi Arabia with a central \
warehouse (CPD) and about 20 branch warehouses. Amounts are in SAR, excluding VAT.

You work from the company's own data through the tools provided: stock balances, sales history, purchase orders, \
transfers, nightly demand forecasts, replenishment suggestions and alerts. Use the tools to look things up rather than \
guessing, and say which data your answer is based on (for example the forecast run date) when it matters.

How the numbers are made:
- Forecasts are per item and warehouse, in units per day, from the last ~6 months of sales. Slow and sporadic parts \
use the TSB method; steady sellers use exponential smoothing.
- Reorder point = forecast x (lead time + review period) + safety stock. Safety stock depends on the item's ABC class \
(A = top 80% of sales value gets the highest service level).
- Each branch stocks only its own range (about 2,500 parts it sells regularly: on at least a few days in the last 6 months). Parts outside a branch's range are supplied from CPD when ordered, and branch stock of them is surplus.
- Transfers are suggested before purchases: stock is moved from CPD or from branches holding far more than they sell.

Approval rule: you can draft a purchase or transfer, which creates a *pending* suggestion. You cannot approve, \
reject or place orders, and nothing you draft happens until a person approves it in the dashboard. Never say that \
something has been ordered, moved or approved. Only draft when the user asks you to, and give the reason in the draft.

Keep answers short and practical: lead with the answer, then the key numbers. Use a small table when comparing \
several items or branches."""


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


S = {"type": "string"}
N = {"type": "number"}
I = {"type": "integer", "minimum": 1, "maximum": 200}

TOOLS: list[dict[str, Any]] = [
    {"name": "get_summary", "description": "Company-wide overview: stock value, pending suggestions, open alerts, "
     "branch lines out of stock, and when each ERP export was last imported.", "input_schema": _obj({}, [])},
    {"name": "list_warehouses", "description": "All warehouses with their category (Central, Branch, In transit, "
     "Slow moving, Hidden, Scrap).", "input_schema": _obj({}, [])},
    {"name": "search_items", "description": "Find items by part code or words in the description.",
     "input_schema": _obj({"query": S, "limit": I}, ["query"])},
    {"name": "get_item_status", "description": "Everything about one item: stock, forecast, reorder point and days of "
     "cover in every warehouse, open purchase orders, pending suggestions and alerts.",
     "input_schema": _obj({"item_code": S}, ["item_code"])},
    {"name": "warehouse_summary", "description": "Stock value, demand, pending transfers in/out, alerts and stock-out "
     "risk for one warehouse. Accepts a full name or a branch name such as 'Jeddah Safa'.",
     "input_schema": _obj({"warehouse": S}, ["warehouse"])},
    {"name": "stockout_risk", "description": "Selling item lines that run out within N days at the forecast rate "
     "(stock + in transit), most important first. Optionally for one warehouse.",
     "input_schema": _obj({"warehouse": S, "within_days": N, "limit": I}, [])},
    {"name": "list_suggestions", "description": "Replenishment suggestions (purchases and transfers) with their reasons. "
     "status: pending (default), approved, rejected, superseded or all.",
     "input_schema": _obj({"kind": {"type": "string", "enum": ["purchase", "transfer"]},
                           "status": {"type": "string", "enum": ["pending", "approved", "rejected", "superseded", "all"]},
                           "warehouse": S, "item_code": S, "limit": I}, [])},
    {"name": "list_alerts", "description": "Open alerts: demand_spike, demand_drop, dead_stock, unexplained_loss, "
     "negative_stock, below_cost_sales, dead_on_order.",
     "input_schema": _obj({"kind": S, "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                           "warehouse": S, "item_code": S, "limit": I}, [])},
    {"name": "draft_transfer", "description": "Draft a stock transfer between warehouses. Creates a PENDING suggestion "
     "that a person must approve; it does not move stock.",
     "input_schema": _obj({"item_code": S, "from_warehouse": S, "to_warehouse": S, "qty": N, "reason": S},
                          ["item_code", "from_warehouse", "to_warehouse", "qty", "reason"])},
    {"name": "draft_purchase", "description": "Draft a purchase. Creates a PENDING suggestion that a person must approve; "
     "it does not place an order. Supplier defaults to the item's last supplier.",
     "input_schema": _obj({"item_code": S, "qty": N, "reason": S, "supplier_code": S}, ["item_code", "qty", "reason"])},
]


class AgentNotConfigured(RuntimeError):
    """No Claude credentials (ANTHROPIC_API_KEY or an `ant auth login` profile) on this server."""


@dataclass
class AgentReply:
    text: str
    messages: list[Any]
    drafted: list[int] = field(default_factory=list)  # ids of suggestions drafted in this turn


class InventoryAgent:
    def __init__(self, session: Session, client: anthropic.Anthropic | None = None, cfg: Settings | None = None):
        self.session = session
        self.cfg = cfg or get_settings()
        try:
            self.client = client or anthropic.Anthropic()
        except anthropic.CredentialsError as e:  # a configured profile that can't be loaded
            raise AgentNotConfigured(str(e)) from e
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "get_summary": lambda a: queries.summary(session),
            "list_warehouses": lambda a: queries.list_warehouses(session),
            "search_items": lambda a: queries.search_items(session, a["query"], a.get("limit", 20)),
            "get_item_status": self._item_status,
            "warehouse_summary": self._warehouse_summary,
            "stockout_risk": lambda a: queries.stockout_risk(
                session, self._wh(a.get("warehouse")), a.get("within_days", 14), a.get("limit", 30)),
            "list_suggestions": lambda a: queries.list_recommendations(
                session, a.get("kind"), a.get("status", "pending"), self._wh(a.get("warehouse")), None,
                a.get("item_code"), a.get("limit", 20)),
            "list_alerts": lambda a: queries.list_alerts(
                session, a.get("kind"), a.get("severity"), self._wh(a.get("warehouse")), a.get("item_code"), a.get("limit", 20)),
            "draft_transfer": self._draft_transfer,
            "draft_purchase": self._draft_purchase,
        }
        self._drafted: list[int] = []

    # ---- tool handlers -----------------------------------------------------
    def _wh(self, text: str | None) -> str | None:
        if not text:
            return None
        name = queries.resolve_warehouse(self.session, text)
        if name is None:
            raise ValueError(f"Unknown or ambiguous warehouse '{text}'. Call list_warehouses for exact names.")
        return name

    def _item_status(self, a: dict[str, Any]) -> Any:
        res = queries.item_status(self.session, a["item_code"], self.cfg)
        if res is None:
            raise ValueError(f"No item with code {a['item_code']}. Try search_items.")
        return res

    def _warehouse_summary(self, a: dict[str, Any]) -> Any:
        return queries.warehouse_summary(self.session, self._wh(a["warehouse"]))

    def _draft_transfer(self, a: dict[str, Any]) -> Any:
        rec = queries.draft_transfer(self.session, a["item_code"], a["from_warehouse"], a["to_warehouse"],
                                     float(a["qty"]), a["reason"])
        self._drafted.append(rec.id)
        return {"suggestion_id": rec.id, "status": "pending", "value": rec.value,
                "note": "Saved as a pending suggestion. A person must approve it in the dashboard."}

    def _draft_purchase(self, a: dict[str, Any]) -> Any:
        rec = queries.draft_purchase(self.session, a["item_code"], float(a["qty"]), a["reason"], a.get("supplier_code"))
        self._drafted.append(rec.id)
        return {"suggestion_id": rec.id, "status": "pending", "supplier": rec.supplier_code, "value": rec.value,
                "note": "Saved as a pending suggestion. A person must approve it in the dashboard."}

    def run_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        handler = self.handlers.get(name)
        if handler is None:
            return f"Unknown tool {name}", True
        try:
            out = json.dumps(handler(args or {}), default=str)
        except (ValueError, KeyError, TypeError) as e:
            return f"Error: {e}", True
        return (out[:60_000] + "... (truncated; narrow the query)") if len(out) > 60_000 else out, False

    # ---- conversation loop ------------------------------------------------
    def ask(self, question: str, history: list[Any] | None = None) -> AgentReply:
        messages: list[Any] = list(history or []) + [{"role": "user", "content": question}]
        self._drafted = []
        for _ in range(self.cfg.agent_max_turns):
            try:
                resp = self.client.beta.messages.create(
                    model=self.cfg.agent_model,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.cfg.agent_effort},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except anthropic.CredentialsError as e:  # a configured profile that can't be loaded
                raise AgentNotConfigured(str(e)) from e
            except TypeError as e:  # the SDK raises this when it finds no credentials at all
                if "authentication" in str(e):
                    raise AgentNotConfigured(str(e)) from e
                raise
            # Keep the full content (including thinking blocks) so later turns can be sent back unchanged.
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "refusal":
                return AgentReply("I can't help with that request.", messages, self._drafted)
            if resp.stop_reason == "pause_turn":
                continue
            calls = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not calls:
                text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
                if resp.stop_reason == "max_tokens":
                    text += "\n\n(answer cut short)"
                return AgentReply(text, messages, self._drafted)
            results = []
            for c in calls:
                content, is_error = self.run_tool(c.name, c.input)
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": content, "is_error": is_error})
            messages.append({"role": "user", "content": results})
        return AgentReply("I stopped after too many steps. Please ask a narrower question.", messages, self._drafted)
