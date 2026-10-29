# Inventory AI

AI-assisted inventory control for a spare-parts warehouse network (central CPD store plus ~20 branches,
built for 200k SKUs). It reads the ERP's daily Excel exports, forecasts demand for every item in every
warehouse, suggests transfers and purchases, raises alerts, and includes a Claude assistant you can ask
questions in plain language.

**Nothing is ordered or moved automatically.** Every purchase and transfer is a *suggestion* that a named
person approves or rejects in the dashboard. Approved lines download as an Excel sheet to key into the ERP.

```
ERP exports (.xlsx/.csv) ──► import ──► PostgreSQL ──► nightly run ──► suggestions + alerts ──► dashboard / API
                                                         │                      ▲
                                                         ▼                      │ approve / reject (people only)
                                              forecasts per item×warehouse      │
                                                                         Claude assistant (reads data, drafts only)
```

## What it does

| Part | How |
|---|---|
| **Import** | Recognises the four ERP exports by their header row: *Spare Part Item Sales Detailed*, *Stock Balance*, *Item-Wise Purchase History*, *Stock Entry*. Skips total rows, fills a missing warehouse from the POS profile, converts USD prices, ignores draft/cancelled stock entries, and reports every skipped row. Re-importing a year-to-date file replaces that date range, never duplicates. |
| **Forecast** | Per item and warehouse, from the last 182 days. Classifies each series (smooth / erratic / intermittent / lumpy). Steady sellers use exponential smoothing; slow and sporadic parts (most spare parts) use **TSB**, which also fades the forecast when a part stops selling. Runs as one matrix per warehouse, so 4M series take minutes, not hours. |
| **ABC / XYZ** | A = items making the first 80% of sales value, B = next 15%, C = the rest. Sets the service level: 98% / 95% / 90%. |
| **Transfers first** | A branch below its reorder point is topped up from CPD surplus, then from branches holding more than 60 days of their own demand or stock they don't sell. |
| **Purchases** | Company level per item: stock + in transit + open POs vs reorder point for the supplier's **learned** lead time (median PO date → last receipt). Orders up to 30 days beyond that. New items get the brand's usual supplier, marked as a guess. |
| **Alerts** | Demand spikes (and suggestions they drive are marked), sudden drops, dead stock (180 days), **unexplained stock loss** between two stock imports, negative stock, sales below cost, dead items still on order. |
| **Assistant** | Claude with read-only tools over the data (summary, item status, branch summary, stock-out risk, suggestions, alerts) plus *draft* purchase/transfer. Drafts land in Approvals as pending; it has no tool to approve. |

Formulas (see `inventory_ai/replenishment.py`):

```
safety stock  = z(service level) × daily std × √(lead time + review period)
reorder point = daily forecast × (lead time + review period) + safety stock
order-up-to   = reorder point + daily forecast × order cycle (30 days)
```

## Quick start (Docker)

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY for the assistant; set APP_TOKEN to require a password
docker compose up -d --build  # PostgreSQL + app on http://localhost:8000 + nightly run at 02:00
```

Then open the dashboard, go to **Import & run**, upload the four exports and press **Run now**.

## Quick start (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[postgres,dev]"
export DATABASE_URL=sqlite:///./inventory.db          # or a PostgreSQL URL

python scripts/generate_sample_data.py --out sample_data   # synthetic exports in the ERP layout (optional)
python -m inventory_ai import sample_data/*.xlsx
python -m inventory_ai run
python -m inventory_ai serve                                # http://127.0.0.1:8000
python -m inventory_ai ask "Which A items will run out in Jeddah this week?"
```

## Daily routine

1. Export from the ERP: Sales Detailed (year to date), Stock Balance (today), Item-Wise Purchase History
   (year to date), Stock Entry (year to date). Keep the date range in the file name
   (`..._2026-01-01_2026-10-28.xlsx`) so re-imports replace exactly that range.
2. Upload them in **Import & run** (or `python -m inventory_ai import ...`).
3. The nightly run (or **Run now**) refreshes forecasts, suggestions and alerts. Yesterday's undecided
   system suggestions are marked *superseded*; decisions and assistant drafts are kept.
4. Buyers and branch managers review **Approvals**, adjust quantities, approve or reject with their name.
5. **Download approved (Excel)**: purchase lines grouped by supplier, transfers grouped by route.

Importing a stock balance every day also lets the loss check compare consecutive days.

## Scale

Measured on 4 CPU cores with PostgreSQL 16, synthetic data at 200,000 SKUs × 20 branches + CPD (see the
commit message / PR for exact timings). Forecasting works one warehouse at a time, so memory stays flat;
PostgreSQL loads use `COPY`. Excel stops at 1,048,576 rows per sheet, so very large exports should be CSV.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/summary` | KPIs |
| POST | `/api/import` | multipart `files` |
| POST | `/api/run` | nightly run now |
| GET | `/api/suggestions?kind=&status=&warehouse=&supplier=&item=` | |
| POST | `/api/suggestions/{id}/decision` | `{decision: approve\|reject, decided_by, note?, qty?}` |
| POST | `/api/suggestions/bulk-decision` | `{ids, decision, decided_by}` |
| GET | `/api/approved.xlsx` | approved lines for the ERP |
| GET | `/api/alerts`, POST `/api/alerts/{id}/acknowledge` | |
| GET | `/api/items/search?q=`, `/api/items/{code}` | |
| GET | `/api/warehouses`, `/api/warehouses/{name}/summary`, `/api/stockout-risk` | |
| POST | `/api/agent` | `{question, conversation_id?}` |

Interactive docs at `/docs`.

## Configuration

All settings are environment variables; see `.env.example`. The assistant uses `claude-opus-5-5` with
adaptive thinking and server-side refusal fallbacks; change `AGENT_MODEL` / `AGENT_EFFORT` to trade cost
for depth.

## Tests

```bash
pytest
```

Covers the import rules for each export, the forecasting methods, transfer-before-purchase logic, the
approval rules (system/agent can't approve), a full import → run → approve → export flow, and the
assistant's tool loop with a scripted model.

## Not yet included

- Login per user (today: optional shared `APP_TOKEN`, approver types their name).
- Writing approved orders back into the ERP automatically (today: Excel export to key in).
- Seasonality (Ramadan, summer) and promotions in the forecast; minimum order quantities and pack sizes.
- Distances between branches (transfers are not route-optimised).
