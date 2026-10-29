"""Command line: python -m inventory_ai <command>

  init                      create the database tables
  import FILE [FILE ...]    load ERP exports (any of the four, any order)
  run [--as-of YYYY-MM-DD]  nightly run: forecasts, suggestions, alerts
  serve [--port 8000]       start the API and dashboard
  ask "question"            ask the inventory agent from the terminal
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date

from . import db
from .config import get_settings


def _import(paths: list[str]) -> int:
    from .importers import ImportError_, load, parse_file

    cfg = get_settings()
    # Stock files are loaded oldest first so the newest one ends up as current stock.
    parsed, failed = [], 0
    for p in paths:
        try:
            parsed.append(parse_file(p, usd_to_sar=cfg.usd_to_sar))
        except ImportError_ as e:
            print(f"SKIPPED {p}: {e}", file=sys.stderr)
            failed += 1
    order = {"stock": 0, "purchase": 1, "transfer": 2, "sales": 3}
    parsed.sort(key=lambda pf: (order[pf.kind], pf.date_to or date.min))
    for pf in parsed:
        t0 = time.time()
        with db.session_scope() as s:
            entry = load(s, pf)
        print(f"{pf.kind:9s} {entry.rows_loaded:>9,} rows  {pf.date_from} .. {pf.date_to}  {time.time() - t0:6.1f}s  {pf.file_name}")
        print("          " + json.dumps(pf.report, default=str))
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="inventory_ai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p_imp = sub.add_parser("import")
    p_imp.add_argument("files", nargs="+")
    p_run = sub.add_parser("run")
    p_run.add_argument("--as-of", type=date.fromisoformat)
    p_srv = sub.add_parser("serve")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db.init_db()
    if args.cmd == "init":
        print("database ready")
        return 0
    if args.cmd == "import":
        return _import(args.files)
    if args.cmd == "run":
        from .pipeline import run_nightly

        with db.session_scope() as s:
            print(json.dumps(run_nightly(s, args.as_of), indent=2))
        return 0
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("inventory_ai.api:app", host=args.host, port=args.port)
        return 0
    if args.cmd == "ask":
        from .agent import InventoryAgent

        with db.session_scope() as s:
            print(InventoryAgent(s).ask(args.question).text)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
