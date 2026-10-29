"""Generate synthetic ERP exports in the same layout as the real ones.

Writes, into --out:
  Spare Part Item Sales Detailed_<from>_<to>.xlsx
  Stock Balance Spare Parts_<d>_<d>.xlsx        (today, plus one a week earlier)
  Item-Wise Purchase History_<from>_<to>.xlsx
  Stock Entry_<from>_<to>.xlsx

Use --csv for large runs (Excel stops at 1,048,576 rows). The data contains
deliberate problems for the alerts to find: a demand spike, dead stock,
stock that went missing between the two stock files, and dead items on order.

    python scripts/generate_sample_data.py --skus 3000 --out sample_data
    python scripts/generate_sample_data.py --skus 200000 --demand-scale 0.2 --csv --out sample_data_full
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CITIES = ["Riyadh Olaya", "Riyadh Malaz", "Riyadh Naseem", "Riyadh Suwaidi", "Jeddah Safa", "Jeddah Balad",
          "Jeddah Rawdah", "Makkah Aziziyah", "Madinah Quba", "Dammam Faisaliyah", "Khobar Thuqbah", "Jubail",
          "Hofuf", "Qassim Buraidah", "Hail", "Tabuk", "Abha", "Khamis Mushait", "Jazan", "Najran",
          "Taif", "Yanbu", "Al Kharj", "Hafr Al Batin"]
BRANDS = ["Toyota", "Hyundai", "Nissan", "MG", "Kia", "Chevrolet", "Ford", "Mitsubishi", "Isuzu", "Changan"]
GROUPS = ["Filters", "Brake Pads", "Brake Discs", "Spark Plugs", "Belts", "Suspension", "Bearings", "Gaskets",
          "Clutch", "Lighting", "Cooling", "Electrical", "Body Parts", "Engine Mounts", "Wipers"]


def branch_names(n: int) -> list[str]:
    return [f"P{i + 1:02d} - {CITIES[i % len(CITIES)]}{'' if i < len(CITIES) else f' {i // len(CITIES) + 1}'} - SP" for i in range(n)]


def make(args) -> dict[str, Path]:
    rng = np.random.default_rng(args.seed)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days - 1)
    days = pd.date_range(start, end, freq="D")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = args.skus
    branches = branch_names(args.branches)
    cpd = "CPD - SP"
    codes = np.array([f"SP-{i + 1:06d}" for i in range(n)])
    brand = rng.choice(BRANDS, n)
    group = rng.choice(GROUPS, n)
    names = np.array([f"{g[:-1] if g.endswith('s') else g} {b} #{i % 997}" for i, (g, b) in enumerate(zip(group, brand))])
    cost = np.round(rng.lognormal(3.3, 1.0, n), 2)
    price = np.round(cost * rng.uniform(1.25, 1.8, n), 2)
    # Heavy-tailed popularity: a few fast movers, a long tail of slow, intermittent parts.
    rate = rng.lognormal(-5.8, 1.5, n)
    rate[: max(1, n // 200)] *= 15
    rate *= args.demand_scale
    dead = rng.choice(n, max(3, n // 50), replace=False)  # never sell during the window
    rate[dead] = 0
    branch_factor = rng.uniform(0.3, 1.6, len(branches))
    wh_names = branches + [cpd]
    wh_factor = np.append(branch_factor, 0.4)  # CPD also sells to trade customers

    # ---- sales lines -------------------------------------------------------
    rows = []
    spike_item, spike_wh = 0, 0  # fastest item gets a spike in the last 7 days at the first branch
    for w, wname in enumerate(wh_names):
        lam = np.outer(rate * wh_factor[w], np.ones(len(days)))
        events = rng.poisson(lam)
        if w == spike_wh:
            events[spike_item, -7:] += 6
        it, dd = np.nonzero(events)
        sizes = events[it, dd] * rng.choice([1, 1, 1, 2, 4], len(it))
        if w == spike_wh:
            sizes[(it == spike_item) & (dd >= len(days) - 7)] *= 3
        rows.append(pd.DataFrame({"i": it, "d": dd, "qty": sizes.astype(float), "w": w}))
    s = pd.concat(rows, ignore_index=True)
    ret = rng.random(len(s)) < 0.02
    s.loc[ret, "qty"] = -s.loc[ret, "qty"].clip(upper=1)
    disc = np.where(rng.random(len(s)) < 0.15, rng.uniform(0.02, 0.25, len(s)), 0.0)
    below = rng.random(len(s)) < 0.003  # a few lines sold under cost
    disc[below] = 0.45
    list_rate = price[s["i"]]
    rate_net = np.round(list_rate * (1 - disc), 2)
    net = np.round(rate_net * s["qty"], 2)
    tcost = np.round(cost[s["i"]] * s["qty"], 2)
    cust = rng.integers(0, 400, len(s))
    walk = rng.random(len(s)) < 0.55
    wh_col = np.array(wh_names)[s["w"]]
    pos_names = np.array([f"POS {x.split(' - ')[1] if ' - ' in x else x}" for x in wh_names])
    day_str = np.array([d.strftime("%d-%m-%Y") for d in days])
    cust_no = np.array([f"CUST-{c:04d}" for c in range(400)])
    sales = pd.DataFrame({
        "Pos Profile": pos_names[s["w"]],
        "Warehouse": wh_col,
        "Invoice Number": "SINV-26-" + pd.Series(np.arange(len(s))).astype(str).str.zfill(7),
        "Posting Date": day_str[s["d"]],
        "Customer Number": np.where(walk, "CUST-0000", cust_no[cust]),
        "Customer Name": np.where(walk, "Cash Customer", "Workshop " + pd.Series(cust).astype(str)),
        "Sales Person": "Sales Person " + pd.Series(rng.integers(0, 60, len(s))).astype(str),
        "Item Code": codes[s["i"]], "Item Name": names[s["i"]],
        "Parent Item Group": brand[s["i"]], "Item Group": group[s["i"]],
        "Transaction Type": np.where(s["qty"] < 0, "Return", np.where(walk, "Cash", "Credit")),
        "Qty": s["qty"], "Price List Rate": list_rate, "Discount Amount": np.round(list_rate * disc * s["qty"], 2),
        "Rate": rate_net, "Net Line Sales": net, "Total Cost": tcost, "Total Profit": np.round(net - tcost, 2),
        "VAT": np.round(net * 0.15, 2), "Total Line": np.round(net * 1.15, 2),
    })
    sales = sales.iloc[np.argsort(s["d"].to_numpy(), kind="stable")]
    total = {c: "" for c in sales.columns}
    total.update({"Pos Profile": "Grand Total", "Qty": sales["Qty"].sum(), "Net Line Sales": round(sales["Net Line Sales"].sum(), 2)})
    sales = pd.concat([sales, pd.DataFrame([total])], ignore_index=True)

    # ---- stock today and a week ago ---------------------------------------
    stock_rows = []
    for w, wname in enumerate(wh_names):
        daily = rate * wh_factor[w]
        cover = rng.uniform(0, 80, n) if wname != cpd else rng.uniform(30, 160, n)
        qty = np.floor(daily * cover + rng.poisson(0.3, n))
        held = rng.random(n) < (0.35 if wname != cpd else 0.9)
        qty = np.where(held | (daily > 0.05), qty, 0)
        qty[np.isin(np.arange(n), dead) & (rng.random(n) < 0.5)] += rng.integers(2, 30)
        stock_rows.append(pd.DataFrame({"i": np.arange(n), "wh": wname, "qty": qty}))
    for extra in ("Slow Moving - SP", "Hidden - SP"):
        idx = rng.choice(n, n // 20, replace=False)
        stock_rows.append(pd.DataFrame({"i": idx, "wh": extra, "qty": rng.integers(1, 20, len(idx)).astype(float)}))
    st = pd.concat(stock_rows, ignore_index=True)
    st = st[st["qty"] != 0].reset_index(drop=True)
    neg = rng.choice(len(st), 3, replace=False)
    st.loc[neg, "qty"] = -st.loc[neg, "qty"].clip(upper=2)

    def stock_frame(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "Item": codes[df["i"]], "Item Name": names[df["i"]], "Item Group": group[df["i"]], "Warehouse": df["wh"],
            "Stock UOM": "Nos", "Balance Qty": df["qty"], "Balance Value": np.round(df["qty"] * cost[df["i"]], 2),
            "Valuation Rate": cost[df["i"]], "In Qty": 0, "Out Qty": 0, "Reserved Stock": 0,
        })

    # A week earlier = today's stock + what was sold since, + a few units that later went missing.
    week_ago = end - timedelta(days=7)
    sold = s[s["d"] > (week_ago - start).days].groupby(["i", "w"])["qty"].sum().reset_index()
    sold["wh"] = np.array(wh_names)[sold["w"]]
    prev = st.merge(sold[["i", "wh", "qty"]].rename(columns={"qty": "sold"}), on=["i", "wh"], how="outer").fillna({"qty": 0, "sold": 0})
    prev["qty"] = prev["qty"] + prev["sold"]
    lost = prev[(prev["wh"] == branches[1]) & (prev["qty"] > 0)].sort_values("qty", ascending=False).head(3).index
    prev.loc[lost, "qty"] += 12
    prev = prev[prev["qty"] != 0]

    # ---- purchase orders ---------------------------------------------------
    suppliers = {b: (f"SUP-{k + 1:03d}", f"{b} Parts Trading Co.") for k, b in enumerate(BRANDS)}
    lead = {b: int(rng.integers(20, 70)) for b in BRANDS}
    po_items = rng.choice(n, max(10, n // 3), replace=True)
    po_dates = rng.integers(0, len(days), len(po_items))
    po_qty = np.maximum(1, np.round(rate[po_items] * 21 * rng.uniform(20, 60, len(po_items)) / 10)) + 1
    arrive = np.array([po_dates[k] + lead[brand[i]] + rng.integers(-5, 10) for k, i in enumerate(po_items)])
    received = np.where(arrive <= len(days) - 1, po_qty, np.where(rng.random(len(po_items)) < 0.2, np.floor(po_qty / 2), 0))
    on_order_dead = dead[: max(1, len(dead) // 3)]
    po = pd.DataFrame({
        "PO No.": [f"PUR-ORD-26-{k // 4 + 1:05d}" for k in range(len(po_items))],
        "Supplier": [suppliers[brand[i]][0] for i in po_items], "Supplier Name": [suppliers[brand[i]][1] for i in po_items],
        "PO Date": [days[d].strftime("%d-%m-%Y") for d in po_dates], "Item": codes[po_items], "Item Name": names[po_items],
        "Warehouse": cpd,
        "Last Receipt": [days[min(a, len(days) - 1)].strftime("%d-%m-%Y") if r > 0 else "" for a, r in zip(arrive, received)],
        "PO Qty": po_qty, "Received Qty": received, "Cancelled Qty": 0, "Remaining Qty": po_qty - received,
        "Status": np.where(po_qty - received > 0, "To Receive and Bill", "Completed"),
        "Currency": np.where(rng.random(len(po_items)) < 0.3, "USD", "SAR"),
        "PO Price": np.round(cost[po_items], 2),
    })
    po.loc[po["Currency"] == "USD", "PO Price"] = np.round(po.loc[po["Currency"] == "USD", "PO Price"] / 3.75, 2)
    dead_po = pd.DataFrame({
        "PO No.": [f"PUR-ORD-26-9{k:04d}" for k in range(len(on_order_dead))],
        "Supplier": [suppliers[brand[i]][0] for i in on_order_dead], "Supplier Name": [suppliers[brand[i]][1] for i in on_order_dead],
        "PO Date": (end - timedelta(days=10)).strftime("%d-%m-%Y"), "Item": codes[on_order_dead], "Item Name": names[on_order_dead],
        "Warehouse": cpd, "Last Receipt": "", "PO Qty": 40.0, "Received Qty": 0.0, "Cancelled Qty": 0, "Remaining Qty": 40.0,
        "Status": "To Receive and Bill", "Currency": "SAR", "PO Price": cost[on_order_dead],
    })
    po = pd.concat([po, dead_po], ignore_index=True)

    # ---- stock entries (transfers CPD -> branch via In Transit) -----------
    tr_items = rng.choice(n, max(10, n // 4), replace=True)
    tr_days = rng.integers(0, len(days), len(tr_items))
    tr_dest = rng.choice(branches, len(tr_items))
    tr_qty = np.maximum(1, np.round(rate[tr_items] * 20))
    recent = tr_days >= len(days) - 4
    entries = pd.DataFrame({
        "ID": [f"MAT-STE-26-{k:06d}" for k in range(len(tr_items))], "docstatus": 1, "Stock Entry Type": "Material Transfer",
        "Posting Date": [days[d].strftime("%d-%m-%Y") for d in tr_days], "Final Target Warehouse": tr_dest,
        "Per Transferred": np.where(recent, 0, 100), "Qty (Stock Entry Detail)": tr_qty,
        "Valuation Rate (Stock Entry Detail)": cost[tr_items], "Source Warehouse (Stock Entry Detail)": cpd,
        "Target Warehouse (Stock Entry Detail)": "In Transit - SP", "Item Code (Stock Entry Detail)": codes[tr_items],
        "Item Name (Stock Entry Detail)": names[tr_items], "owner": "storekeeper@example.com",
        "Default Source Warehouse": cpd, "Default Target Warehouse": "In Transit - SP",
    })
    drafts = entries.sample(min(5, len(entries)), random_state=1).assign(docstatus=0)
    drafts["ID"] = [f"MAT-STE-26-D{k:05d}" for k in range(len(drafts))]
    entries = pd.concat([entries, drafts], ignore_index=True)

    tag = f"{start.isoformat()}_{end.isoformat()}"
    ext = ".csv" if args.csv else ".xlsx"
    files = {
        "sales": out / f"Spare Part Item Sales Detailed_{tag}{ext}",
        "stock_prev": out / f"Stock Balance Spare Parts_{week_ago.isoformat()}_{week_ago.isoformat()}{ext}",
        "stock": out / f"Stock Balance Spare Parts_{end.isoformat()}_{end.isoformat()}{ext}",
        "purchase": out / f"Item-Wise Purchase History_{tag}{ext}",
        "transfer": out / f"Stock Entry_{tag}{ext}",
    }
    frames = {"sales": sales, "stock_prev": stock_frame(prev), "stock": stock_frame(st), "purchase": po, "transfer": entries}
    for k, f in files.items():
        df = frames[k]
        if args.csv:
            df.to_csv(f, index=False)
        else:
            with pd.ExcelWriter(f, engine="openpyxl") as xw:
                # Real exports start with a title line above the header row.
                pd.DataFrame([[f.stem]]).to_excel(xw, index=False, header=False, startrow=0)
                df.to_excel(xw, index=False, startrow=1)
        print(f"{k:11s} {len(df):>9,} rows -> {f}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skus", type=int, default=3000)
    ap.add_argument("--branches", type=int, default=20)
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--end", help="last sales date, YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--demand-scale", type=float, default=1.0, help="multiply every item's sales rate")
    ap.add_argument("--csv", action="store_true", help="write CSV instead of xlsx (needed above ~1M rows)")
    ap.add_argument("--out", default="sample_data")
    make(ap.parse_args())


if __name__ == "__main__":
    main()
