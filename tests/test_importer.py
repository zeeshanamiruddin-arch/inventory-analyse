from datetime import date

import pandas as pd
import pytest

from inventory_ai.importers.erp_excel import (
    ImportError_, detect, make_date_parser, parse_file, parse_purchase, parse_sales, parse_stock_entry,
)


def raw(header, rows, title=True):
    """Shape a sheet the way the ERP exports it: optional title line, header row, data rows."""
    lines = ([["Report title"] + [None] * (len(header) - 1)] if title else []) + [header] + rows
    return pd.DataFrame(lines, dtype=object)


SALES_H = ["Pos Profile", "Warehouse", "Invoice Number", "Posting Date", "Customer Number", "Customer Name", "Sales Person",
           "Item Code", "Item Name", "Parent Item Group", "Item Group", "Transaction Type", "Qty", "Price List Rate",
           "Discount Amount", "Rate", "Net Line Sales", "Total Cost", "Total Profit", "VAT", "Total Line"]


def sale(pos, wh, inv, day, code, qty, net, cost, tt="Cash"):
    return [pos, wh, inv, day, "C1", "Cash Customer", "SP1", code, "Oil Filter", "Toyota", "Filters", tt, qty, 10, 0, 10, net, cost, net - cost, 0, net]


def test_detects_each_export():
    assert detect(raw(SALES_H, [])) == "sales"
    assert detect(raw(["Item", "Warehouse", "Balance Qty", "Balance Value"], [])) == "stock"
    assert detect(raw(["PO No.", "Item", "Remaining Qty"], [])) == "purchase"
    assert detect(raw(["ID", "Stock Entry Type", "Qty (Stock Entry Detail)"], [])) == "transfer"
    assert detect(raw(["Foo", "Bar"], [])) is None


def test_sales_skips_totals_fills_warehouse_and_reads_day_first_dates():
    df = raw(SALES_H, [
        sale("POS Olaya", "P01 - Olaya - SP", "INV1", "13-05-2026", "A1", 2, 20, 12),
        sale("POS Olaya", "", "INV2", "14-05-2026", "A1", 1, 5, 6),  # warehouse missing, sold below cost
        sale("POS Olaya", "P01 - Olaya - SP", "INV3", "01-06-2026", "A2", -1, -10, -6, "Return"),
        ["Grand Total"] + [None] * 11 + [2, None, None, None, 15, None, None, None, None],
    ])
    pf = parse_sales(df, "Sales_2026-05-01_2026-06-30.xlsx")
    assert len(pf.frame) == 3
    assert pf.report["summary_rows_skipped"] == 1
    assert pf.report["warehouse_filled_from_pos"] == 1
    assert pf.frame.loc[1, "warehouse"] == "P01 - Olaya - SP"
    assert pf.frame.loc[0, "posting_date"] == date(2026, 5, 13)
    assert pf.report["below_cost_lines"] == 1 and pf.report["return_lines"] == 1
    # The file name's date range wins, so a re-import replaces exactly that range.
    assert (pf.date_from, pf.date_to) == (date(2026, 5, 1), date(2026, 6, 30))


def test_missing_required_column_is_an_error():
    df = raw(["Invoice Number", "Item Code", "Net Line Sales"], [["I", "A", 1]])
    with pytest.raises(ImportError_, match="Posting Date"):
        parse_sales(df, "x.xlsx")


def test_date_parser_formats():
    p = make_date_parser(["03-04-2026", "25-04-2026"])
    assert p("03-04-2026") == date(2026, 4, 3)
    assert p("2026-04-03") == date(2026, 4, 3)
    assert p(46115) == date(2026, 4, 3)  # Excel serial
    assert make_date_parser(["04-25-2026"])("04-25-2026") == date(2026, 4, 25)
    assert p("") is None


def test_purchase_converts_usd_and_zeroes_closed_lines():
    h = ["PO No.", "Supplier", "Supplier Name", "PO Date", "Item", "Item Name", "Warehouse", "Last Receipt", "PO Qty",
         "Received Qty", "Cancelled Qty", "Remaining Qty", "Status", "Currency", "PO Price"]
    df = raw(h, [
        ["PO1", "S1", "Sup", "01-05-2026", "A1", "x", "CPD - SP", "", 10, 0, 0, 10, "To Receive and Bill", "USD", 2],
        ["PO2", "S1", "Sup", "02-05-2026", "A2", "x", "CPD - SP", "", 10, 4, 0, 6, "Closed", "SAR", 5],
    ])
    pf = parse_purchase(df, "po.xlsx", usd_to_sar=3.75)
    assert pf.frame.loc[0, "price_sar"] == pytest.approx(7.5)
    assert pf.frame.loc[1, "remaining_qty"] == 0
    assert pf.report["closed_lines_with_remaining"] == 1


def test_stock_entry_keeps_submitted_transfers_and_flags_in_transit():
    h = ["ID", "docstatus", "Stock Entry Type", "Posting Date", "Final Target Warehouse", "Per Transferred",
         "Qty (Stock Entry Detail)", "Valuation Rate (Stock Entry Detail)", "Source Warehouse (Stock Entry Detail)",
         "Target Warehouse (Stock Entry Detail)", "Item Code (Stock Entry Detail)", "Item Name (Stock Entry Detail)"]
    df = raw(h, [
        ["E1", 1, "Material Transfer", "01-05-2026", "P01 - A - SP", 0, 5, 2, "CPD - SP", "In Transit - SP", "A1", "x"],
        ["E2", 1, "Material Transfer", "02-05-2026", "P01 - A - SP", 100, 3, 2, "CPD - SP", "In Transit - SP", "A1", "x"],
        ["E3", 1, "Material Transfer", "03-05-2026", "", 100, 3, 2, "In Transit - SP", "P01 - A - SP", "A1", "x"],  # arrival leg
        ["E4", 0, "Material Transfer", "03-05-2026", "", 0, 3, 2, "CPD - SP", "P02 - B - SP", "A1", "x"],  # draft
        ["E5", 2, "Material Transfer", "03-05-2026", "", 0, 3, 2, "CPD - SP", "P02 - B - SP", "A1", "x"],  # cancelled
        ["E6", 1, "Material Issue", "03-05-2026", "", 0, 1, 2, "CPD - SP", "", "A1", "x"],
    ])
    pf = parse_stock_entry(df, "se.xlsx")
    f = pf.frame
    assert list(f["entry_id"]) == ["E1", "E2"]
    assert list(f["destination"]) == ["P01 - A - SP", "P01 - A - SP"]
    assert list(f["in_transit"]) == [True, False]
    assert pf.report["draft_entries"] == 1 and pf.report["cancelled_rows"] == 1
    assert pf.report["arrival_legs_skipped"] == 1 and pf.report["other_entry_types"] == {"Material Issue": 1}


def test_parse_file_reads_xlsx_with_title_row(tmp_path):
    p = tmp_path / "Stock Balance_2026-06-30_2026-06-30.xlsx"
    df = raw(["Item", "Item Name", "Warehouse", "Balance Qty", "Balance Value", "Valuation Rate"],
             [["A1", "x", "CPD - SP", 4, 40, 10], [None, None, None, 4, 40, None]])
    df.to_excel(p, header=False, index=False)
    pf = parse_file(p)
    assert pf.kind == "stock" and len(pf.frame) == 1 and pf.date_to == date(2026, 6, 30)
