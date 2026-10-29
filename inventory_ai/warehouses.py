"""Warehouse classification from ERP warehouse names.

Follows the naming used in the ERP exports: a central "CPD" store, branch stores
("P01 - <name> - SP"), and non-selling stores for transit, slow moving, hidden
and scrap stock.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CENTRAL = "Central"
BRANCH = "Branch"
TRANSIT = "In transit"
SLOW = "Slow moving"
HIDDEN = "Hidden"
SCRAP = "Scrap"
UNASSIGNED = "Unassigned"

SELLABLE = {CENTRAL, BRANCH}


@dataclass(frozen=True)
class WarehouseInfo:
    name: str
    label: str
    category: str

    @property
    def sellable(self) -> bool:
        return self.category in SELLABLE


def classify(name: str) -> WarehouseInfo:
    label = re.sub(r"^P\d+\s*-\s*", "", name, flags=re.I)
    label = re.sub(r"\s*-\s*SP$", "", label, flags=re.I)
    label = re.sub(r"\s+", " ", label).strip() or name
    if re.fullmatch(r"UNASSIGNED|UNKNOWN|", name.strip(), flags=re.I):
        return WarehouseInfo(name, "No warehouse on line", UNASSIGNED)
    if re.search(r"SLOW\s*MOVING", name, re.I):
        cat = SLOW
    elif re.search(r"SCRAP", name, re.I):
        cat = SCRAP
    elif re.search(r"HIDDEN", name, re.I):
        cat = HIDDEN
    elif re.search(r"TRANSIT|OVERSEAS|RECEIVING", name, re.I):
        cat = TRANSIT
    elif re.search(r"CPD", name, re.I):
        cat = CENTRAL
    else:
        cat = BRANCH
    return WarehouseInfo(name, label, cat)


def is_transit(name: str | None) -> bool:
    return bool(name) and bool(re.search(r"TRANSIT", name, re.I))
