"""Runtime settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", "sqlite:///./inventory.db"))

    # Demand history the forecasts learn from (days).
    history_days: int = field(default_factory=lambda: _i("HISTORY_DAYS", 182))

    # Target service level (probability of not stocking out in a cycle) per ABC class.
    service_level_a: float = field(default_factory=lambda: _f("SERVICE_LEVEL_A", 0.98))
    service_level_b: float = field(default_factory=lambda: _f("SERVICE_LEVEL_B", 0.95))
    service_level_c: float = field(default_factory=lambda: _f("SERVICE_LEVEL_C", 0.90))

    # Replenishment policy (days).
    review_period_days: int = field(default_factory=lambda: _i("REVIEW_PERIOD_DAYS", 7))
    order_cycle_days: int = field(default_factory=lambda: _i("ORDER_CYCLE_DAYS", 30))
    transfer_lead_days: int = field(default_factory=lambda: _i("TRANSFER_LEAD_DAYS", 3))
    default_supplier_lead_days: int = field(default_factory=lambda: _i("DEFAULT_SUPPLIER_LEAD_DAYS", 45))
    # A donor warehouse keeps at least this many days of its own demand.
    donor_keep_days: int = field(default_factory=lambda: _i("DONOR_KEEP_DAYS", 60))
    min_transfer_value: float = field(default_factory=lambda: _f("MIN_TRANSFER_VALUE", 100.0))

    # Anomaly thresholds.
    dead_stock_days: int = field(default_factory=lambda: _i("DEAD_STOCK_DAYS", 180))
    spike_window_days: int = field(default_factory=lambda: _i("SPIKE_WINDOW_DAYS", 7))
    spike_sigma: float = field(default_factory=lambda: _f("SPIKE_SIGMA", 4.0))
    shrinkage_min_value: float = field(default_factory=lambda: _f("SHRINKAGE_MIN_VALUE", 200.0))

    # USD purchase prices are converted at this rate.
    usd_to_sar: float = field(default_factory=lambda: _f("USD_TO_SAR", 3.75))

    # Claude agent.
    agent_model: str = field(default_factory=lambda: os.environ.get("AGENT_MODEL", "claude-opus-5-5"))
    agent_effort: str = field(default_factory=lambda: os.environ.get("AGENT_EFFORT", "medium"))
    agent_max_turns: int = field(default_factory=lambda: _i("AGENT_MAX_TURNS", 12))

    def service_level(self, abc: str) -> float:
        return {"A": self.service_level_a, "B": self.service_level_b}.get(abc, self.service_level_c)


def get_settings() -> Settings:
    return Settings()
