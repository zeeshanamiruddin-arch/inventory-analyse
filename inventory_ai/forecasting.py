"""Demand forecasting for every item in every selling warehouse.

200k SKUs x 20 branches is ~4M time series, too many for one model per
series. Instead each warehouse is forecast as one matrix (items x days)
with vectorised smoothing methods:

* Demand pattern per series (Syntetos-Boylan): ADI = average days between
  sales, CV2 = squared variation of sale sizes.
    smooth (ADI < 1.32, CV2 < 0.49), erratic (ADI < 1.32), intermittent
    (CV2 < 0.49), lumpy (both high).
* Smooth/erratic series use simple exponential smoothing (SES).
* Intermittent/lumpy series - most spare parts - use TSB (Teunter-Syntetos-
  Babai), which forecasts the chance of a sale and its size separately and
  decays the forecast when an item stops selling.
* Forecast error (daily std) is the RMSE of one-step-ahead errors, floored
  at the Poisson level sqrt(mean).

ABC (share of sales value) and XYZ (weekly demand variability) classes set
the service level used for safety stock.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

ADI_CUT = 1.32
CV2_CUT = 0.49
SES_ALPHA = 0.15
TSB_ALPHA = 0.10  # size smoothing
TSB_BETA = 0.05  # probability smoothing
WARMUP = 28


@dataclass
class SeriesMatrix:
    items: np.ndarray  # item codes, shape (n,)
    days: pd.DatetimeIndex  # shape (T,)
    y: np.ndarray  # demand, shape (n, T)


def build_matrix(demand: pd.DataFrame, end: date, history_days: int) -> SeriesMatrix:
    """demand: rows of item_code, day, qty for one warehouse."""
    start = end - timedelta(days=history_days - 1)
    days = pd.date_range(start, end, freq="D")
    d = demand[(pd.to_datetime(demand["day"]) >= pd.Timestamp(start)) & (pd.to_datetime(demand["day"]) <= pd.Timestamp(end))]
    items = np.array(sorted(d["item_code"].unique()), dtype=object)
    y = np.zeros((len(items), len(days)), dtype=np.float64)
    if len(items):
        r = pd.Index(items).get_indexer(d["item_code"])
        c = (pd.to_datetime(d["day"]) - pd.Timestamp(start)).dt.days.to_numpy()
        np.add.at(y, (r, c), d["qty"].to_numpy(dtype=np.float64))
    return SeriesMatrix(items, days, y)


def classify_pattern(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz = y > 0
    count = nz.sum(axis=1)
    adi = np.where(count > 0, y.shape[1] / np.maximum(count, 1), np.inf)
    sizes_sum = np.where(nz, y, 0).sum(axis=1)
    mean_size = np.where(count > 0, sizes_sum / np.maximum(count, 1), 0.0)
    sq = np.where(nz, (y - mean_size[:, None]) ** 2, 0).sum(axis=1)
    var = np.where(count > 1, sq / np.maximum(count - 1, 1), 0.0)
    cv2 = np.where(mean_size > 0, var / np.maximum(mean_size, 1e-9) ** 2, 0.0)
    pattern = np.full(y.shape[0], "none", dtype=object)
    has = count > 0
    pattern[has & (adi < ADI_CUT) & (cv2 < CV2_CUT)] = "smooth"
    pattern[has & (adi < ADI_CUT) & (cv2 >= CV2_CUT)] = "erratic"
    pattern[has & (adi >= ADI_CUT) & (cv2 < CV2_CUT)] = "intermittent"
    pattern[has & (adi >= ADI_CUT) & (cv2 >= CV2_CUT)] = "lumpy"
    return pattern, adi, cv2


def ses(y: np.ndarray, alpha: float = SES_ALPHA, warmup: int = WARMUP) -> tuple[np.ndarray, np.ndarray]:
    """Simple exponential smoothing for all rows at once. Returns (level, rmse)."""
    n, T = y.shape
    w = min(warmup, T)
    level = y[:, :w].mean(axis=1) if w else np.zeros(n)
    sse = np.zeros(n)
    m = 0
    for t in range(w, T):
        err = y[:, t] - level
        sse += err * err
        m += 1
        level = level + alpha * err
    rmse = np.sqrt(sse / m) if m else np.zeros(n)
    return level, rmse


def tsb(y: np.ndarray, alpha: float = TSB_ALPHA, beta: float = TSB_BETA, warmup: int = WARMUP) -> tuple[np.ndarray, np.ndarray]:
    """Teunter-Syntetos-Babai method for intermittent demand. Returns (daily mean, rmse)."""
    n, T = y.shape
    w = min(warmup, T)
    head = y[:, :w] if w else y
    nz = head > 0
    p = nz.mean(axis=1) if head.shape[1] else np.zeros(n)
    z = np.where(nz.any(axis=1), np.where(nz, head, 0).sum(axis=1) / np.maximum(nz.sum(axis=1), 1), 0.0)
    if not nz.any(axis=1).all():
        # No sale during warm-up: start the size from the first sale anywhere in the window.
        first = np.where(y > 0, y, np.nan)
        with np.errstate(all="ignore"):
            fallback = np.nan_to_num(np.nanmean(first, axis=1))
        z = np.where(z > 0, z, fallback)
    sse = np.zeros(n)
    m = 0
    for t in range(w, T):
        yt = y[:, t]
        occ = yt > 0
        err = yt - p * z
        sse += err * err
        m += 1
        p = p + beta * (occ.astype(float) - p)
        z = np.where(occ, z + alpha * (yt - z), z)
    rmse = np.sqrt(sse / m) if m else np.zeros(n)
    return p * z, rmse


def xyz_class(y: np.ndarray) -> np.ndarray:
    weeks = y.shape[1] // 7
    if weeks < 2:
        return np.full(y.shape[0], "Z", dtype=object)
    wk = y[:, -weeks * 7 :].reshape(y.shape[0], weeks, 7).sum(axis=2)
    mu = wk.mean(axis=1)
    cv = np.where(mu > 0, wk.std(axis=1) / np.maximum(mu, 1e-9), np.inf)
    return np.where(cv < 0.5, "X", np.where(cv < 1.0, "Y", "Z")).astype(object)


def abc_classes(value_by_item: pd.Series) -> pd.Series:
    """A = items making the first 80% of sales value, B = next 15%, C = the rest."""
    v = value_by_item[value_by_item > 0].sort_values(ascending=False)
    if v.empty:
        return pd.Series(dtype=object)
    share = v.cumsum() / v.sum()
    prev = share.shift(1, fill_value=0.0)  # an item is A if the cumulative share before it is under 80%
    return pd.Series(np.where(prev < 0.80, "A", np.where(prev < 0.95, "B", "C")), index=v.index)


def forecast_warehouse(demand: pd.DataFrame, end: date, history_days: int) -> pd.DataFrame:
    """Forecast one warehouse. Returns item_code, daily_mean, daily_std, method, pattern, xyz."""
    sm = build_matrix(demand, end, history_days)
    if len(sm.items) == 0:
        return pd.DataFrame(columns=["item_code", "daily_mean", "daily_std", "method", "pattern", "xyz"])
    pattern, _, _ = classify_pattern(sm.y)
    ses_mean, ses_rmse = ses(sm.y)
    tsb_mean, tsb_rmse = tsb(sm.y)
    use_tsb = np.isin(pattern, ["intermittent", "lumpy", "none"])
    mean = np.clip(np.where(use_tsb, tsb_mean, ses_mean), 0, None)
    rmse = np.where(use_tsb, tsb_rmse, ses_rmse)
    std = np.maximum(rmse, np.sqrt(mean))
    return pd.DataFrame(
        {
            "item_code": sm.items,
            "daily_mean": mean,
            "daily_std": std,
            "method": np.where(use_tsb, "TSB", "SES"),
            "pattern": pattern,
            "xyz": xyz_class(sm.y),
        }
    )
