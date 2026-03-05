"""Metric calculations for resource-allocation optimization evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


DEFAULT_CAPACITY_SECONDS = 28800.0
EPSILON = 1e-12


def _successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a completed allocation (non-timeout)."""
    if "timed_out" not in df.columns:
        return df.copy()
    return df[df["timed_out"] != True].copy()  # noqa: E712


def compute_case_cycle_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute case-level cycle time metrics in minutes.

    Cycle time per case = max(end_seconds) - min(arrival_seconds).
    """
    valid = _successful_rows(df)
    needed = {"case", "arrival_seconds", "end_seconds"}
    if valid.empty or not needed.issubset(valid.columns):
        return {"mean": np.nan, "p90": np.nan}

    valid = valid.dropna(subset=["case", "arrival_seconds", "end_seconds"])
    if valid.empty:
        return {"mean": np.nan, "p90": np.nan}

    per_case = valid.groupby("case").agg(
        case_start=("arrival_seconds", "min"),
        case_end=("end_seconds", "max"),
    )
    case_cycles = (per_case["case_end"] - per_case["case_start"]).astype(float)
    case_cycles = case_cycles[case_cycles >= 0]
    if case_cycles.empty:
        return {"mean": np.nan, "p90": np.nan}

    return {
        "mean": float(case_cycles.mean() / 60.0),
        "p90": float(np.percentile(case_cycles.values, 90) / 60.0),
    }


def compute_resource_occupation(df: pd.DataFrame, include_system: bool) -> float:
    """
    Compute average resource occupation (%) over observed simulation window.
    """
    valid = _successful_rows(df)
    needed = {"resource", "start_seconds", "end_seconds", "service_seconds", "is_system"}
    if valid.empty or not needed.issubset(valid.columns):
        return np.nan

    valid = valid.dropna(subset=["resource", "start_seconds", "end_seconds"])
    if not include_system:
        valid = valid[valid["is_system"] != True]  # noqa: E712
    if valid.empty:
        return np.nan

    window = float(valid["end_seconds"].max() - valid["start_seconds"].min())
    if window <= 0:
        return np.nan

    service = valid["service_seconds"].where(
        valid["service_seconds"].notna(),
        valid["end_seconds"] - valid["start_seconds"],
    )
    by_resource = valid.assign(service_seconds=service).groupby("resource")["service_seconds"].sum()

    occupation = (by_resource / window).clip(lower=0.0, upper=1.0)
    return float(occupation.mean() * 100.0) if not occupation.empty else np.nan


def compute_weighted_jain_fairness(df: pd.DataFrame, capacities: Dict[str, float]) -> float:
    """
    Compute weighted Jain fairness over human resources.

    x_r = workload seconds per resource
    y_r = x_r / capacity_r
    J = (sum(y_r)^2) / (n * sum(y_r^2))
    """
    valid = _successful_rows(df)
    needed = {"resource", "is_system", "service_seconds", "start_seconds", "end_seconds"}
    if valid.empty or not needed.issubset(valid.columns):
        return np.nan

    valid = valid[(valid["is_system"] != True)].dropna(subset=["resource"])  # noqa: E712
    if valid.empty:
        return 1.0

    service = valid["service_seconds"].where(
        valid["service_seconds"].notna(),
        valid["end_seconds"] - valid["start_seconds"],
    )
    load = valid.assign(service_seconds=service).groupby("resource")["service_seconds"].sum()

    n_resources = len(load)
    if n_resources <= 1:
        return 1.0

    normalized = []
    for resource, workload in load.items():
        cap = float(capacities.get(resource, DEFAULT_CAPACITY_SECONDS))
        cap = cap if cap > EPSILON else DEFAULT_CAPACITY_SECONDS
        normalized.append(float(workload) / cap)

    y = np.array(normalized, dtype=float)
    denominator = n_resources * float(np.sum(np.square(y)))
    if denominator <= EPSILON:
        return 1.0

    fairness = (float(np.sum(y)) ** 2) / denominator
    return float(np.clip(fairness, 0.0, 1.0))


def compute_wait_metrics(df: pd.DataFrame, sla_threshold_seconds: int = 3600) -> Dict[str, float]:
    """Compute waiting-time metrics from wait_seconds."""
    if "wait_seconds" not in df.columns or df.empty:
        return {"avg_wait_min": np.nan, "service_level_pct": np.nan}

    waits = pd.to_numeric(df["wait_seconds"], errors="coerce").dropna()
    if waits.empty:
        return {"avg_wait_min": np.nan, "service_level_pct": np.nan}

    return {
        "avg_wait_min": float(waits.mean() / 60.0),
        "service_level_pct": float((waits <= float(sla_threshold_seconds)).mean() * 100.0),
    }


def compute_optimization_metrics(
    df: pd.DataFrame,
    capacities: Dict[str, float],
    sla_threshold_seconds: int = 3600,
) -> Dict[str, float]:
    """Compute the six optimization metrics used for strategy comparison."""
    cycle = compute_case_cycle_metrics(df)
    wait = compute_wait_metrics(df, sla_threshold_seconds=sla_threshold_seconds)

    return {
        "Avg Case Cycle Time (min)": cycle["mean"],
        "Avg Resource Occupation Humans (%)": compute_resource_occupation(df, include_system=False),
        "Weighted Fairness (Jain, humans)": compute_weighted_jain_fairness(df, capacities),
        "P90 Case Cycle Time (min)": cycle["p90"],
        "Avg Wait Time (min)": wait["avg_wait_min"],
        "Service Level <=60min (%)": wait["service_level_pct"],
    }
