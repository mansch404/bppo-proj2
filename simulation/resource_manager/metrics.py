"""Metric calculations for resource-allocation optimization evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Set

import numpy as np
import pandas as pd


DEFAULT_CAPACITY_SECONDS = 28800.0
EPSILON = 1e-12


def _successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a completed allocation (non-timeout)."""
    if "timed_out" not in df.columns:
        return df.copy()
    return df[df["timed_out"] != True].copy()  # noqa: E712


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce series to numeric while preserving index."""
    return pd.to_numeric(series, errors="coerce")


def _safe_amounts(
    amount_series: pd.Series,
    *,
    fallback_amount: float = 1.0,
) -> pd.Series:
    """
    Parse monetary amounts with a fallback for missing values.
    Negative values are clipped to 0.
    """
    numeric = _to_numeric(amount_series).clip(lower=0.0)
    return numeric.where(numeric.notna(), float(fallback_amount))


def _safe_log_weights(
    amount_series: pd.Series,
    *,
    missing_weight: float = 1.0,
) -> pd.Series:
    """
    Convert requested amounts into log-scaled weights.
    Missing values map directly to `missing_weight`.
    """
    numeric = _to_numeric(amount_series).clip(lower=0.0)
    weights = np.log1p(numeric)
    return weights.where(numeric.notna(), float(missing_weight))


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


def compute_value_weighted_wait(
    df: pd.DataFrame,
    *,
    missing_weight: float = 1.0,
) -> float:
    """
    Mean wait time (minutes), weighted by log1p(requested_amount).
    Missing amounts use a fixed fallback weight.
    """
    if "wait_seconds" not in df.columns or df.empty:
        return np.nan

    waits = _to_numeric(df["wait_seconds"])
    if "requested_amount" in df.columns:
        weights = _safe_log_weights(df["requested_amount"], missing_weight=missing_weight)
    else:
        weights = pd.Series(float(missing_weight), index=df.index, dtype=float)

    mask = waits.notna() & weights.notna()
    waits = waits[mask]
    weights = weights[mask]
    if waits.empty:
        return np.nan

    denominator = float(weights.sum())
    if denominator <= EPSILON:
        return float(waits.mean() / 60.0)

    weighted_wait = float((waits * weights).sum()) / denominator
    return float(weighted_wait / 60.0)


def compute_value_at_risk_sla_breach(
    df: pd.DataFrame,
    *,
    sla_threshold_seconds: int = 3600,
    fallback_amount: float = 1.0,
) -> float:
    """
    Percentage of total requested amount associated with SLA-breaching waits.
    """
    if "wait_seconds" not in df.columns or df.empty:
        return np.nan

    waits = _to_numeric(df["wait_seconds"])
    if "requested_amount" in df.columns:
        amounts = _safe_amounts(df["requested_amount"], fallback_amount=fallback_amount)
    else:
        amounts = pd.Series(float(fallback_amount), index=df.index, dtype=float)

    mask = waits.notna() & amounts.notna()
    waits = waits[mask]
    amounts = amounts[mask]
    if waits.empty:
        return np.nan

    total_amount = float(amounts.sum())
    if total_amount <= EPSILON:
        return np.nan

    breach_amount = float(amounts[waits > float(sla_threshold_seconds)].sum())
    return float((breach_amount / total_amount) * 100.0)


def compute_case_handover_rate(df: pd.DataFrame) -> float:
    """
    Average per-case handover ratio:
    resource switches / (tasks - 1), averaged over all cases.
    """
    valid = _successful_rows(df)
    needed = {"case", "resource"}
    if valid.empty or not needed.issubset(valid.columns):
        return np.nan

    order_col = "start_seconds" if "start_seconds" in valid.columns else None
    if order_col is None and "arrival_seconds" in valid.columns:
        order_col = "arrival_seconds"

    rates = []
    for _, case_df in valid.dropna(subset=["case", "resource"]).groupby("case"):
        if case_df.empty:
            continue

        if order_col is not None:
            case_df = case_df.sort_values(order_col, kind="stable")

        resources = case_df["resource"].astype(str).to_list()
        if len(resources) <= 1:
            rates.append(0.0)
            continue

        switches = sum(
            1 for prev, curr in zip(resources, resources[1:]) if prev != curr
        )
        rates.append(float(switches) / float(len(resources) - 1))

    if not rates:
        return np.nan
    return float(np.mean(rates))


def compute_automation_leverage(
    df: pd.DataFrame,
    automation_eligible_activities: Iterable[str],
) -> float:
    """
    Percentage of eligible tasks that were handled by system resources.
    """
    valid = _successful_rows(df)
    needed = {"activity", "is_system"}
    if valid.empty or not needed.issubset(valid.columns):
        return np.nan

    eligible = {str(a) for a in automation_eligible_activities}
    if not eligible:
        return np.nan

    eligible_rows = valid[valid["activity"].astype(str).isin(eligible)]
    if eligible_rows.empty:
        return np.nan

    is_system = eligible_rows["is_system"].fillna(False).astype(bool)
    return float(is_system.mean() * 100.0)


def compute_human_capacity_stress_ratio(
    capacities: Mapping[str, float],
    daily_work_seconds: Mapping[str, Mapping[str, float]],
) -> float:
    """
    Capacity stress ratio (%) = overload seconds / total capacity seconds.
    Overload is counted only for non-system resources tracked in capacities.
    """
    if not capacities:
        return np.nan

    overload_seconds = 0.0
    capacity_seconds = 0.0

    for resource, day_usage in daily_work_seconds.items():
        if resource not in capacities:
            continue

        capacity = float(capacities.get(resource, DEFAULT_CAPACITY_SECONDS))
        if capacity <= EPSILON:
            capacity = DEFAULT_CAPACITY_SECONDS

        for used in day_usage.values():
            used_seconds = float(used)
            capacity_seconds += capacity
            overload_seconds += max(0.0, used_seconds - capacity)

    if capacity_seconds <= EPSILON:
        return 0.0

    return float((overload_seconds / capacity_seconds) * 100.0)


def compute_custom_optimization_metrics(
    df: pd.DataFrame,
    capacities: Dict[str, float],
    daily_work_seconds: Dict[str, Dict[str, float]],
    automation_eligible_activities: Set[str],
    sla_threshold_seconds: int = 3600,
) -> Dict[str, float]:
    """Compute the advanced custom metric suite for strategy comparison."""
    return {
        "Value-Weighted Wait (min)": compute_value_weighted_wait(df),
        "Value-at-Risk SLA Breach (%)": compute_value_at_risk_sla_breach(
            df, sla_threshold_seconds=sla_threshold_seconds
        ),
        "Case Handover Rate": compute_case_handover_rate(df),
        "Automation Leverage on Eligible Tasks (%)": compute_automation_leverage(
            df,
            automation_eligible_activities=automation_eligible_activities,
        ),
        "Human Capacity Stress Ratio (%)": compute_human_capacity_stress_ratio(
            capacities=capacities,
            daily_work_seconds=daily_work_seconds,
        ),
    }
