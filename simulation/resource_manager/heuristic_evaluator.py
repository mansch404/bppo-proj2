"""Evaluation suite for baseline and custom optimization metrics."""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Use a writable matplotlib cache directory to avoid repeated startup penalties.
_DEFAULT_MPLCONFIGDIR = Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"
if "MPLCONFIGDIR" not in os.environ:
    _DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_DEFAULT_MPLCONFIGDIR)
if "XDG_CACHE_HOME" not in os.environ:
    _xdg_cache = Path(__file__).resolve().parents[2] / ".cache"
    _xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(_xdg_cache)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pm4py

try:
    from .metrics import get_default_registry
    from .resource_manager import (
        AdvancedOptimizationPlanner,
        AdvancedResourceManager,
        AssignmentProblemPlanner,
        BatchPlanner,
        CaseHandlingPlanner,
        RandomPlanner,
        RoundRobinPlanner,
        DeepRLPlanner,
    )
except ImportError:
    from metrics import get_default_registry
    from resource_manager import (
        AdvancedOptimizationPlanner,
        AdvancedResourceManager,
        AssignmentProblemPlanner,
        BatchPlanner,
        CaseHandlingPlanner,
        RandomPlanner,
        RoundRobinPlanner,
        DeepRLPlanner,
    )


DEFAULT_NUM_CASES = 1000
DEFAULT_NUM_SEEDS = 3
DEFAULT_BATCH_WAIT_CAP_SECONDS = 3600.0
DEFAULT_ASSIGNMENT_WAIT_CAP_SECONDS = 3600.0
DEFAULT_NON_BATCH_WAIT_CAP_SECONDS = 86400.0 * 7

_REGISTRY = get_default_registry()

CUSTOM_METRICS = _REGISTRY.list_metrics(group="advanced")

CUSTOM_METRIC_DIRECTIONS = {
    name: direction
    for name, direction in _REGISTRY.get_directions().items()
    if name in CUSTOM_METRICS
}


def _print_perf_hints() -> None:
    if sys.version_info < (3, 11):
        print("[PERF] Consider Python 3.11+ for faster datetime/XES parsing.")
    try:
        import rustxes  # type: ignore # noqa: F401
    except Exception:
        print("[PERF] Optional speedup: `pip install rustxes` for faster XES I/O.")


class FullScaleEvaluator:
    """
    Reproducible evaluator for resource-allocation policies.
    Every strategy is tested under identical initial conditions.
    """

    def __init__(
        self,
        log_path: str,
        *,
        use_parsed_cache: bool = True,
        parsed_cache_path: Optional[str] = None,
    ):
        self.log_path = str(log_path)
        self._use_parsed_cache = bool(use_parsed_cache)
        self._parsed_cache_path = (
            Path(parsed_cache_path)
            if parsed_cache_path is not None
            else Path(self.log_path).with_suffix(".parsed.parquet")
        )
        self.raw_df = self._load_log_dataframe()

        self.start_time = datetime(2024, 1, 1, 8, 0)
        self.time_col = self._resolve_column(["time:timestamp", "timestamp"])
        self.case_col = self._resolve_column(["case:concept:name", "case", "case_id"])
        self.activity_col = self._resolve_column(["concept:name", "activity"])
        self.amount_col = self._resolve_optional_column(
            ["case:RequestedAmount", "RequestedAmount", "requested_amount"]
        )

        self.raw_df[self.time_col] = pd.to_datetime(
            self.raw_df[self.time_col], errors="coerce", utc=True
        )
        self.raw_df = self.raw_df.dropna(subset=[self.time_col]).sort_values(
            by=self.time_col, kind="stable"
        )
        self._manager_template = self._build_manager_template()

    def _load_log_dataframe(self) -> pd.DataFrame:
        if self.log_path.endswith(".xes"):
            if self._use_parsed_cache and self._parsed_cache_path.exists():
                try:
                    return pd.read_parquet(self._parsed_cache_path)
                except Exception:
                    pass

            parsed = pm4py.convert_to_dataframe(pm4py.read_xes(self.log_path))
            if self._use_parsed_cache:
                try:
                    self._parsed_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    parsed.to_parquet(self._parsed_cache_path, index=False)
                except Exception:
                    # Optional cache only; evaluator must still run if parquet engine is unavailable.
                    pass
            return parsed
        return pd.read_csv(self.log_path)

    def _build_manager_template(self) -> Dict[str, object]:
        template_manager = AdvancedResourceManager(self.start_time, RandomPlanner())
        template_manager.mine_organizational_model(self.raw_df.copy())

        return {
            "activity_permissions": copy.deepcopy(template_manager.activity_permissions),
            "roles": copy.deepcopy(template_manager.roles),
            "history_competence": copy.deepcopy(template_manager.history_competence),
            "system_resources": copy.deepcopy(template_manager.system_resources),
            "availability_matrix": copy.deepcopy(template_manager.availability_matrix),
            "daily_effort_capacities": copy.deepcopy(template_manager.daily_effort_capacities),
            "competence_scores": copy.deepcopy(template_manager.competence_scores),
            "senior_role_id": template_manager.senior_role_id,
            "setup_penalty_multiplier": template_manager.setup_penalty_multiplier,
        }

    def _apply_manager_template(self, manager: AdvancedResourceManager) -> None:
        template = self._manager_template
        manager.activity_permissions = copy.deepcopy(template["activity_permissions"])
        manager.roles = copy.deepcopy(template["roles"])
        manager.history_competence = copy.deepcopy(template["history_competence"])
        manager.system_resources = copy.deepcopy(template["system_resources"])
        manager.availability_matrix = copy.deepcopy(template["availability_matrix"])
        manager.daily_effort_capacities = copy.deepcopy(template["daily_effort_capacities"])
        manager.competence_scores = copy.deepcopy(template["competence_scores"])
        manager.senior_role_id = template["senior_role_id"]
        manager.setup_penalty_multiplier = template["setup_penalty_multiplier"]

        # These are per-run mutable states and must always start clean.
        manager.busy_until = {}
        manager.daily_work_seconds = {}
        manager.last_activity = {}
        manager.case_assignments = {}

    def _resolve_column(self, candidates: Sequence[str]) -> str:
        for candidate in candidates:
            if candidate in self.raw_df.columns:
                return candidate
        raise ValueError(f"Missing required column. Tried: {candidates}")

    def _resolve_optional_column(self, candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in self.raw_df.columns:
                return candidate
        return None

    def _build_tasks(self, num_cases: int) -> List[Dict[str, float]]:
        """
        Build a deterministic ordered task list from the first N cases.
        """
        if num_cases <= 0:
            return []

        first_event_per_case = (
            self.raw_df.groupby(self.case_col, dropna=False)[self.time_col]
            .min()
            .sort_values(kind="stable")
        )
        selected_cases = set(first_event_per_case.head(num_cases).index)

        task_df = self.raw_df[self.raw_df[self.case_col].isin(selected_cases)].sort_values(
            by=self.time_col, kind="stable"
        )
        tasks: List[Dict[str, float]] = []
        for _, row in task_df.iterrows():
            amount_value = row[self.amount_col] if self.amount_col is not None else 1.0
            amount = pd.to_numeric(pd.Series([amount_value]), errors="coerce").iloc[0]
            if pd.isna(amount):
                amount = 1.0

            duration = row["duration"] if "duration" in row.index else 1800.0
            duration = pd.to_numeric(pd.Series([duration]), errors="coerce").iloc[0]
            if pd.isna(duration) or float(duration) <= 0:
                duration = 1800.0

            tasks.append(
                {
                    "activity": str(row[self.activity_col]),
                    "case_id": str(row[self.case_col]),
                    "duration": float(duration),
                    "amount": float(max(0.0, amount)),
                }
            )
        return tasks

    def run_experiment(
        self,
        planner_instance,
        strategy_name: str,
        tasks: List[Dict[str, float]],
        *,
        seed: Optional[int] = None,
        sla_threshold_seconds: int = 3600,
        batch_wait_cap_seconds: float = DEFAULT_BATCH_WAIT_CAP_SECONDS,
        assignment_wait_cap_seconds: float = DEFAULT_ASSIGNMENT_WAIT_CAP_SECONDS,
        progress_every: Optional[int] = None,
        reuse_mined_model: bool = True,
    ) -> Dict[str, float]:
        """
        Run one seeded strategy simulation over a fixed task list.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        manager = AdvancedResourceManager(self.start_time, planner_instance)
        if reuse_mined_model:
            self._apply_manager_template(manager)
        else:
            manager.mine_organizational_model(self.raw_df.copy())

        sim_clock_seconds = 0.0
        execution_log = []
        timeout_count = 0
        heartbeat_every = progress_every
        if heartbeat_every is None:
            heartbeat_every = max(1, len(tasks) // 10)
        is_batch_strategy = isinstance(planner_instance, BatchPlanner)
        is_assignment_strategy = isinstance(planner_instance, AssignmentProblemPlanner)
        max_wait_seconds = (
            float(batch_wait_cap_seconds)
            if is_batch_strategy
            else (
                float(assignment_wait_cap_seconds)
                if is_assignment_strategy
                else DEFAULT_NON_BATCH_WAIT_CAP_SECONDS
            )
        )
        max_pending_seen = 0

        for idx, task in enumerate(tasks, start=1):
            activity = task["activity"]
            case_id = task["case_id"]
            base_duration = float(task["duration"])
            amount = float(task["amount"])
            arrival_seconds = float(sim_clock_seconds)

            allocated_res = None
            wait_time = 0.0
            timed_out = False

            while allocated_res is None:
                allocated_res = manager.request_resource(
                    activity,
                    arrival_seconds + wait_time,
                    base_duration,
                    case_id=case_id,
                    amount=amount,
                )

                if allocated_res is None:
                    wait_time += 300.0

                if wait_time > max_wait_seconds:
                    allocated_res = "TIMEOUT_FAIL"
                    timed_out = True

            start_seconds = float(arrival_seconds + wait_time)
            if timed_out or allocated_res == "TIMEOUT_FAIL":
                planner_timeout_hook = getattr(planner_instance, "on_task_timeout", None)
                if callable(planner_timeout_hook):
                    planner_timeout_hook(
                        case_id=case_id,
                        activity=activity,
                        amount=amount,
                    )
                end_seconds = np.nan
                service_seconds = np.nan
                timeout_count += 1
            else:
                busy_until = manager.busy_until.get(allocated_res)
                end_seconds = (
                    float((busy_until - manager.simulation_start_time).total_seconds())
                    if busy_until is not None
                    else np.nan
                )
                service_seconds = (
                    float(end_seconds - start_seconds)
                    if np.isfinite(end_seconds)
                    else np.nan
                )

            execution_log.append(
                {
                    "case": case_id,
                    "activity": activity,
                    "resource": allocated_res,
                    "wait_seconds": float(wait_time),
                    "requested_amount": amount,
                    "arrival_seconds": float(arrival_seconds),
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "service_seconds": service_seconds,
                    "timed_out": timed_out,
                    "is_system": allocated_res in manager.system_resources,
                    "is_senior": (
                        manager.senior_role_id is not None
                        and allocated_res in manager.roles.get(manager.senior_role_id, [])
                    ),
                }
            )

            # Keep deterministic task arrivals independent of duration.
            sim_clock_seconds += 5.0
            if idx % heartbeat_every == 0 or idx == len(tasks):
                if is_assignment_strategy:
                    pending_count = len(getattr(planner_instance, "pending_tasks", {}))
                    max_pending_seen = max(max_pending_seen, pending_count)
                    print(
                        f"  [{strategy_name}] progress {idx}/{len(tasks)} | "
                        f"timeouts={timeout_count} | pending={pending_count} "
                        f"(max={max_pending_seen})"
                    )
                else:
                    print(
                        f"  [{strategy_name}] progress {idx}/{len(tasks)} | "
                        f"timeouts={timeout_count}"
                    )

        return self._calculate_metrics(
            execution_log,
            strategy_name,
            manager,
            seed=seed,
            sla_threshold_seconds=sla_threshold_seconds,
        )

    def _calculate_metrics(
        self,
        log: List[Dict[str, float]],
        name: str,
        manager: AdvancedResourceManager,
        *,
        seed: Optional[int],
        sla_threshold_seconds: int,
    ) -> Dict[str, float]:
        """Compute baseline + custom metrics for one run via the registry."""
        df_res = pd.DataFrame(log)
        capacities = dict(getattr(manager, "daily_effort_capacities", {}))
        daily_work_seconds = {
            resource: dict(usage_by_day)
            for resource, usage_by_day in getattr(manager, "daily_work_seconds", {}).items()
        }
        automation_eligible_activities = {
            activity
            for activity, role_ids in manager.activity_permissions.items()
            if -1 in role_ids
        }

        timed_out_series = (
            df_res["timed_out"].fillna(False).astype(bool)
            if "timed_out" in df_res.columns
            else pd.Series(dtype=bool)
        )
        timeout_rate = (
            float(timed_out_series.mean() * 100.0)
            if len(timed_out_series) > 0
            else np.nan
        )

        registry = get_default_registry()
        all_metrics = registry.compute_all(
            df_res,
            capacities=capacities,
            sla_threshold_seconds=sla_threshold_seconds,
            daily_work_seconds=daily_work_seconds,
            automation_eligible_activities=automation_eligible_activities,
        )

        result = {"Strategy": name, "Seed": seed, "Timeout Rate (%)": timeout_rate}
        result.update(all_metrics)
        return result

    def run_seeded_study(
            self,
            *,
            num_cases: int = DEFAULT_NUM_CASES,
            seeds: Optional[Sequence[int]] = None,
            sla_threshold_seconds: int = 3600,
            batch_wait_cap_seconds: float = DEFAULT_BATCH_WAIT_CAP_SECONDS,
            assignment_wait_cap_seconds: float = DEFAULT_ASSIGNMENT_WAIT_CAP_SECONDS,
            reuse_mined_model: bool = True,
    ) -> pd.DataFrame:
        """
        Run all strategies over all seeds and return raw per-run metrics.
        """
        tasks = self._build_tasks(num_cases=num_cases)
        if not tasks:
            return pd.DataFrame()

        # Extract all unique resource names from the template
        # This ensures the DeepRLPlanner knows the size of its input/output layers
        all_resources = []
        if "roles" in self._manager_template:
            for role_resources in self._manager_template["roles"].values():
                all_resources.extend(role_resources)
        all_resources = sorted(list(set(all_resources)))

        run_seeds = list(range(DEFAULT_NUM_SEEDS)) if seeds is None else list(seeds)

        strategies: List[Tuple[str, Callable[[], object]]] = [
            ("Basic: Random", RandomPlanner),
            ("Heuristic: Round-Robin", RoundRobinPlanner),
            ("Pattern: Case Handling", CaseHandlingPlanner),
            ("Batch: Greedy k=5", lambda: BatchPlanner(k=5)),
            ("Advanced: Assignment Problem", lambda: AssignmentProblemPlanner(delta=1.2)),
            ("Advanced: OR-Optimized", AdvancedOptimizationPlanner),
            # Pass the extracted resource list to the DRL Planner
            ("Advanced: Deep RL (Trained)", lambda: DeepRLPlanner(
                all_resource_names=all_resources,
                model_path="rl_model_best.pt",
                is_training=False
            )),
        ]

        all_results = []
        for seed in run_seeds:
            for strategy_name, factory in strategies:
                print(f"Seed {seed:02d} | Testing {strategy_name} ...")
                t0 = time.perf_counter()

                # Create a fresh instance of the planner for this specific seed
                planner_instance = factory()

                metrics = self.run_experiment(
                    planner_instance,
                    strategy_name,
                    tasks=tasks,
                    seed=seed,
                    sla_threshold_seconds=sla_threshold_seconds,
                    batch_wait_cap_seconds=batch_wait_cap_seconds,
                    assignment_wait_cap_seconds=assignment_wait_cap_seconds,
                    reuse_mined_model=reuse_mined_model,
                )

                elapsed = time.perf_counter() - t0
                print(
                    f"Seed {seed:02d} | Done {strategy_name} in {elapsed:.2f}s | "
                    f"timeout_rate={metrics.get('Timeout Rate (%)', np.nan):.2f}%"
                )
                all_results.append(metrics)

        return pd.DataFrame(all_results)


def aggregate_strategy_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-run results by strategy (mean/std/95% CI).
    """
    if raw_results.empty:
        return pd.DataFrame()

    value_columns = [
        col for col in raw_results.columns if col not in {"Strategy", "Seed"}
    ]

    rows: List[Dict[str, float]] = []
    for strategy, strat_df in raw_results.groupby("Strategy", sort=False):
        row: Dict[str, float] = {"Strategy": strategy, "Runs": int(len(strat_df))}
        for metric in value_columns:
            values = pd.to_numeric(strat_df[metric], errors="coerce").dropna()
            n = len(values)
            if n == 0:
                row[f"{metric} Mean"] = np.nan
                row[f"{metric} Std"] = np.nan
                row[f"{metric} CI95"] = np.nan
                continue

            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
            row[f"{metric} Mean"] = mean
            row[f"{metric} Std"] = std
            row[f"{metric} CI95"] = ci95
        rows.append(row)

    return pd.DataFrame(rows)


def _normalize_metric(values: pd.Series, direction: str) -> pd.Series:
    """
    Min-max normalize metric values into [0, 1], where 1 is best.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    vmin = numeric.min(skipna=True)
    vmax = numeric.max(skipna=True)

    if pd.isna(vmin) or pd.isna(vmax):
        return pd.Series(np.nan, index=values.index, dtype=float)
    if abs(float(vmax) - float(vmin)) <= 1e-12:
        return pd.Series(1.0, index=values.index, dtype=float)

    if direction == "higher":
        out = (numeric - float(vmin)) / (float(vmax) - float(vmin))
    else:
        out = (float(vmax) - numeric) / (float(vmax) - float(vmin))
    return out.clip(lower=0.0, upper=1.0)


def compute_custom_metric_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build direction-aware normalized strategy ranking on custom metrics.
    """
    if summary_df.empty:
        return pd.DataFrame()

    rank_df = pd.DataFrame({"Strategy": summary_df["Strategy"]})
    normalized_cols: List[str] = []

    for metric in CUSTOM_METRICS:
        mean_col = f"{metric} Mean"
        if mean_col not in summary_df.columns:
            continue

        direction = CUSTOM_METRIC_DIRECTIONS[metric]
        norm_col = f"Norm {metric}"
        rank_df[norm_col] = _normalize_metric(summary_df[mean_col], direction=direction)
        normalized_cols.append(norm_col)

    if not normalized_cols:
        return pd.DataFrame({"Strategy": summary_df["Strategy"]})

    rank_df["Composite Custom Score"] = rank_df[normalized_cols].fillna(0.0).mean(axis=1)
    rank_df = rank_df.sort_values(
        by="Composite Custom Score", ascending=False, kind="stable"
    ).reset_index(drop=True)
    return rank_df


def _metric_filename(metric_name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", metric_name.strip()).strip("_")
    return token.lower() or "metric"


def save_metric_plots(summary_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """
    Save one error-bar bar chart per custom metric.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for metric in CUSTOM_METRICS:
        mean_col = f"{metric} Mean"
        ci_col = f"{metric} CI95"
        if mean_col not in summary_df.columns:
            continue

        means = pd.to_numeric(summary_df[mean_col], errors="coerce")
        ci = (
            pd.to_numeric(summary_df[ci_col], errors="coerce")
            if ci_col in summary_df.columns
            else pd.Series(0.0, index=summary_df.index)
        )
        if means.isna().all():
            continue

        x = np.arange(len(summary_df))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x, means.values, yerr=ci.values, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["Strategy"], rotation=20, ha="right")
        ax.set_title(f"{metric} (Mean ± 95% CI)")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()

        plot_path = output_dir / f"{_metric_filename(metric)}_mean_ci95.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


def write_custom_metric_appendix(output_path: Path) -> None:
    """
    Write report-ready metric appendix with formula, value, and toy example.
    """
    text = """# Advanced Custom Metric Appendix

This appendix documents the five custom metrics used to evaluate resource-allocation strategies.

## 1) Value-Weighted Wait (min)
- Formula: sum(wait_i * w_i) / sum(w_i) / 60, where w_i = log(1 + requested_amount_i).
- Why valuable: penalizes long waits on high-value cases more strongly than low-value cases.
- Example: two tasks each wait 30 minutes; amounts are 1,000 and 100,000. Plain average wait is equal, but this metric exposes the larger business impact of delaying the high-value case.

## 2) Value-at-Risk SLA Breach (%)
- Formula: 100 * (sum(requested_amount_i for wait_i > SLA) / sum(requested_amount_i)).
- Why valuable: quantifies how much business value is exposed to SLA violations, not just how many tasks are late.
- Example: Strategy A breaches SLA on 10 small cases, Strategy B breaches on 2 very large cases; count-based SLA looks better for B, but value-at-risk reveals higher downside.

## 3) Case Handover Rate
- Formula: average across cases of (resource switches / (tasks_in_case - 1)).
- Why valuable: captures continuity loss, coordination overhead, and context-switch friction per case.
- Example: two strategies have similar cycle times; one keeps ownership stable while the other frequently reassigns between resources. This metric distinguishes them.

## 4) Automation Leverage on Eligible Tasks (%)
- Formula: 100 * (eligible tasks handled by system resources / all eligible tasks).
- Why valuable: measures whether a strategy actually uses automation opportunities where available.
- Example: if both strategies have similar wait time but one offloads repetitive eligible tasks to bots, this metric shows better automation utilization.

## 5) Human Capacity Stress Ratio (%)
- Formula: 100 * (sum(overload_seconds) / sum(capacity_seconds)), with overload_seconds = max(0, used - capacity) per resource-day.
- Why valuable: detects hidden overcommitment that may not appear in short-term cycle-time metrics.
- Example: a strategy can produce low average wait by overloading a few humans well beyond daily capacity; this metric flags that operational risk.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def run_full_evaluation(
    *,
    log_path: Optional[str] = None,
    output_dir: str = "optimization_artifacts",
    num_cases: int = DEFAULT_NUM_CASES,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    sla_threshold_seconds: int = 3600,
    batch_wait_cap_seconds: float = DEFAULT_BATCH_WAIT_CAP_SECONDS,
    assignment_wait_cap_seconds: float = DEFAULT_ASSIGNMENT_WAIT_CAP_SECONDS,
    use_parsed_cache: bool = True,
    parsed_cache_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Execute seeded multi-run evaluation and write report artifacts.
    """
    _print_perf_hints()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        module_dir = Path(__file__).resolve().parent
        default_candidates = [
            Path("data/bpi-chall.xes"),  # when called from repo root
            module_dir.parents[1] / "data" / "bpi-chall.xes",  # robust script-relative fallback
        ]
        existing = next((p for p in default_candidates if p.exists()), None)
        if existing is None:
            raise FileNotFoundError(
                "Could not find default event log. Tried: "
                + ", ".join(str(p) for p in default_candidates)
            )
        resolved_log_path = existing
    else:
        provided = Path(log_path)
        if provided.is_absolute():
            resolved_log_path = provided
        else:
            module_dir = Path(__file__).resolve().parent
            candidates = [
                provided,  # relative to current working directory
                module_dir / provided,  # relative to this file
                module_dir.parents[1] / provided,  # relative to repo root
            ]
            resolved = next((p for p in candidates if p.exists()), None)
            resolved_log_path = resolved if resolved is not None else provided

    evaluator = FullScaleEvaluator(
        log_path=str(resolved_log_path),
        use_parsed_cache=use_parsed_cache,
        parsed_cache_path=parsed_cache_path,
    )
    seeds = list(range(num_seeds))

    raw_results = evaluator.run_seeded_study(
        num_cases=num_cases,
        seeds=seeds,
        sla_threshold_seconds=sla_threshold_seconds,
        batch_wait_cap_seconds=batch_wait_cap_seconds,
        assignment_wait_cap_seconds=assignment_wait_cap_seconds,
    )
    summary = aggregate_strategy_results(raw_results)
    ranking = compute_custom_metric_ranking(summary)

    raw_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "aggregated_summary.csv"
    ranking_path = out_dir / "metric_ranking.csv"
    appendix_path = out_dir / "custom_metrics_appendix.md"
    plots_dir = out_dir / "plots"

    raw_results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    save_metric_plots(summary, plots_dir)
    write_custom_metric_appendix(appendix_path)

    print("\n--- ARTIFACTS ---")
    print(f"Raw results: {raw_path}")
    print(f"Aggregated summary: {summary_path}")
    print(f"Metric ranking: {ranking_path}")
    print(f"Plots directory: {plots_dir}")
    print(f"Appendix: {appendix_path}")

    return {
        "raw_results": str(raw_path),
        "aggregated_summary": str(summary_path),
        "metric_ranking": str(ranking_path),
        "plots_dir": str(plots_dir),
        "appendix": str(appendix_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run seeded strategy evaluation for resource-allocation metrics."
    )
    parser.add_argument("--log-path", default=None, help="Path to XES/CSV event log.")
    parser.add_argument(
        "--output-dir",
        default="optimization_artifacts",
        help="Directory for CSV/plot/markdown outputs.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=DEFAULT_NUM_CASES,
        help="Number of cases to evaluate.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Number of random seeds (0..num_seeds-1).",
    )
    parser.add_argument(
        "--sla-threshold-seconds",
        type=int,
        default=3600,
        help="SLA threshold in seconds.",
    )
    parser.add_argument(
        "--batch-wait-cap-seconds",
        type=float,
        default=DEFAULT_BATCH_WAIT_CAP_SECONDS,
        help="Per-task wait cap used only for Batch strategy.",
    )
    parser.add_argument(
        "--assignment-wait-cap-seconds",
        type=float,
        default=DEFAULT_ASSIGNMENT_WAIT_CAP_SECONDS,
        help="Per-task wait cap used only for Assignment Problem strategy.",
    )
    parser.add_argument(
        "--parsed-cache-path",
        default=None,
        help="Optional parquet cache path for parsed XES logs.",
    )
    parser.add_argument(
        "--disable-log-cache",
        action="store_true",
        help="Disable XES parsed-log cache usage.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_full_evaluation(
        log_path=args.log_path,
        output_dir=args.output_dir,
        num_cases=args.num_cases,
        num_seeds=args.num_seeds,
        sla_threshold_seconds=args.sla_threshold_seconds,
        batch_wait_cap_seconds=args.batch_wait_cap_seconds,
        assignment_wait_cap_seconds=args.assignment_wait_cap_seconds,
        use_parsed_cache=not args.disable_log_cache,
        parsed_cache_path=args.parsed_cache_path,
    )
