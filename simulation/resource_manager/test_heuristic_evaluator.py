"""Smoke tests for seeded evaluator and artifact generation."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.optimize import linear_sum_assignment

from simulation.resource_manager.heuristic_evaluator import (
    CUSTOM_METRICS,
    FullScaleEvaluator,
    run_full_evaluation,
)
from simulation.resource_manager.resource_manager import (
    AssignmentProblemPlanner,
    BatchPlanner,
    RandomPlanner,
)


def _write_smoke_log(csv_path: Path, *, num_cases: int = 2) -> None:
    rows = []
    for idx in range(1, num_cases + 1):
        amount = 1000 if idx % 2 else 20000
        rows.append(
            {
                "case:concept:name": f"c{idx}",
                "concept:name": "A",
                "org:resource": "User_1",
                "case:RequestedAmount": amount,
                "time:timestamp": f"2024-01-01T08:{(2*idx-2)%60:02d}:00Z",
                "duration": 60.0 + (idx % 3) * 10.0,
            }
        )
        rows.append(
            {
                "case:concept:name": f"c{idx}",
                "concept:name": "B",
                "org:resource": "Alice" if idx % 2 else "Bob",
                "case:RequestedAmount": amount,
                "time:timestamp": f"2024-01-01T08:{(2*idx-1)%60:02d}:00Z",
                "duration": 90.0 + (idx % 4) * 10.0,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def test_seeded_study_is_reproducible_and_contains_custom_metrics(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path)

    evaluator = FullScaleEvaluator(str(log_path))
    run_a = evaluator.run_seeded_study(num_cases=2, seeds=[0, 1], sla_threshold_seconds=3600)
    run_b = evaluator.run_seeded_study(num_cases=2, seeds=[0, 1], sla_threshold_seconds=3600)

    assert "Seed" in run_a.columns
    assert "Strategy" in run_a.columns
    assert "Timeout Rate (%)" in run_a.columns
    for metric in CUSTOM_METRICS:
        assert metric in run_a.columns

    assert_frame_equal(
        run_a.sort_values(["Seed", "Strategy"]).reset_index(drop=True),
        run_b.sort_values(["Seed", "Strategy"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_run_full_evaluation_writes_expected_artifacts(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path)
    out_dir = tmp_path / "artifacts"

    outputs = run_full_evaluation(
        log_path=str(log_path),
        output_dir=str(out_dir),
        num_cases=2,
        num_seeds=2,
        sla_threshold_seconds=3600,
    )

    raw_path = Path(outputs["raw_results"])
    summary_path = Path(outputs["aggregated_summary"])
    ranking_path = Path(outputs["metric_ranking"])
    appendix_path = Path(outputs["appendix"])
    plots_dir = Path(outputs["plots_dir"])

    assert raw_path.exists()
    assert summary_path.exists()
    assert ranking_path.exists()
    assert appendix_path.exists()
    assert plots_dir.exists()
    assert any(plots_dir.glob("*.png"))

    raw_df = pd.read_csv(raw_path)
    for required in ("Seed", "Strategy", "Timeout Rate (%)", "Value-Weighted Wait (min)"):
        assert required in raw_df.columns


def test_batch_wait_cap_triggers_timeout_without_affecting_non_batch(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path)
    evaluator = FullScaleEvaluator(str(log_path))

    tasks = evaluator._build_tasks(num_cases=1)
    assert tasks, "expected at least one task for timeout smoke test"

    batch_out = evaluator.run_experiment(
        BatchPlanner(k=5),
        "Batch",
        tasks=tasks,
        seed=0,
        batch_wait_cap_seconds=600.0,
        progress_every=1000,
    )
    random_out = evaluator.run_experiment(
        RandomPlanner(),
        "Random",
        tasks=tasks,
        seed=0,
        batch_wait_cap_seconds=600.0,
        progress_every=1000,
    )

    assert batch_out["Timeout Rate (%)"] > 0.0
    assert random_out["Timeout Rate (%)"] == 0.0


def test_batch_timeout_eviction_clears_pending_state(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path)
    evaluator = FullScaleEvaluator(str(log_path))
    tasks = evaluator._build_tasks(num_cases=1)

    planner = BatchPlanner(k=5)
    metrics = evaluator.run_experiment(
        planner,
        "Batch",
        tasks=tasks,
        seed=0,
        batch_wait_cap_seconds=600.0,
        progress_every=1000,
    )

    assert metrics["Timeout Rate (%)"] > 0.0
    assert planner.pending_tasks == {}
    assert planner.batch_assignments == {}
    assert planner._task_insertion_order == {}


def test_non_batch_results_are_invariant_to_batch_wait_cap(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path, num_cases=5)
    evaluator = FullScaleEvaluator(str(log_path))
    tasks = evaluator._build_tasks(num_cases=5)

    out_small_cap = evaluator.run_experiment(
        RandomPlanner(),
        "Random",
        tasks=tasks,
        seed=7,
        batch_wait_cap_seconds=600.0,
        progress_every=1000,
    )
    out_large_cap = evaluator.run_experiment(
        RandomPlanner(),
        "Random",
        tasks=tasks,
        seed=7,
        batch_wait_cap_seconds=7200.0,
        progress_every=1000,
    )

    assert out_small_cap == out_large_cap


def test_batch_run_is_deterministic_with_fixed_seed(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path, num_cases=8)
    evaluator = FullScaleEvaluator(str(log_path))
    tasks = evaluator._build_tasks(num_cases=8)

    out_a = evaluator.run_experiment(
        BatchPlanner(k=5),
        "Batch",
        tasks=tasks,
        seed=3,
        batch_wait_cap_seconds=1200.0,
        progress_every=1000,
    )
    out_b = evaluator.run_experiment(
        BatchPlanner(k=5),
        "Batch",
        tasks=tasks,
        seed=3,
        batch_wait_cap_seconds=1200.0,
        progress_every=1000,
    )

    assert out_a == out_b


def test_batch_heavy_timeout_smoke_completes_and_cleans_state(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path, num_cases=40)
    evaluator = FullScaleEvaluator(str(log_path))
    tasks = evaluator._build_tasks(num_cases=40)
    planner = BatchPlanner(k=500)

    t0 = time.perf_counter()
    metrics = evaluator.run_experiment(
        planner,
        "Batch",
        tasks=tasks,
        seed=11,
        batch_wait_cap_seconds=600.0,
        progress_every=1000,
    )
    elapsed = time.perf_counter() - t0

    assert metrics["Timeout Rate (%)"] > 0.0
    assert planner.pending_tasks == {}
    assert planner.batch_assignments == {}
    assert elapsed < 30.0


def test_assignment_timeout_eviction_clears_pending_state(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path, num_cases=6)
    evaluator = FullScaleEvaluator(str(log_path))
    tasks = evaluator._build_tasks(num_cases=6)
    planner = AssignmentProblemPlanner(delta=0.1)

    metrics = evaluator.run_experiment(
        planner,
        "Advanced: Assignment Problem",
        tasks=tasks,
        seed=13,
        assignment_wait_cap_seconds=600.0,
        progress_every=1000,
    )

    assert metrics["Timeout Rate (%)"] > 0.0
    assert planner.pending_tasks == {}


def test_non_batch_results_are_invariant_to_assignment_wait_cap(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path, num_cases=5)
    evaluator = FullScaleEvaluator(str(log_path))
    tasks = evaluator._build_tasks(num_cases=5)

    out_small_cap = evaluator.run_experiment(
        RandomPlanner(),
        "Random",
        tasks=tasks,
        seed=17,
        assignment_wait_cap_seconds=600.0,
        progress_every=1000,
    )
    out_large_cap = evaluator.run_experiment(
        RandomPlanner(),
        "Random",
        tasks=tasks,
        seed=17,
        assignment_wait_cap_seconds=7200.0,
        progress_every=1000,
    )

    assert out_small_cap == out_large_cap


def test_assignment_run_is_deterministic_with_fixed_seed(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path, num_cases=8)
    evaluator = FullScaleEvaluator(str(log_path))
    tasks = evaluator._build_tasks(num_cases=8)

    out_a = evaluator.run_experiment(
        AssignmentProblemPlanner(delta=1.2),
        "Advanced: Assignment Problem",
        tasks=tasks,
        seed=5,
        assignment_wait_cap_seconds=3600.0,
        progress_every=1000,
    )
    out_b = evaluator.run_experiment(
        AssignmentProblemPlanner(delta=1.2),
        "Advanced: Assignment Problem",
        tasks=tasks,
        seed=5,
        assignment_wait_cap_seconds=3600.0,
        progress_every=1000,
    )

    assert out_a == out_b


def test_assignment_heavy_timeout_smoke_completes_and_cleans_state(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path, num_cases=40)
    evaluator = FullScaleEvaluator(str(log_path))
    tasks = evaluator._build_tasks(num_cases=40)
    planner = AssignmentProblemPlanner(delta=0.1)

    t0 = time.perf_counter()
    metrics = evaluator.run_experiment(
        planner,
        "Advanced: Assignment Problem",
        tasks=tasks,
        seed=19,
        assignment_wait_cap_seconds=600.0,
        progress_every=1000,
    )
    elapsed = time.perf_counter() - t0

    assert metrics["Timeout Rate (%)"] > 0.0
    assert planner.pending_tasks == {}
    assert elapsed < 30.0


def test_rectangular_hungarian_matches_padded_square_assignments():
    # Unique costs avoid tie-related alternative optima.
    rectangular = np.array(
        [
            [4.0, 40.0, 60.0, 80.0, 90.0],
            [50.0, 3.0, 70.0, 85.0, 95.0],
            [55.0, 65.0, 2.0, 75.0, 100.0],
        ],
        dtype=float,
    )
    row_r, col_r = linear_sum_assignment(rectangular)
    row_to_col_rect = dict(zip(row_r.tolist(), col_r.tolist()))

    size = max(rectangular.shape)
    BIG = 1e9
    padded = np.full((size, size), BIG, dtype=float)
    padded[: rectangular.shape[0], : rectangular.shape[1]] = rectangular
    row_p, col_p = linear_sum_assignment(padded)
    row_to_col_padded = {
        r: c
        for r, c in zip(row_p.tolist(), col_p.tolist())
        if r < rectangular.shape[0]
    }

    assert row_to_col_rect == row_to_col_padded


def test_optimized_template_path_matches_legacy_remine_output(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path, num_cases=20)
    evaluator = FullScaleEvaluator(str(log_path), use_parsed_cache=False)

    optimized = evaluator.run_seeded_study(
        num_cases=20,
        seeds=[0, 1],
        sla_threshold_seconds=3600,
        batch_wait_cap_seconds=3600.0,
        reuse_mined_model=True,
    ).sort_values(["Seed", "Strategy"]).reset_index(drop=True)
    legacy = evaluator.run_seeded_study(
        num_cases=20,
        seeds=[0, 1],
        sla_threshold_seconds=3600,
        batch_wait_cap_seconds=3600.0,
        reuse_mined_model=False,
    ).sort_values(["Seed", "Strategy"]).reset_index(drop=True)

    assert_frame_equal(optimized, legacy, check_dtype=False, atol=1e-12, rtol=0.0)


def test_optimized_template_path_is_faster_than_legacy_remine(tmp_path: Path):
    log_path = tmp_path / "smoke_log.csv"
    _write_smoke_log(log_path, num_cases=100)
    evaluator = FullScaleEvaluator(str(log_path), use_parsed_cache=False)

    t0 = time.perf_counter()
    evaluator.run_seeded_study(
        num_cases=80,
        seeds=[0],
        sla_threshold_seconds=3600,
        batch_wait_cap_seconds=600.0,
        reuse_mined_model=False,
    )
    legacy_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    evaluator.run_seeded_study(
        num_cases=80,
        seeds=[0],
        sla_threshold_seconds=3600,
        batch_wait_cap_seconds=600.0,
        reuse_mined_model=True,
    )
    optimized_seconds = time.perf_counter() - t0

    assert optimized_seconds < legacy_seconds
