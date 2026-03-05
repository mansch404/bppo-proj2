import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_metrics_module():
    metrics_path = Path(__file__).with_name("metrics.py")
    spec = importlib.util.spec_from_file_location("resource_metrics", metrics_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


metrics = _load_metrics_module()


def test_case_cycle_mean_and_p90():
    df = pd.DataFrame(
        [
            {"case": "c1", "arrival_seconds": 0.0, "end_seconds": 600.0, "timed_out": False},
            {"case": "c1", "arrival_seconds": 60.0, "end_seconds": 900.0, "timed_out": False},
            {"case": "c2", "arrival_seconds": 120.0, "end_seconds": 480.0, "timed_out": False},
        ]
    )

    out = metrics.compute_case_cycle_metrics(df)

    assert out["mean"] == pytest.approx(10.5)
    assert out["p90"] == pytest.approx(14.1)


def test_resource_occupation_humans_and_all():
    df = pd.DataFrame(
        [
            {
                "resource": "Alice",
                "start_seconds": 0.0,
                "end_seconds": 100.0,
                "service_seconds": 100.0,
                "is_system": False,
                "timed_out": False,
            },
            {
                "resource": "Bob",
                "start_seconds": 0.0,
                "end_seconds": 50.0,
                "service_seconds": 50.0,
                "is_system": False,
                "timed_out": False,
            },
            {
                "resource": "User_1",
                "start_seconds": 0.0,
                "end_seconds": 100.0,
                "service_seconds": 100.0,
                "is_system": True,
                "timed_out": False,
            },
        ]
    )

    humans = metrics.compute_resource_occupation(df, include_system=False)
    all_resources = metrics.compute_resource_occupation(df, include_system=True)

    assert humans == pytest.approx(75.0)
    assert all_resources == pytest.approx(83.3333333333)


def test_weighted_jain_equals_one_on_capacity_normalized_balance():
    df = pd.DataFrame(
        [
            {
                "resource": "Alice",
                "service_seconds": 100.0,
                "start_seconds": 0.0,
                "end_seconds": 100.0,
                "is_system": False,
                "timed_out": False,
            },
            {
                "resource": "Bob",
                "service_seconds": 200.0,
                "start_seconds": 0.0,
                "end_seconds": 200.0,
                "is_system": False,
                "timed_out": False,
            },
        ]
    )
    capacities = {"Alice": 100.0, "Bob": 200.0}

    fairness = metrics.compute_weighted_jain_fairness(df, capacities)

    assert fairness == pytest.approx(1.0)


def test_weighted_jain_drops_with_imbalance():
    df = pd.DataFrame(
        [
            {
                "resource": "Alice",
                "service_seconds": 200.0,
                "start_seconds": 0.0,
                "end_seconds": 200.0,
                "is_system": False,
                "timed_out": False,
            },
            {
                "resource": "Bob",
                "service_seconds": 100.0,
                "start_seconds": 0.0,
                "end_seconds": 100.0,
                "is_system": False,
                "timed_out": False,
            },
        ]
    )
    capacities = {"Alice": 100.0, "Bob": 200.0}

    fairness = metrics.compute_weighted_jain_fairness(df, capacities)

    assert fairness < 1.0
    assert fairness == pytest.approx(6.25 / 8.5)


def test_wait_metrics_with_sla_boundary():
    df = pd.DataFrame(
        [
            {"wait_seconds": 0.0},
            {"wait_seconds": 3600.0},
            {"wait_seconds": 3601.0},
        ]
    )

    out = metrics.compute_wait_metrics(df, sla_threshold_seconds=3600)

    assert out["avg_wait_min"] == pytest.approx((7201.0 / 3.0) / 60.0)
    assert out["service_level_pct"] == pytest.approx((2.0 / 3.0) * 100.0)


def test_timeout_rows_excluded_from_cycle_occupation_and_fairness():
    df = pd.DataFrame(
        [
            {
                "case": "c1",
                "resource": "Alice",
                "arrival_seconds": 0.0,
                "start_seconds": 0.0,
                "end_seconds": 100.0,
                "service_seconds": 100.0,
                "wait_seconds": 0.0,
                "is_system": False,
                "timed_out": False,
            },
            {
                "case": "c_timeout",
                "resource": "TIMEOUT_FAIL",
                "arrival_seconds": 5.0,
                "start_seconds": 605000.0,
                "end_seconds": None,
                "service_seconds": None,
                "wait_seconds": 605000.0,
                "is_system": False,
                "timed_out": True,
            },
        ]
    )
    capacities = {"Alice": 100.0}

    cycle = metrics.compute_case_cycle_metrics(df)
    occupation = metrics.compute_resource_occupation(df, include_system=False)
    fairness = metrics.compute_weighted_jain_fairness(df, capacities)

    assert cycle["mean"] == pytest.approx(100.0 / 60.0)
    assert cycle["p90"] == pytest.approx(100.0 / 60.0)
    assert occupation == pytest.approx(100.0)
    assert fairness == pytest.approx(1.0)
