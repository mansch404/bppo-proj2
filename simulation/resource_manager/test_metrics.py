import importlib.util
from pathlib import Path

import numpy as np
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


def test_value_weighted_wait_uses_log_amount_weights():
    df = pd.DataFrame(
        [
            {"wait_seconds": 60.0, "requested_amount": 100.0},
            {"wait_seconds": 120.0, "requested_amount": 10000.0},
        ]
    )

    out = metrics.compute_value_weighted_wait(df)

    w1 = np.log1p(100.0)
    w2 = np.log1p(10000.0)
    expected_seconds = (60.0 * w1 + 120.0 * w2) / (w1 + w2)
    assert out == pytest.approx(expected_seconds / 60.0)


def test_value_at_risk_sla_breach_is_amount_share():
    df = pd.DataFrame(
        [
            {"wait_seconds": 100.0, "requested_amount": 100.0},
            {"wait_seconds": 3700.0, "requested_amount": 100.0},
            {"wait_seconds": 4000.0, "requested_amount": 300.0},
        ]
    )

    out = metrics.compute_value_at_risk_sla_breach(df, sla_threshold_seconds=3600)

    assert out == pytest.approx((400.0 / 500.0) * 100.0)


def test_case_handover_rate_counts_resource_switches_per_case():
    df = pd.DataFrame(
        [
            {"case": "c1", "resource": "A", "start_seconds": 0.0, "timed_out": False},
            {"case": "c1", "resource": "A", "start_seconds": 1.0, "timed_out": False},
            {"case": "c1", "resource": "B", "start_seconds": 2.0, "timed_out": False},
            {"case": "c2", "resource": "C", "start_seconds": 0.0, "timed_out": False},
            {"case": "c2", "resource": "D", "start_seconds": 1.0, "timed_out": False},
            {"case": "c2", "resource": "D", "start_seconds": 2.0, "timed_out": False},
        ]
    )

    out = metrics.compute_case_handover_rate(df)

    assert out == pytest.approx(0.5)


def test_automation_leverage_on_eligible_tasks():
    df = pd.DataFrame(
        [
            {"activity": "X", "is_system": True, "timed_out": False},
            {"activity": "Y", "is_system": False, "timed_out": False},
            {"activity": "X", "is_system": False, "timed_out": False},
        ]
    )

    out = metrics.compute_automation_leverage(df, automation_eligible_activities={"X"})

    assert out == pytest.approx(50.0)


def test_human_capacity_stress_ratio_overload_percentage():
    capacities = {"Alice": 100.0, "Bob": 200.0}
    daily_work_seconds = {
        "Alice": {"2026-01-01": 120.0, "2026-01-02": 90.0},
        "Bob": {"2026-01-01": 260.0},
        "User_1": {"2026-01-01": 2000.0},  # ignored (system / unknown)
    }

    out = metrics.compute_human_capacity_stress_ratio(capacities, daily_work_seconds)

    assert out == pytest.approx((80.0 / 400.0) * 100.0)


def test_value_weighted_wait_falls_back_to_unweighted_when_weights_sum_to_zero():
    df = pd.DataFrame(
        [
            {"wait_seconds": 60.0, "requested_amount": 0.0},
            {"wait_seconds": 180.0, "requested_amount": 0.0},
        ]
    )

    out = metrics.compute_value_weighted_wait(df)

    assert out == pytest.approx((120.0 / 60.0))


def test_custom_metrics_handle_missing_columns_and_no_eligible_tasks():
    df = pd.DataFrame([{"wait_seconds": 10.0}])

    out = metrics.compute_custom_optimization_metrics(
        df=df,
        capacities={},
        daily_work_seconds={},
        automation_eligible_activities=set(),
    )

    assert out["Value-Weighted Wait (min)"] == pytest.approx(10.0 / 60.0)
    assert out["Value-at-Risk SLA Breach (%)"] == pytest.approx(0.0)
    assert np.isnan(out["Case Handover Rate"])
    assert np.isnan(out["Automation Leverage on Eligible Tasks (%)"])
    assert np.isnan(out["Human Capacity Stress Ratio (%)"])


def test_custom_metrics_handle_all_timeout_rows():
    df = pd.DataFrame(
        [
            {
                "case": "c_timeout",
                "activity": "A",
                "resource": "TIMEOUT_FAIL",
                "wait_seconds": 20000.0,
                "requested_amount": 1000.0,
                "timed_out": True,
                "is_system": False,
            }
        ]
    )

    out = metrics.compute_custom_optimization_metrics(
        df=df,
        capacities={"Alice": 100.0},
        daily_work_seconds={"Alice": {"2026-01-01": 0.0}},
        automation_eligible_activities={"A"},
    )

    assert out["Value-Weighted Wait (min)"] == pytest.approx(20000.0 / 60.0)
    assert out["Value-at-Risk SLA Breach (%)"] == pytest.approx(100.0)
    assert np.isnan(out["Case Handover Rate"])
    assert np.isnan(out["Automation Leverage on Eligible Tasks (%)"])
    assert out["Human Capacity Stress Ratio (%)"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Registry API tests
# ---------------------------------------------------------------------------


def test_registry_lists_11_metrics():
    registry = metrics.get_default_registry()
    assert len(registry.list_metrics()) == 11


def test_registry_basic_group_has_6():
    registry = metrics.get_default_registry()
    assert len(registry.list_metrics(group="basic")) == 6


def test_registry_advanced_group_has_5():
    registry = metrics.get_default_registry()
    assert len(registry.list_metrics(group="advanced")) == 5


def test_compute_metric_by_name():
    df = pd.DataFrame(
        [
            {"case": "c1", "resource": "A", "start_seconds": 0.0, "timed_out": False},
            {"case": "c1", "resource": "A", "start_seconds": 1.0, "timed_out": False},
            {"case": "c1", "resource": "B", "start_seconds": 2.0, "timed_out": False},
        ]
    )
    result = metrics.compute_metric("Case Handover Rate", df)
    assert isinstance(result, float)
    assert result == pytest.approx(0.5)


def test_compute_all_returns_11_keys():
    df = pd.DataFrame(
        [
            {
                "case": "c1",
                "activity": "X",
                "resource": "Alice",
                "arrival_seconds": 0.0,
                "start_seconds": 0.0,
                "end_seconds": 100.0,
                "service_seconds": 100.0,
                "wait_seconds": 10.0,
                "requested_amount": 500.0,
                "is_system": False,
                "timed_out": False,
            }
        ]
    )
    result = metrics.compute_all_metrics(
        df,
        capacities={"Alice": 28800.0},
        sla_threshold_seconds=3600,
        daily_work_seconds={"Alice": {"2026-01-01": 100.0}},
        automation_eligible_activities={"X"},
    )
    assert len(result) == 11


def test_registry_basic_matches_legacy():
    df = pd.DataFrame(
        [
            {
                "case": "c1",
                "resource": "Alice",
                "arrival_seconds": 0.0,
                "start_seconds": 0.0,
                "end_seconds": 600.0,
                "service_seconds": 600.0,
                "wait_seconds": 30.0,
                "is_system": False,
                "timed_out": False,
            },
            {
                "case": "c2",
                "resource": "Bob",
                "arrival_seconds": 10.0,
                "start_seconds": 10.0,
                "end_seconds": 400.0,
                "service_seconds": 390.0,
                "wait_seconds": 0.0,
                "is_system": False,
                "timed_out": False,
            },
        ]
    )
    caps = {"Alice": 28800.0, "Bob": 28800.0}

    legacy = metrics.compute_optimization_metrics(df, caps, sla_threshold_seconds=3600)
    new = metrics.compute_basic_metrics(df, capacities=caps, sla_threshold_seconds=3600)

    for key in legacy:
        if np.isnan(legacy[key]):
            assert np.isnan(new[key]), f"Mismatch on {key}"
        else:
            assert new[key] == pytest.approx(legacy[key]), f"Mismatch on {key}"


def test_registry_advanced_matches_legacy():
    df = pd.DataFrame(
        [
            {
                "case": "c1",
                "activity": "X",
                "resource": "Alice",
                "start_seconds": 0.0,
                "end_seconds": 100.0,
                "service_seconds": 100.0,
                "wait_seconds": 60.0,
                "requested_amount": 100.0,
                "is_system": False,
                "timed_out": False,
            },
            {
                "case": "c1",
                "activity": "Y",
                "resource": "Bob",
                "start_seconds": 100.0,
                "end_seconds": 200.0,
                "service_seconds": 100.0,
                "wait_seconds": 120.0,
                "requested_amount": 10000.0,
                "is_system": False,
                "timed_out": False,
            },
        ]
    )
    caps = {"Alice": 28800.0, "Bob": 28800.0}
    dws = {"Alice": {"2026-01-01": 100.0}, "Bob": {"2026-01-01": 100.0}}
    eligible = {"X"}

    legacy = metrics.compute_custom_optimization_metrics(
        df, capacities=caps, daily_work_seconds=dws,
        automation_eligible_activities=eligible, sla_threshold_seconds=3600,
    )
    new = metrics.compute_advanced_metrics(
        df, capacities=caps, daily_work_seconds=dws,
        automation_eligible_activities=eligible, sla_threshold_seconds=3600,
    )

    for key in legacy:
        if np.isnan(legacy[key]):
            assert np.isnan(new[key]), f"Mismatch on {key}"
        else:
            assert new[key] == pytest.approx(legacy[key]), f"Mismatch on {key}"
