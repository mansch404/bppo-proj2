"""Tests for BatchPlanner and AssignmentProblemPlanner."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from simulation.resource_manager.resource_manager import (
    AssignmentProblemPlanner,
    BatchPlanner,
)


class MockManager:
    """Minimal mock of AdvancedResourceManager for testing."""

    def __init__(self):
        self.simulation_start_time = datetime(2017, 1, 1, 8, 0, 0)
        self.busy_until = {}
        self.activity_permissions = {
            "TaskA": {0},  # Role 0 is authorized for TaskA
        }
        self.roles = {
            0: ["Alice", "Bob"],
        }
        self.case_assignments = {}


# --- Test 1: Instantiation ---

class TestInstantiation:
    def test_ap_planner_default_delta(self):
        planner = AssignmentProblemPlanner(delta=1.2)
        assert planner.delta == 1.2
        assert planner.pending_tasks == {}

    def test_batch_planner_default_k(self):
        planner = BatchPlanner(k=5)
        assert planner.k == 5
        assert planner.pending_tasks == {}
        assert planner.batch_assignments == {}


# --- Test 2: select_resource method exists ---

class TestInterface:
    def test_ap_planner_has_select_resource(self):
        planner = AssignmentProblemPlanner()
        assert callable(getattr(planner, "select_resource", None))

    def test_batch_planner_has_select_resource(self):
        planner = BatchPlanner()
        assert callable(getattr(planner, "select_resource", None))


# --- Test 3: Cost matrix building ---

class TestCostMatrix:
    def test_single_task_two_resources_both_free(self):
        """One pending task, two free authorized resources → matrix has correct shape and values."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = MockManager()

        # Register one pending task
        planner.pending_tasks["case_1_TaskA"] = {
            "activity": "TaskA",
            "sim_time": 0,
            "duration": 100.0,
            "kwargs": {"case_id": "case_1"},
        }

        task_ids = list(planner.pending_tasks.keys())
        resources = sorted(["Alice", "Bob"])  # ['Alice', 'Bob']

        matrix = planner._build_cost_matrix(manager, task_ids, resources, current_sim_time=0)

        n_tasks = 1
        n_resources = 2
        n_cols = n_resources + n_tasks  # 3

        # Matrix should be square: max(1, 3) = 3
        assert matrix.shape == (3, 3)

        # Both resources are free and authorized → cost = duration = 100
        assert matrix[0, 0] == 100.0  # Alice
        assert matrix[0, 1] == 100.0  # Bob

        # Dummy column for task 0 is at index n_resources + 0 = 2
        # dummy_cost = delta * avg_cost = 1.2 * 100.0 = 120.0
        assert matrix[0, 2] == pytest.approx(120.0)

    def test_single_task_one_busy_resource(self):
        """One resource busy, one free → busy resource has higher cost."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = MockManager()

        # Alice is busy for 200 more seconds
        manager.busy_until["Alice"] = manager.simulation_start_time + timedelta(seconds=200)

        planner.pending_tasks["case_1_TaskA"] = {
            "activity": "TaskA",
            "sim_time": 0,
            "duration": 100.0,
            "kwargs": {"case_id": "case_1"},
        }

        task_ids = list(planner.pending_tasks.keys())
        resources = ["Alice", "Bob"]

        matrix = planner._build_cost_matrix(manager, task_ids, resources, current_sim_time=0)

        # Alice: busy → cost = duration + remaining = 100 + 200 = 300
        assert matrix[0, 0] == pytest.approx(300.0)
        # Bob: free → cost = duration = 100
        assert matrix[0, 1] == pytest.approx(100.0)

    def test_unauthorized_resource_gets_big_cost(self):
        """A resource not authorized for the activity gets BIG cost."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = MockManager()

        # Add a third resource in a different role, not authorized for TaskA
        manager.roles[1] = ["Charlie"]

        planner.pending_tasks["case_1_TaskA"] = {
            "activity": "TaskA",
            "sim_time": 0,
            "duration": 100.0,
            "kwargs": {"case_id": "case_1"},
        }

        task_ids = list(planner.pending_tasks.keys())
        resources = ["Alice", "Bob", "Charlie"]

        matrix = planner._build_cost_matrix(manager, task_ids, resources, current_sim_time=0)

        # Charlie not authorized → BIG
        assert matrix[0, 2] == 1e9


# --- Test 4: End-to-end select_resource ---

class TestSelectResource:
    def test_ap_planner_assigns_free_resource(self):
        """With free authorized resources, AP planner should return one of them."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = MockManager()

        result = planner.select_resource(
            manager, "TaskA", current_sim_time=0, duration=100.0, case_id="case_1"
        )

        assert result in ("Alice", "Bob")
        # Task should be removed from pending after assignment
        assert "case_1_TaskA" not in planner.pending_tasks

    def test_ap_planner_returns_none_when_no_resources(self):
        """With no authorized resources, AP planner should return None."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = MockManager()
        manager.activity_permissions = {}  # No permissions at all

        result = planner.select_resource(
            manager, "TaskA", current_sim_time=0, duration=100.0, case_id="case_1"
        )

        assert result is None

    def test_batch_planner_waits_until_k(self):
        """BatchPlanner should return None until k tasks are pending."""
        planner = BatchPlanner(k=3)
        manager = MockManager()

        # First two tasks: batch not full → None
        r1 = planner.select_resource(manager, "TaskA", 0, 100.0, case_id="case_1")
        r2 = planner.select_resource(manager, "TaskA", 0, 100.0, case_id="case_2")
        assert r1 is None
        assert r2 is None
        assert len(planner.pending_tasks) == 2
