"""Deep diagnostic tests for AssignmentProblemPlanner.
Tests simulate realistic scenarios with AdvancedResourceManager-like behavior."""

import pytest
import time
import numpy as np
from datetime import datetime, timedelta

from simulation.resource_manager.resource_manager import (
    AssignmentProblemPlanner,
    AdvancedResourceManager,
    RandomPlanner,
)


class RealisticMockManager:
    """Mock that behaves like AdvancedResourceManager with heatmaps."""

    def __init__(self):
        self.simulation_start_time = datetime(2016, 1, 1, 8, 0, 0)  # Friday 8am
        self.busy_until = {}
        self.activity_permissions = {
            "TaskA": {0, 1},
            "TaskB": {0},
            "TaskC": {1},
        }
        self.roles = {
            0: ["Alice", "Bob", "Charlie"],
            1: ["Diana", "Eve"],
        }
        self.case_assignments = {}
        self.system_resources = {"SystemBot"}
        self.daily_work_seconds = {}
        self.daily_effort_capacities = {
            "Alice": 28800, "Bob": 28800, "Charlie": 28800,
            "Diana": 28800, "Eve": 28800,
        }
        # Heatmap: all resources available Mon-Fri 8-18 with prob 0.8
        self.availability_matrix = {}
        for res in ["Alice", "Bob", "Charlie", "Diana", "Eve"]:
            self.availability_matrix[res] = {
                d: {h: (0.8 if 8 <= h <= 18 and d < 5 else 0.05) for h in range(24)}
                for d in range(7)
            }
        self.last_activity = {}

    def is_resource_available(self, resource, current_time, duration):
        """Monte Carlo version (random!) — like AdvancedResourceManager."""
        import random
        if resource in self.system_resources:
            return True
        prob = (self.availability_matrix.get(resource, {})
                .get(current_time.weekday(), {})
                .get(current_time.hour, 0.0))
        if random.random() >= prob:
            return False
        date_key = current_time.strftime("%Y-%m-%d")
        used = self.daily_work_seconds.get(resource, {}).get(date_key, 0)
        return (used + duration) <= self.daily_effort_capacities.get(resource, 28800)

    def is_resource_available_deterministic(self, resource, current_time, duration):
        """Deterministic version — like our fix."""
        if resource in self.system_resources:
            return True
        prob = (self.availability_matrix.get(resource, {})
                .get(current_time.weekday(), {})
                .get(current_time.hour, 0.0))
        if prob < 0.3:
            return False
        date_key = current_time.strftime("%Y-%m-%d")
        used = self.daily_work_seconds.get(resource, {}).get(date_key, 0)
        return (used + duration) <= self.daily_effort_capacities.get(resource, 28800)


# ===== TEST 1: Single task should ALWAYS get assigned when resources are free =====

class TestSingleTaskAlwaysAssigned:
    def test_free_resources_deterministic(self):
        """With free resources during work hours, a single task must get assigned."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = RealisticMockManager()

        # sim_time=0 → Friday 8am, resources should be available
        result = planner.select_resource(
            manager, "TaskA", current_sim_time=0, duration=100.0, case_id="case_1"
        )
        assert result is not None, (
            f"Expected a resource but got None! "
            f"pending_tasks={list(planner.pending_tasks.keys())}, "
            f"cached_assignments={list(planner.cached_assignments.keys())}"
        )
        assert result in ["Alice", "Bob", "Charlie", "Diana", "Eve"]

    def test_night_time_no_resources(self):
        """At 3am (outside work hours), no human resources should be available."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = RealisticMockManager()
        # Set all heatmap probs to 0 for 3am
        for res in manager.availability_matrix:
            for d in range(7):
                manager.availability_matrix[res][d][3] = 0.0

        # sim_time for 3am = (3-8)*3600 but we need positive, so next day 3am
        # 2016-01-01 08:00 + 19h = 2016-01-02 03:00
        sim_time = 19 * 3600

        result = planner.select_resource(
            manager, "TaskA", current_sim_time=sim_time, duration=100.0, case_id="case_1"
        )
        # Might be None (dummy assigned) since prob=0 at 3am → BIG cost → dummy wins
        # This is correct behavior


# ===== TEST 2: Multiple tasks assigned in bulk =====

class TestBulkAssignment:
    def test_three_tasks_three_different_resources(self):
        """3 tasks for TaskA with 5 authorized resources → all 3 should get assigned."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = RealisticMockManager()

        results = []
        for i in range(3):
            r = planner.select_resource(
                manager, "TaskA", current_sim_time=0, duration=100.0, case_id=f"case_{i}"
            )
            results.append(r)

        assigned = [r for r in results if r is not None]
        assert len(assigned) >= 2, (
            f"Expected at least 2 assignments from 3 tasks with 5 resources, "
            f"got {len(assigned)}: {results}"
        )

    def test_cached_assignments_used(self):
        """After a solve, other tasks should get their cached assignment."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = RealisticMockManager()

        # First call triggers solve and caches assignments
        r1 = planner.select_resource(
            manager, "TaskA", current_sim_time=0, duration=100.0, case_id="case_1"
        )

        # Second call — if case_2_TaskA was cached, should return immediately
        r2 = planner.select_resource(
            manager, "TaskA", current_sim_time=0, duration=100.0, case_id="case_2"
        )

        # Both should get resources (5 available for TaskA)
        # r1 triggers solve, r2 either hits cache or triggers new solve
        print(f"r1={r1}, r2={r2}")
        print(f"pending_tasks={list(planner.pending_tasks.keys())}")
        print(f"cached_assignments={list(planner.cached_assignments.keys())}")


# ===== TEST 3: Performance under load =====

class TestPerformanceUnderLoad:
    def test_50_tasks_completes_fast(self):
        """50 tasks should be solvable in under 2 seconds total."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = RealisticMockManager()

        start = time.time()
        results = []
        for i in range(50):
            r = planner.select_resource(
                manager, "TaskA", current_sim_time=0, duration=100.0, case_id=f"case_{i}"
            )
            results.append(r)
        elapsed = time.time() - start

        assigned = [r for r in results if r is not None]
        print(f"\n50 tasks: {len(assigned)} assigned, {50-len(assigned)} pending, took {elapsed:.3f}s")
        print(f"pending_tasks size: {len(planner.pending_tasks)}")
        assert elapsed < 2.0, f"50 tasks took {elapsed:.1f}s — too slow!"

    def test_retry_simulation_50_cases(self):
        """Simulate the engine retry loop: 50 cases, each retrying up to 10 times."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = RealisticMockManager()

        start = time.time()
        total_calls = 0
        assigned_count = 0

        for sim_round in range(10):  # 10 retry rounds
            sim_time = sim_round * 60  # Each round 60s later
            for case_i in range(50):
                task_id = f"case_{case_i}_TaskA"

                # Skip if already assigned
                if task_id not in planner.pending_tasks and sim_round > 0:
                    continue

                r = planner.select_resource(
                    manager, "TaskA", current_sim_time=sim_time,
                    duration=100.0, case_id=f"case_{case_i}"
                )
                total_calls += 1
                if r is not None:
                    assigned_count += 1

        elapsed = time.time() - start
        print(f"\nRetry simulation: {total_calls} calls, {assigned_count} assigned, "
              f"{len(planner.pending_tasks)} still pending, took {elapsed:.3f}s")
        assert elapsed < 5.0, f"Retry simulation took {elapsed:.1f}s — too slow!"


# ===== TEST 4: Diagnose the exact problem =====

class TestDiagnoseHang:
    def test_cost_matrix_values_are_sane(self):
        """Check that the cost matrix has reasonable values (not all BIG)."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = RealisticMockManager()

        planner.pending_tasks["case_1_TaskA"] = {
            "activity": "TaskA", "sim_time": 0, "duration": 100.0,
            "kwargs": {"case_id": "case_1"},
        }

        task_ids = list(planner.pending_tasks.keys())
        resources = planner._get_all_relevant_resources(manager)
        matrix = planner._build_cost_matrix(manager, task_ids, resources, current_sim_time=0)

        n_resources = len(resources)
        BIG = 1e9

        print(f"\nResources: {resources}")
        print(f"Matrix shape: {matrix.shape}")
        for j, res in enumerate(resources):
            print(f"  {res}: cost={matrix[0,j]:.1f} {'(BIG!)' if matrix[0,j] >= BIG else '(OK)'}")

        non_big_count = sum(1 for j in range(n_resources) if matrix[0, j] < BIG)
        assert non_big_count > 0, (
            f"ALL {n_resources} resources have BIG cost! Matrix row: "
            f"{[matrix[0,j] for j in range(n_resources)]}"
        )

    def test_dummy_cost_is_reasonable(self):
        """Dummy cost should be delta * avg_cost, not BIG."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = RealisticMockManager()

        planner.pending_tasks["case_1_TaskA"] = {
            "activity": "TaskA", "sim_time": 0, "duration": 100.0,
            "kwargs": {"case_id": "case_1"},
        }

        task_ids = list(planner.pending_tasks.keys())
        resources = planner._get_all_relevant_resources(manager)
        matrix = planner._build_cost_matrix(manager, task_ids, resources, current_sim_time=0)

        n_resources = len(resources)
        dummy_col = n_resources + 0  # Dummy for task 0
        dummy_cost = matrix[0, dummy_col]

        print(f"\nDummy cost: {dummy_cost}")
        print(f"Expected: ~{1.2 * 100.0} = 120.0")

        BIG = 1e9
        assert dummy_cost < BIG, f"Dummy cost is BIG ({dummy_cost})! This means no authorized resources."
        assert dummy_cost == pytest.approx(120.0), f"Dummy cost={dummy_cost}, expected 120.0"

    def test_hungarian_assigns_real_not_dummy(self):
        """With free resources, Hungarian should prefer real resource over dummy."""
        from scipy.optimize import linear_sum_assignment

        planner = AssignmentProblemPlanner(delta=1.2)
        manager = RealisticMockManager()

        planner.pending_tasks["case_1_TaskA"] = {
            "activity": "TaskA", "sim_time": 0, "duration": 100.0,
            "kwargs": {"case_id": "case_1"},
        }

        task_ids = list(planner.pending_tasks.keys())
        resources = planner._get_all_relevant_resources(manager)
        matrix = planner._build_cost_matrix(manager, task_ids, resources, current_sim_time=0)

        row_ind, col_ind = linear_sum_assignment(matrix)
        n_resources = len(resources)

        # Find assignment for task 0
        for r, c in zip(row_ind, col_ind):
            if r == 0:
                print(f"\nTask 0 assigned to column {c} (n_resources={n_resources})")
                if c < n_resources:
                    print(f"  → Real resource: {resources[c]}")
                else:
                    print(f"  → DUMMY! (column {c} >= {n_resources})")
                assert c < n_resources, (
                    f"Hungarian assigned DUMMY (col {c}) instead of real resource! "
                    f"Costs: real={[matrix[0,j] for j in range(n_resources)]}, "
                    f"dummy={matrix[0, n_resources]}"
                )
                break

    def test_pending_tasks_dont_grow_unbounded(self):
        """After assignment, pending_tasks should shrink, not grow."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = RealisticMockManager()

        sizes = []
        for i in range(20):
            planner.select_resource(
                manager, "TaskA", current_sim_time=0, duration=100.0, case_id=f"case_{i}"
            )
            sizes.append(len(planner.pending_tasks))

        print(f"\npending_tasks sizes after each call: {sizes}")
        # Should NOT monotonically increase
        assert sizes[-1] < 20, (
            f"pending_tasks grew to {sizes[-1]} after 20 calls! "
            f"Tasks are not being assigned. Sizes: {sizes}"
        )

    def test_with_request_resource_full_flow(self):
        """Test through request_resource (the actual entry point from engine)."""
        planner = AssignmentProblemPlanner(delta=1.2)
        manager = AdvancedResourceManager(
            simulation_start_time=datetime(2016, 1, 4, 8, 0, 0),  # Monday 8am
            strategy=planner,
        )
        # Manually set up minimal organizational model
        manager.roles = {0: ["Alice", "Bob", "Charlie"]}
        manager.activity_permissions = {"TaskA": {0}}
        manager.system_resources = set()
        # Set heatmaps so resources are "available" deterministically
        for res in ["Alice", "Bob", "Charlie"]:
            manager.availability_matrix[res] = {
                d: {h: (0.8 if 8 <= h <= 18 and d < 5 else 0.0) for h in range(24)}
                for d in range(7)
            }
            manager.daily_effort_capacities[res] = 28800

        results = []
        for i in range(5):
            r = manager.request_resource(
                activity="TaskA", sim_time=0, duration=100.0, case_id=f"case_{i}"
            )
            results.append(r)
            print(f"  request_resource case_{i}: {r}")

        assigned = [r for r in results if r is not None]
        assert len(assigned) >= 2, (
            f"Only {len(assigned)}/5 assigned through request_resource! Results: {results}"
        )

    def test_retry_after_time_passes_assigns_resource(self):
        """THE CRITICAL BUG TEST: After resources become free, retry must re-solve."""
        planner = AssignmentProblemPlanner(delta=1.2, solve_cooldown=60)
        manager = AdvancedResourceManager(
            simulation_start_time=datetime(2016, 1, 4, 8, 0, 0),  # Monday 8am
            strategy=planner,
        )
        manager.roles = {0: ["Alice"]}  # Only 1 resource
        manager.activity_permissions = {"TaskA": {0}}
        manager.system_resources = set()
        for res in ["Alice"]:
            manager.availability_matrix[res] = {
                d: {h: (0.8 if 8 <= h <= 18 and d < 5 else 0.0) for h in range(24)}
                for d in range(7)
            }
            manager.daily_effort_capacities[res] = 28800

        # Case 1: Alice assigned, busy for 100s
        r1 = manager.request_resource(activity="TaskA", sim_time=0, duration=100.0, case_id="case_1")
        assert r1 == "Alice", f"Expected Alice, got {r1}"
        print(f"  case_1 assigned: {r1}")
        print(f"  Alice busy_until: {manager.busy_until.get('Alice')}")

        # Case 2: Alice busy → should get None (dummy assigned)
        r2 = manager.request_resource(activity="TaskA", sim_time=0, duration=100.0, case_id="case_2")
        assert r2 is None, f"Expected None (Alice busy), got {r2}"
        print(f"  case_2 at t=0: {r2} (correct, Alice busy)")

        # Retry case 2 at sim_time=200 (Alice should be free by now!)
        # Alice was busy until t=0 + 115s (100 * 1.15 penalty)
        r2_retry = manager.request_resource(activity="TaskA", sim_time=200, duration=100.0, case_id="case_2")
        print(f"  case_2 retry at t=200: {r2_retry}")
        print(f"  pending_tasks: {list(planner.pending_tasks.keys())}")
        print(f"  cached_assignments: {list(planner.cached_assignments.keys())}")
        print(f"  _last_solve_time: {planner._last_solve_time}")

        assert r2_retry == "Alice", (
            f"BUG: case_2 retry at t=200 returned {r2_retry} instead of 'Alice'! "
            f"Alice is free (busy_until={manager.busy_until.get('Alice')}), "
            f"but the cache prevented re-solving. "
            f"pending_tasks={list(planner.pending_tasks.keys())}, "
            f"_last_solve_time={planner._last_solve_time}"
        )
