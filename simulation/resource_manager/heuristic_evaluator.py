import pandas as pd
import numpy as np
import pm4py
from datetime import datetime
import matplotlib.pyplot as plt

try:
    from .metrics import compute_optimization_metrics
    from .resource_manager import (
        AdvancedOptimizationPlanner,
        AdvancedResourceManager,
        CaseHandlingPlanner,
        RandomPlanner,
        RoundRobinPlanner,
    )
except ImportError:
    from metrics import compute_optimization_metrics
    from resource_manager import (
        AdvancedOptimizationPlanner,
        AdvancedResourceManager,
        CaseHandlingPlanner,
        RandomPlanner,
        RoundRobinPlanner,
    )


class FullScaleEvaluator:
    """
    Independent Evaluation Suite. 
    Maintains scientific integrity by ensuring identical starting conditions 
    for every heuristic and optimization policy.
    """

    def __init__(self, log_path: str):
        # Load data once to save memory/time
        if log_path.endswith(".xes"):
            self.raw_df = pm4py.convert_to_dataframe(pm4py.read_xes(log_path))
        else:
            self.raw_df = pd.read_csv(log_path)

        self.start_time = datetime(2024, 1, 1, 8, 0)
        # Pre-sort for chronological simulation
        self.tasks = self.raw_df.sort_values(by="time:timestamp").to_dict("records")

    def run_experiment(self, planner_instance, strategy_name: str):
        """
        The Simulation Loop: Orchestrates the interaction between 
        the Log, the Manager, and the Environment.
        """
        # 1. Setup the Manager with a fresh state
        manager = AdvancedResourceManager(self.start_time, planner_instance)
        manager.mine_organizational_model(self.raw_df)

        sim_clock_seconds = 0
        execution_log = []

        # 2. Iterate through events (The Simulation)
        for task in self.tasks:
            activity = task["concept:name"]
            case_id = task["case:concept:name"]
            base_duration = task.get("duration", 1800)  # Task 1.5/1.6 duration
            amount = task.get("case:RequestedAmount", 0)
            arrival_seconds = sim_clock_seconds

            allocated_res = None
            wait_time = 0
            timed_out = False

            # 3. Request logic (Wait-until-available loop)
            # This implements the 'Queue' logic without needing a complex SimPy setup
            while allocated_res is None:
                allocated_res = manager.request_resource(
                    activity,
                    arrival_seconds + wait_time,
                    base_duration,
                    case_id=case_id,
                    amount=amount,
                )

                if allocated_res is None:
                    wait_time += 300  # Wait 5 minutes before trying again

                if wait_time > 86400 * 7:  # Safety break (7 days)
                    allocated_res = "TIMEOUT_FAIL"
                    timed_out = True

            # 4. Record the result
            start_seconds = float(arrival_seconds + wait_time)
            if timed_out or allocated_res == "TIMEOUT_FAIL":
                end_seconds = np.nan
                service_seconds = np.nan
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

            execution_log.append({
                "case": case_id,
                "activity": activity,
                "resource": allocated_res,
                "wait_seconds": float(wait_time),
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
            })

            # Clock moves forward as tasks arrive, but we don't block for task duration
            # because tasks can run in parallel.
            sim_clock_seconds += 5  # Now cases arrive every 5 seconds.

        return self._calculate_metrics(execution_log, strategy_name, manager)

    def _calculate_metrics(self, log, name, manager):
        """Task 3: Evaluate strategies using six optimization metrics."""
        df_res = pd.DataFrame(log)
        capacities = dict(getattr(manager, "daily_effort_capacities", {}))

        metrics = {"Strategy": name}
        metrics.update(compute_optimization_metrics(df_res, capacities, sla_threshold_seconds=3600))
        return metrics


def run_simulation_study():
    # 1. Initialize Evaluator with your event log
    evaluator = FullScaleEvaluator("../../data/bpi-chall.xes")

    # 2. Define the strategies to test
    strategies = [
        (RandomPlanner(), "Basic: Random"),
        (RoundRobinPlanner(), "Heuristic: Round-Robin"),
        (CaseHandlingPlanner(), "Pattern: Case Handling"),
        (AdvancedOptimizationPlanner(), "Advanced: OR-Optimized"),
    ]

    # 3. Run Experiments and Collect Results
    all_results = []
    for planner, name in strategies:
        print(f"Testing {name}...")
        metrics = evaluator.run_experiment(planner, name)
        all_results.append(metrics)

    # 4. Compare and Argument
    results_df = pd.DataFrame(all_results)
    print("\n--- PERFORMANCE COMPARISON ---")
    print(results_df.sort_values(by="Avg Wait Time (min)"))

    # 5. Visualization (Task 3)
    results_df.set_index("Strategy")["Avg Case Cycle Time (min)"].plot(
        kind="bar",
        title="Avg Case Cycle Time Comparison",
    )


if __name__ == "__main__":
    run_simulation_study()
