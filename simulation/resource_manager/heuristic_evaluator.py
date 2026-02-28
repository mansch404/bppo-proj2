import pandas as pd
import numpy as np
import pm4py
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from resource_manager import AdvancedResourceManager


class FullScaleEvaluator:
    """
    Independent Evaluation Suite. 
    Maintains scientific integrity by ensuring identical starting conditions 
    for every heuristic and optimization policy.
    """

    def __init__(self, log_path: str):
        # Load data once to save memory/time
        if log_path.endswith('.xes'):
            self.raw_df = pm4py.convert_to_dataframe(pm4py.read_xes(log_path))
        else:
            self.raw_df = pd.read_csv(log_path)

        self.start_time = datetime(2024, 1, 1, 8, 0)
        # Pre-sort for chronological simulation
        self.tasks = self.raw_df.sort_values(by='time:timestamp').to_dict('records')

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
            activity = task['concept:name']
            case_id = task['case:concept:name']
            base_duration = task.get('duration', 1800)  # Task 1.5/1.6 duration
            amount = task.get('case:RequestedAmount', 0)

            allocated_res = None
            wait_time = 0

            # 3. Request logic (Wait-until-available loop)
            # This implements the 'Queue' logic without needing a complex SimPy setup
            while allocated_res is None:
                allocated_res = manager.request_resource(
                    activity,
                    sim_clock_seconds + wait_time,
                    base_duration,
                    case_id=case_id,
                    amount=amount
                )

                if allocated_res is None:
                    wait_time += 300  # Wait 5 minutes before trying again

                if wait_time > 86400 * 7:  # Safety break (7 days)
                    allocated_res = "TIMEOUT_FAIL"

            # 4. Record the result
            execution_log.append({
                'case': case_id,
                'activity': activity,
                'resource': allocated_res,
                'wait_seconds': wait_time,
                'is_system': allocated_res in manager.system_resources,
                'is_senior': (manager.senior_role_id is not None and
                              allocated_res in manager.roles.get(manager.senior_role_id, []))
            })

            # Clock moves forward as tasks arrive, but we don't block for task duration
            # because tasks can run in parallel.
            sim_clock_seconds += 5  # Now cases arrive every 5 seconds.

        return self._calculate_metrics(execution_log, strategy_name)

    def _calculate_metrics(self, log, name):
        """Task 3: Evaluate on at least three metrics."""
        df_res = pd.DataFrame(log)

        metrics = {
            "Strategy": name,
            "Mean Wait (min)": df_res['wait_seconds'].mean() / 60,
            "System Usage %": df_res['is_system'].mean() * 100,
            "Senior Resource Load": df_res[df_res['is_senior'] == True].shape[0],
            "Max Bottleneck (hrs)": df_res['wait_seconds'].max() / 3600,
            "Service Level %": (df_res['wait_seconds'] < 3600).mean() * 100  # % under 1 hour wait
        }
        return metrics


from resource_manager import *


def run_simulation_study():
    # 1. Initialize Evaluator with your event log
    evaluator = FullScaleEvaluator("../../data/bpi-chall.xes")

    # 2. Define the strategies to test
    strategies = [
        (RandomPlanner(), "Basic: Random"),
        (RoundRobinPlanner(), "Heuristic: Round-Robin"),
        (CaseHandlingPlanner(), "Pattern: Case Handling"),
        (AdvancedOptimizationPlanner(), "Advanced: OR-Optimized")
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
    print(results_df.sort_values(by="Mean Wait (min)"))

    # 5. Visualization (Task 3)
    results_df.set_index("Strategy")["Mean Wait (min)"].plot(kind='bar', title='Efficiency Comparison')


if __name__ == "__main__":
    run_simulation_study()