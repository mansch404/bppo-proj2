import pm4py
import random
import copy
import numpy as np
import pandas as pd
from simulation.resource_manager.metrics import compute_all_metrics
from pathlib import Path
from datetime import datetime
from simulation.spawner.dynamic_spawner import DynamicSpawner_KDE
from simulation.engine.engine import SimulationEngine
from simulation.resource_manager.resource_manager import AdvancedResourceManager
from simulation.resource_manager.resource_manager import (
    AdvancedOptimizationPlanner,
    AdvancedResourceManager,
    AssignmentProblemPlanner,
    BatchPlanner,
    ShortestQueuePlanner,
    CaseHandlingPlanner,
    RandomPlanner,
    RoundRobinPlanner,
)
'''
This script generates simulation-logs using all different 
resource allocation methods.
'''

# 1. DEFINE EVALUATION SETTINGS
SETTINGS = {
    "sim_start": datetime(2016, 1, 1, 8, 0, 0),
    "sim_end": datetime(2016, 2, 1, 8, 0, 0),  # 3 months to ensure we get >2500 cases
    "base_random_seed": 42,
    "warm_up_cases": 500,
    "evaluation_cases": 2000,
    "runs_per_method": 1  # Start small for testing
}

# Map strings to the actual class implementations you built
STRATEGIES = {
    "Basic_ShortestQueue": ShortestQueuePlanner,
    "Batch_Allocation": BatchPlanner,
    "Advanced_Optimizer": AdvancedOptimizationPlanner,
    "Advanced_AssignmentProblem": AssignmentProblemPlanner
}

# Store mined organizational data for the metrics function
ORG_CONTEXT = {
    "capacities": {},
    "system_resources": set(),
    "daily_work_seconds": {},
    "automation_eligible_activities": set(),
    "sla_threshold_seconds": 3600,
}

eval_log_dir_name = "eval_logs_test" # Change for creating new folder

def run_sims_for_evaluations():

    script_dir = Path(__file__).parent.parent
    bpmn_path = str(script_dir / "process_model.bpmn")
    training_data_path = str(script_dir.parent / "data" / "bpi-chall.xes")

    # Load Process Model once
    bpmn_graph = pm4py.read_bpmn(str(bpmn_path))
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_graph)

    # Pre-train organizational model to save time in the loop
    log = pm4py.read_xes(training_data_path)
    df = pm4py.convert_to_dataframe(log)

    # 1. Initialize the Spawner ONCE outside the loops
    global_spawner = DynamicSpawner_KDE()
    global_spawner.fit_with_event_log_path(training_data_path)

    for run_index in range(SETTINGS["runs_per_method"]):

        current_seed = SETTINGS["base_random_seed"] + run_index
        random.seed(current_seed)
        np.random.seed(current_seed)

        print(f"\n--- Starting Evaluation Run {run_index} ---")

        # 2. Generate the "Ground Truth" arrivals for this specific run
        # This list will be shared across all methods to ensure fairness
        global_arrivals = global_spawner.generate_arrivals(
            SETTINGS["sim_start"],
            SETTINGS["sim_end"]
        )

        for method_name, PlannerClass in STRATEGIES.items():

            # 1. Initialize the specific resource allocation strategy for this run
            chosen_strategy = PlannerClass()
            resource_manager = AdvancedResourceManager(
                simulation_start_time=SETTINGS["sim_start"],
                strategy=chosen_strategy
            )

            # Set retry intervals just like in main.py
            if isinstance(chosen_strategy, AssignmentProblemPlanner):
                resource_manager.retry_interval = 60
            elif isinstance(chosen_strategy, BatchPlanner):
                resource_manager.retry_interval = 120

            resource_manager.mine_organizational_model(df)

            # Save capabilities for metrics.py (Only needs to happen once since the training df is the same)
            if not ORG_CONTEXT["capacities"]:
                ORG_CONTEXT["capacities"] = resource_manager.daily_effort_capacities
                ORG_CONTEXT["system_resources"] = resource_manager.system_resources
                ORG_CONTEXT["automation_eligible_activities"] = {
                    act for act, role_ids in resource_manager.activity_permissions.items()
                    if -1 in role_ids
                }

            # 2 Setup distinct output log name
            output_log_name = f"{eval_log_dir_name}/{method_name}_run_{run_index}.csv"

            # Initialize Engine
            # evaluation behavior is handled in this script via clean-and-aggregate steps,
            # not through a dedicated SimulationEngine evaluation flag.
            engine = SimulationEngine(
                net=net,
                initial_marking=initial_marking,
                final_marking=final_marking,
                branching_mode="basic",
                event_log_path=output_log_name,
                simulation_start_datetime=SETTINGS["sim_start"],
                simulation_end_datetime=SETTINGS["sim_end"],
                use_advanced_model=True,
                resource_manager=resource_manager,
                original_log_path=training_data_path,
                spawner_advanced=True
            )

            # 3. OVERWRITE the internally generated arrivals with global list
            # We use deepcopy so if the engine modifies the list, it doesn't break the next method
            engine.list_of_arrivals = copy.deepcopy(global_arrivals)

            # 4. Run the simulation
            from simulation.main import arrival_generator
            # The arrival_generator will now sort and use the injected global list
            engine.env.process(arrival_generator(engine, SETTINGS["sim_start"]))
            duration_seconds = (SETTINGS["sim_end"] - SETTINGS["sim_start"]).total_seconds()

            print(f"  -> Running {method_name} with {len(engine.list_of_arrivals)} shared arrivals...")
            engine.run(until=duration_seconds)

            # Capture per-run daily_work_seconds for stress ratio metric
            ORG_CONTEXT["daily_work_seconds"] = {
                res: dict(usage)
                for res, usage in resource_manager.daily_work_seconds.items()
            }

            # Pass the in-memory metric_records directly to the cleaner
            clean_and_save_records(engine.metric_records, method_name, run_index)


def clean_and_save_records(records_list, method_name, run_index):
    """Converts engine records to DataFrame, truncates warm-up, and saves."""
    try:
        if not records_list:
            print(f"    WARNING: No metric records captured for {method_name}!")
            return

        df = pd.DataFrame(records_list)

        # Group by case to find the exact arrival time of each case
        case_starts = df.groupby('case')['arrival_seconds'].min().reset_index()
        case_starts = case_starts.sort_values(by='arrival_seconds')

        warm_up = SETTINGS["warm_up_cases"]
        eval_cases = SETTINGS["evaluation_cases"]

        if len(case_starts) < (warm_up + eval_cases):
            print(
                f"    WARNING: Not enough cases generated! Only got {len(case_starts)}. Metrics will be skewed. Increase sim_end.")
            # We still extract what we have so it doesn't crash
            valid_case_ids = case_starts.iloc[warm_up:]['case']
        else:
            valid_case_ids = case_starts.iloc[warm_up: warm_up + eval_cases]['case']

        # Filter the DataFrame to ONLY include the valid steady-state cases
        final_df = df[df['case'].isin(valid_case_ids)]

        clean_path = Path(f"{eval_log_dir_name}/{method_name}_run_{run_index}_CLEAN.csv")
        final_df.to_csv(clean_path, index=False)
        print(f"    -> Saved {len(valid_case_ids)} steady-state cases to {clean_path.name}")

    except Exception as e:
        print(f"    Error cleaning records: {e}")


def compute_and_aggregate_results():
    print("\n" + "=" * 50)
    print("CALCULATING FINAL OPTIMIZATION METRICS (ALL 11)")
    print("=" * 50)

    final_report_data = []

    for method_name in STRATEGIES.keys():
        run_metrics = []

        for run_index in range(SETTINGS["runs_per_method"]):
            clean_log_path = Path(f"{eval_log_dir_name}/{method_name}_run_{run_index}_CLEAN.csv")
            if not clean_log_path.exists():
                print(f"  Missing file: {clean_log_path.name}")
                continue

            df = pd.read_csv(clean_log_path)

            if df.empty:
                print(f"  Skipping {clean_log_path.name} (File is empty)")
                continue

            run_result = compute_all_metrics(
                df,
                capacities=ORG_CONTEXT["capacities"],
                sla_threshold_seconds=ORG_CONTEXT["sla_threshold_seconds"],
                daily_work_seconds=ORG_CONTEXT.get("daily_work_seconds", {}),
                automation_eligible_activities=ORG_CONTEXT.get("automation_eligible_activities", set()),
            )
            run_metrics.append(run_result)

        if run_metrics:
            avg_metrics = {"Method": method_name}
            for key in run_metrics[0].keys():
                avg_metrics[key] = np.nanmean([m[key] for m in run_metrics])
            final_report_data.append(avg_metrics)

    if final_report_data:
        report_df = pd.DataFrame(final_report_data)
        print(report_df.to_string(index=False))
        report_df.to_csv(f"{eval_log_dir_name}/FINAL_REPORT_METRICS.csv", index=False)
        print(f"\nSaved Final Report to {eval_log_dir_name}/FINAL_REPORT_METRICS.csv")
    else:
        print("\nFAILED: No data could be processed.")


if __name__ == "__main__":
    # Ensure the eval_logs directory exists before running
    Path(f"{eval_log_dir_name}").mkdir(exist_ok=True)
    run_sims_for_evaluations()
    compute_and_aggregate_results()



