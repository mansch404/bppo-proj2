import pm4py
import random
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from simulation.spawner.dynamic_spawner import DynamicSpawner_KDE
from simulation.engine.engine import SimulationEngine
from simulation.resource_manager.metrics import compute_all_metrics
from simulation.resource_manager.resource_manager import (
    AdvancedResourceManager,
    AdvancedOptimizationPlanner,
    AssignmentProblemPlanner,
    BatchPlanner,
    ShortestQueuePlanner,
    RandomPlanner,
    DeepRLPlanner 
)

# 1. DEFINE EVALUATION SETTINGS
SETTINGS = {
    "sim_start": datetime(2016, 1, 1, 8, 0, 0),
    "sim_end": datetime(2016, 2, 1, 8, 0, 0),  # 1 Month
    "base_random_seed": 42,
    "warm_up_cases": 200,
    "evaluation_cases": 1000,
    "runs_per_method": 15
}

METHODS = [
    "Basic_ShortestQueue",
    "Batch_Allocation",
    "Advanced_Optimizer",
    "Advanced_AssignmentProblem",
    "Advanced_DeepRL" 
]

ORG_CONTEXT = {
    "capacities": {},
    "system_resources": set(),
    "daily_work_seconds": {},
    "automation_eligible_activities": set(),
    "sla_threshold_seconds": 3600,
}

eval_log_dir_name = "eval_logs_resource_allocation_15runs_1month"  # Change this to your desired output directory

def run_sims_for_evaluations():
    script_dir = Path(__file__).parent.parent
    bpmn_path = str(script_dir / "process_model.bpmn")
    training_data_path = str(script_dir.parent / "data" / "bpi-chall.xes")

    bpmn_graph = pm4py.read_bpmn(str(bpmn_path))
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_graph)

    log = pm4py.read_xes(training_data_path)
    df = pm4py.convert_to_dataframe(log)

    # Pre-mine once OUTSIDE the loop to extract the resources for DeepRL
    print("Pre-training base organizational model...")
    dummy_manager = AdvancedResourceManager(SETTINGS["sim_start"], strategy=RandomPlanner())
    dummy_manager.mine_organizational_model(df)
    
    # Extract all unique resources across all roles for DeepRL model initialization
    all_resources = []
    for role_resources in dummy_manager.roles.values():
        all_resources.extend(role_resources)
    all_resources = sorted(list(set(all_resources)))

    ORG_CONTEXT["capacities"] = dummy_manager.daily_effort_capacities
    ORG_CONTEXT["system_resources"] = dummy_manager.system_resources
    ORG_CONTEXT["automation_eligible_activities"] = {
        act for act, role_ids in dummy_manager.activity_permissions.items()
        if -1 in role_ids
    }

    global_spawner = DynamicSpawner_KDE()
    global_spawner.fit_with_event_log_path(training_data_path)

    for run_index in range(SETTINGS["runs_per_method"]):
        current_seed = SETTINGS["base_random_seed"] + run_index
        random.seed(current_seed)
        np.random.seed(current_seed)

        print(f"\n--- Starting Evaluation Run {run_index} ---")

        global_arrivals = global_spawner.generate_arrivals(
            SETTINGS["sim_start"],
            SETTINGS["sim_end"]
        )

        for method_name in METHODS:
            
            # Instantiate the specific resource allocation strategy
            if method_name == "Basic_ShortestQueue":
                chosen_strategy = ShortestQueuePlanner()
            elif method_name == "Batch_Allocation":
                chosen_strategy = BatchPlanner()
            elif method_name == "Advanced_Optimizer":
                chosen_strategy = AdvancedOptimizationPlanner()
            elif method_name == "Advanced_AssignmentProblem":
                chosen_strategy = AssignmentProblemPlanner()
            elif method_name == "Advanced_DeepRL":
                model_file = str(Path(__file__).parent.parent / "rl_model_best.pt")
                if not Path(model_file).exists():
                    model_file = "rl_model_best.pt"
                chosen_strategy = DeepRLPlanner(
                    all_resource_names=all_resources, 
                    model_path=model_file, 
                    is_training=False
                )

            resource_manager = AdvancedResourceManager(
                simulation_start_time=SETTINGS["sim_start"],
                strategy=chosen_strategy
            )

            if isinstance(chosen_strategy, AssignmentProblemPlanner):
                resource_manager.retry_interval = 60
            elif isinstance(chosen_strategy, BatchPlanner):
                resource_manager.retry_interval = 120

            # Mine the model for this specific run instance
            resource_manager.mine_organizational_model(df)

            output_log_name = f"{eval_log_dir_name}/{method_name}_run_{run_index}.csv"

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
                spawner_advanced=True,
                evaluation_flag=True,
            )

            engine.list_of_arrivals = copy.deepcopy(global_arrivals)

            from simulation.main import arrival_generator
            engine.env.process(arrival_generator(engine, SETTINGS["sim_start"]))
            duration_seconds = (SETTINGS["sim_end"] - SETTINGS["sim_start"]).total_seconds()

            print(f"  -> Running {method_name} with {len(engine.list_of_arrivals)} shared arrivals...")
            engine.run(until=duration_seconds)

            ORG_CONTEXT["daily_work_seconds"] = {
                res: dict(usage)
                for res, usage in resource_manager.daily_work_seconds.items()
            }

            clean_and_save_records(engine.metric_records, method_name, run_index)

def clean_and_save_records(records_list, method_name, run_index):
    try:
        if not records_list:
            print(f"    WARNING: No metric records captured for {method_name}!")
            return

        df = pd.DataFrame(records_list)
        case_starts = df.groupby('case')['arrival_seconds'].min().reset_index()
        case_starts = case_starts.sort_values(by='arrival_seconds')

        warm_up = SETTINGS["warm_up_cases"]
        eval_cases = SETTINGS["evaluation_cases"]

        if len(case_starts) < (warm_up + eval_cases):
            print(f"    WARNING: Not enough cases generated! Only got {len(case_starts)}. Metrics will be skewed.")
            valid_case_ids = case_starts.iloc[warm_up:]['case']
        else:
            valid_case_ids = case_starts.iloc[warm_up: warm_up + eval_cases]['case']

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

    for method_name in METHODS:
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
    Path(f"{eval_log_dir_name}").mkdir(exist_ok=True)
    run_sims_for_evaluations()
    compute_and_aggregate_results()