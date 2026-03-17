import pm4py
import random
import copy
import pickle
import json
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
from datetime import datetime

from simulation.resource_manager.metrics import get_default_registry
from simulation.spawner.dynamic_spawner import DynamicSpawner_KDE
from simulation.engine.engine import SimulationEngine
from simulation.routing.branching_basic import BranchingBasic
from simulation.resource_manager.resource_manager import (
    AdvancedResourceManager,
    ShortestQueuePlanner,
    BatchPlanner,
    AdvancedOptimizationPlanner,
    AssignmentProblemPlanner,
    RandomPlanner,
    DeepRLPlanner 
)

# 1. DEFINE EVALUATION SETTINGS
eval_log_dir_name = "eval_logs_resource_allocation_15runs_1month"  # Change this to your desired output directory

SETTINGS = {
    "sim_start": datetime(2016, 1, 1, 8, 0, 0),
    "sim_end": datetime(2016, 2, 1, 8, 0, 0),
    "base_random_seed": 42,
    "warm_up_cases": 200,
    "evaluation_cases": 1000,
    "runs_per_method": 15
}

# Changed from a dictionary of classes to a simple list of names
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
    "automation_eligible_activities": set()
}

# 2. THE WORKER FUNCTION (Top-Level)
def run_single_simulation(args):
    """This function runs entirely independently on a separate CPU core."""
    (run_index, method_name, global_arrivals, net, initial_marking, final_marking, 
     temp_df_path, branching_path, all_resources) = args

    current_seed = SETTINGS["base_random_seed"] + run_index
    random.seed(current_seed)
    np.random.seed(current_seed)

    # Load the pre-parsed binary DataFrame (Super fast, low memory!)
    df = pd.read_pickle(temp_df_path)

    # 1. Instantiate the correct strategy INSIDE the worker
    if method_name == "Basic_ShortestQueue":
        chosen_strategy = ShortestQueuePlanner()
    elif method_name == "Batch_Allocation":
        chosen_strategy = BatchPlanner()
    elif method_name == "Advanced_Optimizer":
        chosen_strategy = AdvancedOptimizationPlanner()
    elif method_name == "Advanced_AssignmentProblem":
        chosen_strategy = AssignmentProblemPlanner()
    elif method_name == "Advanced_DeepRL":
        # Path to the trained model file
        model_file = str(Path(__file__).parent.parent / "resource_manager" / "rl_model_best.pt")
        if not Path(model_file).exists():
            model_file = "rl_model_best.pt"  # Fallback to current working directory
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
        resource_manager.retry_interval = 300
    elif isinstance(chosen_strategy, BatchPlanner):
        resource_manager.retry_interval = 300

    resource_manager.mine_organizational_model(df)

    output_log_name = f"{eval_log_dir_name}/{method_name}_run_{run_index + 1}_RAW.csv"

    engine = SimulationEngine(
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        branching_mode="basic",
        branching_model_path=branching_path,
        original_log_path=None,
        event_log_path=output_log_name,
        simulation_start_datetime=SETTINGS["sim_start"],
        simulation_end_datetime=SETTINGS["sim_end"],
        use_advanced_model=True,
        resource_manager=resource_manager,
        spawner_advanced=False,
        evaluation_flag=True
    )

    engine.list_of_arrivals = copy.deepcopy(global_arrivals)

    from simulation.main import arrival_generator
    engine.env.process(arrival_generator(engine, SETTINGS["sim_start"]))
    duration_seconds = (SETTINGS["sim_end"] - SETTINGS["sim_start"]).total_seconds()

    print(f"  -> [Core Active] Running {method_name} Run {run_index + 1}...")
    engine.run(until=duration_seconds)

    clean_and_save_records(engine.metric_records, method_name, run_index + 1)

    # Checkpoint worker data to disk immediately
    daily_work_seconds_clean = {
        res: dict(usage) for res, usage in resource_manager.daily_work_seconds.items()
    }
    
    json_path = Path(eval_log_dir_name) / f"{method_name}_run_{run_index + 1}_work.json"
    with open(json_path, "w") as f:
        json.dump(daily_work_seconds_clean, f)

    return method_name, run_index + 1

def clean_and_save_records(records_list, method_name, run_idx):
    try:
        if not records_list:
            return

        df = pd.DataFrame(records_list)
        case_starts = df.groupby('case')['arrival_seconds'].min().reset_index()
        case_starts = case_starts.sort_values(by='arrival_seconds')

        warm_up = SETTINGS["warm_up_cases"]
        eval_cases = SETTINGS["evaluation_cases"]

        if len(case_starts) < (warm_up + eval_cases):
            valid_case_ids = case_starts.iloc[warm_up:]['case']
        else:
            valid_case_ids = case_starts.iloc[warm_up: warm_up + eval_cases]['case']

        final_df = df[df['case'].isin(valid_case_ids)]
        clean_path = Path(f"{eval_log_dir_name}/{method_name}_run_{run_idx}_CLEAN.csv")
        final_df.to_csv(clean_path, index=False)

    except Exception as e:
        print(f"    Error cleaning records: {e}")

# 3. THE MULTIPROCESSING ORCHESTRATOR
def run_sims_for_evaluations():
    script_dir = Path(__file__).parent.parent
    bpmn_path = str(script_dir / "process_model.bpmn")
    training_data_path = str(script_dir.parent / "data" / "bpi-chall.xes")

    bpmn_graph = pm4py.read_bpmn(str(bpmn_path))
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_graph)
    
    log = pm4py.read_xes(training_data_path)
    df_main = pm4py.convert_to_dataframe(log)

    print("Mining base organizational model for capacities and permissions...")
    dummy_manager = AdvancedResourceManager(SETTINGS["sim_start"], strategy=RandomPlanner())
    dummy_manager.mine_organizational_model(df_main)
    ORG_CONTEXT["capacities"] = dummy_manager.daily_effort_capacities
    ORG_CONTEXT["system_resources"] = dummy_manager.system_resources

    ORG_CONTEXT["automation_eligible_activities"] = {
        act for act, role_ids in dummy_manager.activity_permissions.items()
        if -1 in role_ids
    }

    # Extract all unique resources for the DeepRL Planner
    all_resources = []
    for role_resources in dummy_manager.roles.values():
        all_resources.extend(role_resources)
    all_resources = sorted(list(set(all_resources)))

    temp_df_path = str(Path(eval_log_dir_name) / "temp_parsed_log.pkl")
    df_main.to_pickle(temp_df_path)

    del df_main
    del log

    print("Pre-fitting Branching Model to prevent redundant parsing in workers...")
    branching_model = BranchingBasic().fit_from_event_log(training_data_path)
    branching_path_obj = Path(eval_log_dir_name) / "pre_fitted_branching.pkl"
    branching_path_obj.parent.mkdir(parents=True, exist_ok=True)
    branching_path = str(branching_path_obj)

    with open(branching_path, "wb") as f:
        pickle.dump(branching_model, f)

    global_spawner = DynamicSpawner_KDE()
    global_spawner.fit_with_event_log_path(training_data_path)

    tasks = []
    for run_index in range(SETTINGS["runs_per_method"]):
        current_seed = SETTINGS["base_random_seed"] + run_index
        random.seed(current_seed)
        np.random.seed(current_seed)

        global_arrivals = global_spawner.generate_arrivals(SETTINGS["sim_start"], SETTINGS["sim_end"])

        for method_name in METHODS:
            run_idx = run_index + 1
            expected_clean_log = Path(f"{eval_log_dir_name}/{method_name}_run_{run_idx}_CLEAN.csv")
            expected_json = Path(f"{eval_log_dir_name}/{method_name}_run_{run_idx}_work.json")

            if expected_clean_log.exists() and expected_json.exists():
                print(f"⏩ Skipping {method_name} Run {run_idx} (Already completed)")
                continue

            # Pass all_resources so the worker can initialize the Neural Network
            task_args = (
                run_index, method_name, global_arrivals, net, initial_marking, 
                final_marking, temp_df_path, branching_path, all_resources
            )
            tasks.append(task_args)

    if tasks:
        print(f"\n{'=' * 50}\nLaunching {len(tasks)} parallel simulations!\n{'=' * 50}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor: # Change max_workers based on your CPU cores
            futures = [executor.submit(run_single_simulation, arg) for arg in tasks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    method_name, run_idx = future.result()
                    print(f"✅ Completed {method_name} - Run {run_idx}")
                except Exception as e:
                    print(f"❌ A simulation crashed: {e}")
                    
        Path(temp_df_path).unlink(missing_ok=True)
    else:
        print("\n✅ All simulations already completed. Proceeding to aggregation...")
        Path(temp_df_path).unlink(missing_ok=True)

# 4. METRICS AGGREGATION
def compute_and_aggregate_results():
    print("\n" + "=" * 50)
    print("CALCULATING FINAL OPTIMIZATION METRICS (ALL 11)")
    print("=" * 50)

    final_report_data = []
    registry = get_default_registry()

    for method_name in METHODS:
        run_metrics = []

        for run_index in range(SETTINGS["runs_per_method"]):
            run_idx = run_index + 1
            clean_log_path = Path(f"{eval_log_dir_name}/{method_name}_run_{run_idx}_CLEAN.csv")
            json_path = Path(f"{eval_log_dir_name}/{method_name}_run_{run_idx}_work.json")
            
            if not clean_log_path.exists():
                continue

            df_clean = pd.read_csv(clean_log_path)
            if df_clean.empty:
                continue

            if json_path.exists():
                with open(json_path, "r") as f:
                    daily_work = json.load(f)
            else:
                daily_work = {}

            run_result = registry.compute_all(
                df_clean,
                capacities=ORG_CONTEXT["capacities"],
                sla_threshold_seconds=3600,
                daily_work_seconds=daily_work,
                automation_eligible_activities=ORG_CONTEXT["automation_eligible_activities"]
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

if __name__ == "__main__":
    Path(eval_log_dir_name).mkdir(exist_ok=True)
    run_sims_for_evaluations()
    compute_and_aggregate_results()