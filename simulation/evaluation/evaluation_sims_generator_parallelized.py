import pm4py
import random
import copy
import pickle
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
from datetime import datetime

from simulation.resource_manager.metrics import get_default_registry
from simulation.spawner.dynamic_spawner import DynamicSpawner_KDE
from simulation.engine.engine import SimulationEngine
from simulation.resource_manager.resource_manager import AdvancedResourceManager
from simulation.routing.branching_basic import BranchingBasic
from simulation.resource_manager.resource_manager import (
    ShortestQueuePlanner,
    BatchPlanner,
    AdvancedOptimizationPlanner,
    AssignmentProblemPlanner,
    RandomPlanner
)

# ==========================================
# 1. DEFINE EVALUATION SETTINGS
# ==========================================
eval_log_dir_name = "eval_logs_15runs_1month"

SETTINGS = {
    "sim_start": datetime(2016, 1, 1, 8, 0, 0),
    "sim_end": datetime(2016, 2, 1, 8, 0, 0),
    "base_random_seed": 42,
    "warm_up_cases": 200,
    "evaluation_cases": 1000,
    "runs_per_method": 15
}

STRATEGIES = {
    "Basic_ShortestQueue": ShortestQueuePlanner,
    "Batch_Allocation": BatchPlanner,
    "Advanced_Optimizer": AdvancedOptimizationPlanner,
    "Advanced_AssignmentProblem": AssignmentProblemPlanner
}

ORG_CONTEXT = {
    "capacities": {},
    "system_resources": set(),
    "automation_eligible_activities": set(),
    "daily_work_seconds": {}  # Will hold the usage data mapped by method and run_index
}


# ==========================================
# 2. THE WORKER FUNCTION (Top-Level)
# ==========================================
def run_single_simulation(args):
    """This function runs entirely independently on a separate CPU core."""
    (run_index, method_name, PlannerClass, global_arrivals,
     net, initial_marking, final_marking, df, branching_path, org_context) = args

    current_seed = SETTINGS["base_random_seed"] + run_index
    random.seed(current_seed)
    np.random.seed(current_seed)

    chosen_strategy = PlannerClass()
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

    # --- CRITICAL FIX FOR MULTIPROCESSING ---
    # Extract the daily work seconds from this specific isolated worker run
    # and return it to the main orchestrator!
    daily_work_seconds_clean = {
        res: dict(usage) for res, usage in resource_manager.daily_work_seconds.items()
    }

    return method_name, run_index + 1, daily_work_seconds_clean


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


# ==========================================
# 3. THE MULTIPROCESSING ORCHESTRATOR
# ==========================================
def run_sims_for_evaluations():
    script_dir = Path(__file__).parent.parent
    bpmn_path = str(script_dir / "process_model.bpmn")
    training_data_path = str(script_dir.parent / "data" / "bpi-chall.xes")

    bpmn_graph = pm4py.read_bpmn(str(bpmn_path))
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_graph)
    log = pm4py.read_xes(training_data_path)
    df = pm4py.convert_to_dataframe(log)

    print("Mining base organizational model for capacities and permissions...")
    dummy_manager = AdvancedResourceManager(SETTINGS["sim_start"], strategy=RandomPlanner())
    dummy_manager.mine_organizational_model(df)
    ORG_CONTEXT["capacities"] = dummy_manager.daily_effort_capacities
    ORG_CONTEXT["system_resources"] = dummy_manager.system_resources

    # Extract automation eligible activities for the MetricsRegistry
    ORG_CONTEXT["automation_eligible_activities"] = {
        act for act, role_ids in dummy_manager.activity_permissions.items()
        if -1 in role_ids
    }

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

        for method_name, PlannerClass in STRATEGIES.items():
            task_args = (
                run_index, method_name, PlannerClass, global_arrivals,
                net, initial_marking, final_marking, df, branching_path, ORG_CONTEXT
            )
            tasks.append(task_args)

    print(f"\n{'=' * 50}\nLaunching {len(tasks)} parallel simulations!\n{'=' * 50}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_single_simulation, arg) for arg in tasks]

        for future in concurrent.futures.as_completed(futures):
            try:
                # Catch the packaged data returned by the worker
                method_name, run_idx, daily_work = future.result()

                # Store it in the main process's ORG_CONTEXT so the metric registry can use it
                ORG_CONTEXT["daily_work_seconds"][f"{method_name}_{run_idx}"] = daily_work
                print(f"✅ Completed {method_name} - Run {run_idx}")
            except Exception as e:
                print(f"❌ A simulation crashed: {e}")


# ==========================================
# 4. METRICS AGGREGATION
# ==========================================
def compute_and_aggregate_results():
    print("\n" + "=" * 50)
    print("CALCULATING FINAL OPTIMIZATION METRICS (ALL 11)")
    print("=" * 50)

    final_report_data = []

    # Initialize the new Metrics Registry
    registry = get_default_registry()

    for method_name in STRATEGIES.keys():
        run_metrics = []

        for run_index in range(SETTINGS["runs_per_method"]):
            run_idx = run_index + 1
            clean_log_path = Path(f"{eval_log_dir_name}/{method_name}_run_{run_idx}_CLEAN.csv")
            if not clean_log_path.exists():
                continue

            df = pd.read_csv(clean_log_path)
            if df.empty:
                continue

            # Fetch the specific daily_work_seconds for this exact run
            daily_work = ORG_CONTEXT["daily_work_seconds"].get(f"{method_name}_{run_idx}", {})

            # Compute using the registry object directly
            run_result = registry.compute_all(
                df,
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