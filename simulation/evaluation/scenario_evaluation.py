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
from simulation.resource_manager.resource_manager import AdvancedResourceManager, AdvancedOptimizationPlanner, \
    RandomPlanner
from simulation.routing.branching_basic import BranchingBasic


# CONFIGURATION

OUTPUT_DIR = "eval_logs_scenario"

SETTINGS = {
    "sim_start": datetime(2016, 1, 1, 8, 0, 0),
    "sim_end": datetime(2016, 2, 1, 8, 0, 0),  # 1 Month duration
    "base_random_seed": 42,
    "warm_up_cases": 200,  # Adjusted for 1 month
    "evaluation_cases": 1000,  # Adjusted for 1 month
    "runs_per_scenario": 15  # Number of parallel runs
}

ORG_CONTEXT = {
    "capacities": {},
    "system_resources": set(),
    "automation_eligible_activities": set(),
    "daily_work_seconds": {}  # Stores the worker data for the metrics
}

# --- UPDATE THESE NAMES BASED ON YOUR HISTORICAL SCRIPT ---
SCENARIOS = {
    "Baseline": [],
    "Reduced_Staff": ["User_111", "User_110"]  # <--- Change to the two you are firing
}

# FUNCTIONS

def fire_employees(resource_manager, employees_to_fire):
    """Surgically removes specific employees from the mined organizational model."""
    for emp in employees_to_fire:
        if emp in resource_manager.daily_effort_capacities:
            del resource_manager.daily_effort_capacities[emp]
        for role_id, members in resource_manager.roles.items():
            if emp in members:
                members.remove(emp)
        if emp in resource_manager.availability_matrix:
            del resource_manager.availability_matrix[emp]
        if emp in resource_manager.competence_scores:
            del resource_manager.competence_scores[emp]


def run_scenario_simulation(args):
    """Worker function for parallel processing."""
    (run_index, scenario_name, fired_list, global_arrivals,
     net, initial_marking, final_marking, df, branching_path) = args

    current_seed = SETTINGS["base_random_seed"] + run_index
    random.seed(current_seed)
    np.random.seed(current_seed)

    resource_manager = AdvancedResourceManager(
        simulation_start_time=SETTINGS["sim_start"],
        strategy=AdvancedOptimizationPlanner()
    )

    resource_manager.mine_organizational_model(df)

    if fired_list:
        fire_employees(resource_manager, fired_list)

    output_log_name = f"{OUTPUT_DIR}/Scenario_{scenario_name}_run_{run_index + 1}_RAW.csv"

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
        spawner_advanced=False,  # <--- Fix applied to bypass KDE in workers!
        evaluation_flag=True
    )

    engine.list_of_arrivals = copy.deepcopy(global_arrivals)

    from simulation.main import arrival_generator
    engine.env.process(arrival_generator(engine, SETTINGS["sim_start"]))
    duration_seconds = (SETTINGS["sim_end"] - SETTINGS["sim_start"]).total_seconds()

    engine.run(until=duration_seconds)

    clean_and_save_records(engine.metric_records, f"Scenario_{scenario_name}", run_index + 1)

    # --- Capture the daily work usage for the advanced metrics ---
    daily_work_seconds_clean = {
        res: dict(usage) for res, usage in resource_manager.daily_work_seconds.items()
    }

    return scenario_name, run_index + 1, daily_work_seconds_clean


def clean_and_save_records(records_list, method_name, run_index):
    try:
        if not records_list: return
        df = pd.DataFrame(records_list)
        case_starts = df.groupby('case')['arrival_seconds'].min().reset_index().sort_values(by='arrival_seconds')
        warm_up, eval_cases = SETTINGS["warm_up_cases"], SETTINGS["evaluation_cases"]
        valid_case_ids = case_starts.iloc[warm_up: warm_up + eval_cases]['case'] if len(case_starts) >= (
                    warm_up + eval_cases) else case_starts.iloc[warm_up:]['case']

        df[df['case'].isin(valid_case_ids)].to_csv(Path(f"{OUTPUT_DIR}/{method_name}_run_{run_index}_CLEAN.csv"),
                                                   index=False)
    except Exception as e:
        print(f"Error cleaning records: {e}")


def main():
    script_dir = Path(__file__).parent.parent
    bpmn_path = str(script_dir / "process_model.bpmn")
    training_data_path = str(script_dir.parent / "data" / "bpi-chall.xes")

    bpmn_graph = pm4py.read_bpmn(str(bpmn_path))
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_graph)
    log = pm4py.read_xes(training_data_path)
    df = pm4py.convert_to_dataframe(log)

    # 1. Mine base capacities
    dummy_manager = AdvancedResourceManager(SETTINGS["sim_start"], strategy=RandomPlanner())
    dummy_manager.mine_organizational_model(df)
    ORG_CONTEXT["capacities"] = dummy_manager.daily_effort_capacities
    ORG_CONTEXT["system_resources"] = dummy_manager.system_resources

    # Grab automation permissions for the metrics module
    ORG_CONTEXT["automation_eligible_activities"] = {
        act for act, role_ids in dummy_manager.activity_permissions.items() if -1 in role_ids
    }

    # 2. Pre-fit branching model
    print("Pre-fitting components...")
    branching_model = BranchingBasic().fit_from_event_log(training_data_path)
    branching_path_obj = Path(OUTPUT_DIR) / "pre_fitted_branching.pkl"
    branching_path_obj.parent.mkdir(parents=True, exist_ok=True)
    branching_path = str(branching_path_obj)
    with open(branching_path, "wb") as f:
        pickle.dump(branching_model, f)

    global_spawner = DynamicSpawner_KDE()
    global_spawner.fit_with_event_log_path(training_data_path)

    # 3. Multiprocessing execution
    tasks = []
    for run_index in range(SETTINGS["runs_per_scenario"]):
        current_seed = SETTINGS["base_random_seed"] + run_index
        random.seed(current_seed)
        np.random.seed(current_seed)
        global_arrivals = global_spawner.generate_arrivals(SETTINGS["sim_start"], SETTINGS["sim_end"])

        for scenario_name, fired_list in SCENARIOS.items():
            tasks.append(
                (run_index, scenario_name, fired_list, global_arrivals, net, initial_marking, final_marking, df,
                 branching_path))

    print(f"\nLaunching {len(tasks)} parallel scenario simulations...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_scenario_simulation, arg) for arg in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                scenario_name, run_idx, daily_work = future.result()
                # Store the worker's data so the metrics registry can use it
                ORG_CONTEXT["daily_work_seconds"][f"{scenario_name}_{run_idx}"] = daily_work
                print(f"✅ Completed {scenario_name} - Run {run_idx}")
            except Exception as e:
                print(f"❌ Error: {e}")

    # 4. Metrics Aggregation
    print("\n" + "=" * 50 + "\nSCENARIO IMPACT REPORT\n" + "=" * 50)
    final_report_data = []
    registry = get_default_registry()

    for scenario_name in SCENARIOS.keys():
        run_metrics = []
        for run_index in range(SETTINGS["runs_per_scenario"]):
            run_idx = run_index + 1
            clean_log_path = Path(f"{OUTPUT_DIR}/Scenario_{scenario_name}_run_{run_idx}_CLEAN.csv")
            if clean_log_path.exists() and not (df_clean := pd.read_csv(clean_log_path)).empty:
                # Fetch specific run history
                daily_work = ORG_CONTEXT["daily_work_seconds"].get(f"{scenario_name}_{run_idx}", {})

                run_metrics.append(registry.compute_all(
                    df=df_clean,
                    capacities=ORG_CONTEXT["capacities"],
                    sla_threshold_seconds=3600,
                    daily_work_seconds=daily_work,
                    automation_eligible_activities=ORG_CONTEXT["automation_eligible_activities"]
                ))

        if run_metrics:
            avg_metrics = {"Scenario": scenario_name}
            for key in run_metrics[0].keys():
                avg_metrics[key] = np.nanmean([m[key] for m in run_metrics])
            final_report_data.append(avg_metrics)

    if final_report_data:
        report_df = pd.DataFrame(final_report_data)
        # Score calculation
        if "Service Level <=60min (%)" in report_df.columns and "Weighted Fairness (Jain, humans)" in report_df.columns:
            report_df["System Efficiency Score"] = (
                    (report_df["Service Level <=60min (%)"] / 100) * report_df[
                "Weighted Fairness (Jain, humans)"] * 10000 /
                    (report_df["Avg Case Cycle Time (min)"] + report_df["Avg Wait Time (min)"])
            )

        print(report_df.to_string(index=False))
        report_df.to_csv(f"{OUTPUT_DIR}/SCENARIO_IMPACT_METRICS.csv", index=False)


if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    main()