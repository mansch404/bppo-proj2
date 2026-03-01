"""
Main Execution Script
Run the business process simulation using Petri Net
"""

import pm4py
from simulation.engine.engine import SimulationEngine
from resource_manager import AdvancedResourceManager
from pathlib import Path
from datetime import datetime, timedelta

from simulation.resource_manager.resource_manager import (
    AdvancedResourceManager,
    AdvancedOptimizationPlanner
)


def main():
    """Run simulation with Petri Net process model"""

    # 1. Setup Paths
   # resource_manager = AdvancedResourceManager()
    script_dir = Path(__file__).parent

    bpmn_path = str(script_dir / "process_model.bpmn")
    training_data_path = str(script_dir.parent / "data" / "bpi-chall.xes")

    # 2. Load Process Model (Structure)
    bpmn_graph = pm4py.read_bpmn(str(bpmn_path))
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_graph)

    print(f"Petri Net loaded: {len(net.transitions)} transitions")

    # 3. Initialize & Train Resource Manager (The Brain)
    # set the start time for the NEW simulation
    sim_start = datetime(2016, 1, 1, 8, 0, 0)
    sim_end = datetime(2016, 4, 1, 8, 0, 0) # Used for arrival generation in spawner class

    chosen_strategy = AdvancedOptimizationPlanner()
    resource_manager = AdvancedResourceManager(simulation_start_time=sim_start, strategy=chosen_strategy)

    log = pm4py.read_xes(training_data_path)
    df = pm4py.convert_to_dataframe(log)

    print("Trainiere Organizational Model (K-Means & Heatmaps)...")
    resource_manager.mine_organizational_model(df)



    # 4. Initialize Simulation Engine (The Machine)
    engine = SimulationEngine(
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        branching_mode="basic", # "none" | "basic" | "advanced"
        event_log_path="simulation_log_advancedSpawner_1.csv", # This is where we WRITE new data
        simulation_start_datetime=sim_start,
        simulation_end_datetime=sim_end,
        use_advanced_model=True,
        resource_manager=resource_manager,
        original_log_path=training_data_path,
        spawner_advanced=True # True if advanced spawner is used, else static spawner
    )

    # 5. Register the Spawner Process
    # CHANGE: iteration is managed by arrival_generator()
    engine.env.process(arrival_generator(engine, sim_start))

    # 6. Run Simulation
    # Calculate exact duration in seconds
    duration_seconds = (sim_end - sim_start).total_seconds()

    print("Running simulation...")
    engine.run(until=duration_seconds)

# CHANGE
def arrival_generator(engine, sim_start):
    """
    This process yields control to the environment to simulate waiting
    between case arrivals.
    """
    print("Spawning process started...")

    # Sort arrivals
    sorted_arrivals = sorted(engine.list_of_arrivals)

    for next_arrival in sorted_arrivals:
        # Calculate when this arrival happens relative to simulation start
        arrival_offset = (next_arrival.replace(tzinfo=None) - sim_start.replace(tzinfo=None)).total_seconds()

        # Determine how long to wait from the CURRENT simulation time (env.now)
        # env.now is usually 0 at the start
        wait_duration = arrival_offset - engine.env.now

        if wait_duration > 0:
            # This 'yield' tells the engine: "Pause this function for X seconds"
            yield engine.env.timeout(wait_duration)

        # Spawn the instance
        engine.spawn_instance()


if __name__ == "__main__":
    main()