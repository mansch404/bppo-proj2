"""
Main Execution Script
Run the business process simulation using Petri Net
"""

import pm4py
from engine import SimulationEngine
from resource_manager import AdvancedResourceManager
from pathlib import Path
from datetime import datetime, timedelta

def main():
    """Run simulation with Petri Net process model"""

    # 1. Setup Paths
    script_dir = Path(__file__).parent

    bpmn_path = str(script_dir / "process_model.bpmn")
    training_data_path = str(script_dir.parent / "data" / "BPI Challenge 2017.xes")

    # 2. Load Process Model (Structure)
    bpmn_graph = pm4py.read_bpmn(str(bpmn_path))
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_graph)

    print(f"Petri Net loaded: {len(net.transitions)} transitions")

    # 3. Initialize & Train Resource Manager (The Brain)
    # set the start time for the NEW simulation
    sim_start = datetime(2017, 1, 1, 8, 0, 0)
    sim_end = datetime(2017, 6, 1, 8, 0, 0) # Used for arrival generation in spawner class

    resource_manager = AdvancedResourceManager(simulation_start_time=sim_start)

    print(f"Training Resource Manager on {training_data_path}...")
    try:
        resource_manager.load_log_and_mine_profiles(training_data_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not train on BPI data: {e}")
        print("Please ensure 'BPI Challenge 2017.xes' is in the project folder.")
        return


    # 4. Initialize Simulation Engine (The Machine)
    engine = SimulationEngine(
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        branching_mode="advanced", # "none" | "basic" | "advanced"
        event_log_path="simulation_log.csv", # This is where we WRITE new data
        simulation_start_datetime=sim_start,
        simulation_end_datetime=sim_end,
        use_advanced_model=True,
        resource_manager=resource_manager,
        original_log_path=training_data_path,
        spawner_advanced=False # True if advanced spawner is used, else static spawner
    )

    # 5. Run Simulation
    print("Spawning cases...")
    for next_arrival in engine.list_of_arrivals: # <--- Changes spawner
        delay_seconds = (next_arrival - sim_start).total_seconds()
        engine.spawn_instance()

        # If the arrival is before the start time, skip or spawn immediately
        if delay_seconds < 0:
            continue

        # Calculate how long to wait from NOW until that arrival
        # env.now is the current simulation time in seconds
        wait_duration = delay_seconds - engine.env.now
        if wait_duration > 0:
            yield engine.env.timeout(wait_duration)

        # Spawn instance after delay
        engine.spawn_instance()

    # Calculate exact duration in seconds
    duration_seconds = (sim_end - sim_start).total_seconds()

    print("Running simulation...")
    engine.run(until=duration_seconds)


if __name__ == "__main__":
    main()
