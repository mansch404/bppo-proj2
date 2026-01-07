"""
Main Execution Script
Run the business process simulation using Petri Net
"""

import pm4py
from engine import SimulationEngine
from resource_manager import AdvancedResourceManager
from pathlib import Path
from datetime import datetime

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
        use_advanced_model=True,
        resource_manager=resource_manager
    )

    # 5. Run Simulation
    print("Spawning cases...")
    for i in range(50):
        engine.spawn_instance()

    print("Running simulation...")
    engine.run(until=1200000)

if __name__ == "__main__":
    main()
