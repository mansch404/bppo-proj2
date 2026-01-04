"""
Main Execution Script
Run the business process simulation using Petri Net
"""

import pm4py
from engine import SimulationEngine
from pathlib import Path


def main():
    """Run simulation with Petri Net process model"""

    # Load BPMN and convert to Petri Net
    script_dir = Path(__file__).parent
    bpmn_path = script_dir / "process_model.bpmn"
    bpmn_graph = pm4py.read_bpmn(str(bpmn_path))

    # Convert BPMN to Petri Net
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_graph)

    print(f"Petri Net loaded:")
    print(f"  Places: {len(net.places)}")
    print(f"  Transitions: {len(net.transitions)}")
    print(f"  Initial marking: {initial_marking}")
    print(f"  Final marking: {final_marking}")

    # Create simulation engine
    engine = SimulationEngine(
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        event_log_path="simulation_log.csv",
        fitted_distributions_path="fitted_distributions.pkl",
        use_advanced_model=False,  # Set to True to use advanced processing time model (quantile regression)
    )

    # Spawn process instances
    for i in range(10):
        engine.spawn_instance()

    # Run simulation
    engine.run(until=1000)


if __name__ == "__main__":
    main()
