"""
Simulation Engine Module
Core discrete event simulation engine using Petri Net semantics
"""

import simpy
import random
from typing import Set, Dict
from pm4py.objects.petri_net.obj import PetriNet, Marking
from .logger import EventLogger
# from .resource_manager import ResourceManager


class SimulationEngine:
    def __init__(
        self,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        event_log_path: str,
    ):
        """
        Initialize simulation engine with Petri Net

        Args:
            net: Petri Net object from pm4py
            initial_marking: Initial marking (token positions)
            final_marking: Final marking (simulation end condition)
            event_log_path: Path for output CSV event log
        """
        self.env = simpy.Environment()
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.event_logger = EventLogger(event_log_path)
        # self.resource_manager = ResourceManager()
        self.case_counter = 0

        # Debug info
        print(f"\nSimulation Engine initialized:")
        print(f"  Places: {len(net.places)}")
        print(f"  Transitions: {len(net.transitions)}")
        print(f"  Initial marking: {self._marking_to_dict(initial_marking)}")
        print(f"  Final marking: {self._marking_to_dict(final_marking)}")

    def _marking_to_dict(self, marking: Marking) -> Dict:
        """Convert pm4py Marking to simple dict for comparison"""
        return {place.name: tokens for place, tokens in marking.items()}

    def _markings_equal(self, m1: Marking, m2: Marking) -> bool:
        """Compare two markings"""
        d1 = self._marking_to_dict(m1)
        d2 = self._marking_to_dict(m2)
        return d1 == d2

    def spawn_instance(self):
        """Spawn a new process instance"""
        self.case_counter += 1
        case_id = f"case_{self.case_counter}"
        self.env.process(self.execute_instance(case_id))
        return case_id

    def spawn_at_time(self, delay: float):
        """Spawn a process instance after a delay"""
        yield self.env.timeout(delay)
        self.spawn_instance()

    def execute_instance(self, case_id: str):
        """Execute a process instance using token-based Petri Net semantics"""
        # Create working copy of marking (as dict for easier handling)
        marking = Marking()
        for place, tokens in self.initial_marking.items():
            marking[place] = tokens

        iteration = 0
        max_iterations = 1000  # Safety limit for loops

        while not self._markings_equal(marking, self.final_marking):
            iteration += 1

            if iteration > max_iterations:
                print(f"Warning: {case_id} exceeded max iterations at t={self.env.now}")
                break

            # Get all enabled transitions
            enabled = self.get_enabled_transitions(marking)

            if not enabled:
                print(f"Warning: {case_id} reached deadlock at t={self.env.now}")
                print(f"  Current marking: {self._marking_to_dict(marking)}")
                print(f"  Expected final: {self._marking_to_dict(self.final_marking)}")
                break

            # Choose transition to fire (random for Task 1.1)
            transition = self.choose_transition(enabled)

            # Fire transition (execute activity)
            yield self.env.process(self.fire_transition(case_id, transition))

            # Update marking (consume/produce tokens)
            marking = self.update_marking(transition, marking)

    def get_enabled_transitions(self, marking: Marking) -> Set[PetriNet.Transition]:
        """Get all enabled transitions for current marking"""
        enabled = set()

        for transition in self.net.transitions:
            if self.is_enabled(transition, marking):
                enabled.add(transition)

        return enabled

    def is_enabled(self, transition: PetriNet.Transition, marking: Marking) -> bool:
        """Check if a transition is enabled (all input places have tokens)"""
        for arc in transition.in_arcs:
            place = arc.source
            required = arc.weight if hasattr(arc, "weight") else 1

            if place not in marking or marking[place] < required:
                return False

        return True

    def choose_transition(
        self, enabled: Set[PetriNet.Transition]
    ) -> PetriNet.Transition:
        """
        Choose which transition to fire from enabled set
        Random choice for Task 1.1 (Task 1.4 will add probabilities)
        """
        return random.choice(list(enabled))

    def fire_transition(self, case_id: str, transition: PetriNet.Transition):
        """Fire a transition: execute activity with processing time and logging"""
        # Skip invisible/silent transitions (tau transitions)
        if transition.label is None or transition.label == "":
            return self.env.timeout(0)  # Instant event

        activity_name = transition.label

        # Log activity start
        self.event_logger.log_event(
            case_id=case_id,
            activity=activity_name,
            timestamp=self.env.now,
            lifecycle="start",
        )

        # Get processing time (Task 1.3 - for now: fixed value)
        processing_time = self.get_processing_time(activity_name)

        # Simulate processing time
        yield self.env.timeout(processing_time)

        # Log activity completion
        self.event_logger.log_event(
            case_id=case_id,
            activity=activity_name,
            timestamp=self.env.now,
            lifecycle="complete",
        )

    def update_marking(
        self, transition: PetriNet.Transition, marking: Marking
    ) -> Marking:
        """Execute Petri Net semantics: consume tokens from inputs, produce to outputs"""
        new_marking = Marking()

        # Copy current marking
        for place, tokens in marking.items():
            new_marking[place] = tokens

        # Consume tokens from input places
        for arc in transition.in_arcs:
            place = arc.source
            tokens_to_remove = arc.weight if hasattr(arc, "weight") else 1
            new_marking[place] -= tokens_to_remove

            # Remove place if no tokens left
            if new_marking[place] == 0:
                del new_marking[place]

        # Produce tokens in output places
        for arc in transition.out_arcs:
            place = arc.target
            tokens_to_add = arc.weight if hasattr(arc, "weight") else 1

            if place in new_marking:
                new_marking[place] += tokens_to_add
            else:
                new_marking[place] = tokens_to_add

        return new_marking

    def get_processing_time(self, activity: str) -> float:
        """
        Get processing time for activity
        Placeholder for Task 1.3 - Will implement distributions and ML models
        """
        return 10.0

    def run(self, until: float):
        """Run simulation until specified time"""
        print(f"\nStarting simulation (until t={until})...")
        self.env.run(until=until)
        print(f"Simulation completed. Processed {self.case_counter} cases.")

        # Write event log to CSV
        self.event_logger.write_to_csv()
