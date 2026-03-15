"""
Simulation Engine Module
Core discrete event simulation engine using Petri Net semantics
"""

import simpy
import random
import pickle
from typing import Set, Dict, Optional
import math
from pm4py.objects.petri_net.obj import PetriNet, Marking
from datetime import datetime, timedelta
from scipy import stats

from .logger import EventLogger

from ..routing.branching_basic import BranchingBasic
from ..routing.branching_advanced import BranchingAdvanced
from ..spawner.dynamic_spawner import DynamicSpawner_KDE
from ..spawner.static_distribution import StaticSpawner

# Import inference functions from advanced_processing_time module
try:
    from timing.advanced_processing_time import (
        predict_processing_time_distribution,
        sample_from_quantiles,
        load_models as load_quantile_models,
    )
except ImportError:
    print(
        "Warning: Could not import advanced_processing_time. Advanced mode will fail if enabled."
    )


class SimulationEngine:
    def __init__(
        self,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        event_log_path: str,
        fitted_distributions_path: str = "fitted_distributions.pkl",
        quantile_models_path: str = None,
        use_advanced_model: bool = False,
        branching_mode: str = "none",  # <--- NEW ARGUMENT
        branching_model_path: str = None,  # <--- NEW ARGUMENT
        case_attributes: Dict = None,
        simulation_start_datetime: datetime = None,
        simulation_end_datetime: datetime = None,  # <--- New argument
        original_log_path: str = "",
        resource_manager=None,  # <--- NEW ARGUMENT
        spawner=None,  # <--- New argument
        spawner_advanced: bool = False,  # <--- New argument
        list_of_arrivals=None,  # <--- New argument
        evaluation_flag: bool = False
    ):
        self.env = simpy.Environment()
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.event_logger = EventLogger(
            event_log_path, start_time=simulation_start_datetime
        )
        self.case_counter = 0
        self.metric_records = []

        self.resource_manager = resource_manager

        # Fallback if no manager provided (simple list)
        self.fallback_resources = ["System", "User_1", "User_2", "User_3", "Manager"]

        # Default to Monday 2016-01-04 08:00
        self.simulation_start_datetime = simulation_start_datetime or datetime(
            2016, 1, 4, 8, 0, 0
        )

        self.simulation_end_datetime = simulation_end_datetime or datetime(
            2016, 6, 4, 20, 0, 0
        )

        self.use_advanced_model = use_advanced_model

        # Spawner (Task 1.2)
        self.spawner_advanced = spawner_advanced  # True if advanced approach

        self.evaluation_flag = evaluation_flag  # True if running in evaluation mode
        if self.evaluation_flag:
            # Empty list filled in evaluation script
            self.list_of_arrivals = []
            
        else:
            if self.spawner_advanced:
                self.spawner = DynamicSpawner_KDE()
                self.spawner.fit_with_event_log_path(original_log_path)
            else:
                self.spawner = StaticSpawner()
                self.spawner.fit_with_log_path(original_log_path)
            # Normal run: Generate a list of arrivals (as datetimes) for the simulation
            self.list_of_arrivals = self.spawner.generate_arrivals(
                self.simulation_start_datetime, self.simulation_end_datetime
            )
            

        # Branching (Task 1.4)
        # branching_mode: "none" | "basic" | "advanced"
        self.branching_mode = (branching_mode or "none").lower()
        self.branching_model = None

        if self.branching_mode in {"basic", "advanced"}:
            if branching_model_path:
                # Expect a pickled BranchingBasic / BranchingAdvanced instance
                with open(branching_model_path, "rb") as f:
                    self.branching_model = pickle.load(f)
            elif original_log_path:
                # Fit directly from the original log (XES/CSV)
                if self.branching_mode == "basic":
                    self.branching_model = BranchingBasic().fit_from_event_log(
                        original_log_path
                    )
                else:
                    self.branching_model = BranchingAdvanced().fit_from_event_log(
                        original_log_path, net, initial_marking
                    )

        self.default_case_attributes = case_attributes or {
            "RequestedAmount": 15000.0,
            "LoanGoal": "Home improvement",
            "ApplicationType": "New credit",
        }

        self.fitted_distributions = self._load_basic_distributions(
            fitted_distributions_path
        )

        self.quantile_models = None
        if use_advanced_model:
            if quantile_models_path is None:
                quantile_models_path = "quantile_models.pkl"
            self.quantile_models = self._load_advanced_model(quantile_models_path)

        # Debug info
        print(f"\nSimulation Engine initialized:")
        print(f"  Places: {len(net.places)}")
        print(f"  Transitions: {len(net.transitions)}")
        print(
            f"  Processing time model: {'Advanced' if self.use_advanced_model else 'Basic'}"
        )
        print(f"  Resource Manager attached: {self.resource_manager is not None}")

    def _get_current_datetime(self) -> datetime:
        return self.simulation_start_datetime + timedelta(seconds=self.env.now)

    def _load_basic_distributions(self, filepath: str) -> Dict:
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find {filepath}. Using defaults.")
            return {}

    def _load_advanced_model(self, filepath: str) -> Optional[Dict]:
        try:
            return load_quantile_models(filepath)
        except Exception as e:
            # print(f"Error loading advanced models: {e}. Falling back to Basic.")
            self.use_advanced_model = False
            return None

    def _marking_to_dict(self, marking: Marking) -> Dict:
        return {place.name: tokens for place, tokens in marking.items()}

    def _markings_equal(self, m1: Marking, m2: Marking) -> bool:
        return self._marking_to_dict(m1) == self._marking_to_dict(m2)

    def spawn_instance(self, case_attributes: Dict = None):
        self.case_counter += 1
        case_id = f"case_{self.case_counter}"

        attrs = self.default_case_attributes.copy()
        if case_attributes:
            attrs.update(case_attributes)

        self.env.process(self.execute_instance(case_id, attrs))
        return case_id

    def spawn_at_time(self, delay: float, case_attributes: Dict = None):
        yield self.env.timeout(delay)
        self.spawn_instance(case_attributes)

    def execute_instance(self, case_id: str, case_attributes: Dict):
        marking = Marking()
        for place, tokens in self.initial_marking.items():
            marking[place] = tokens

        case_context = {
            "case_id": case_id,
            "case_start_time": self.env.now,
            "previous_activity": "START",
            "event_nr": 0,
            "history": [],  # <-- New Context
            "case_attributes": case_attributes,
            "offer_info": {
                "CreditScore": None,
                "OfferedAmount": None,
                "NumberOfTerms": None,
                "MonthlyCost": None,
            },
        }

        iteration = 0
        while not self._markings_equal(marking, self.final_marking):
            iteration += 1
            if iteration > 1000:
                print(f"Warning: {case_id} exceeded max iterations")
                break

            enabled = self.get_enabled_transitions(marking)
            if not enabled:
                print(f"Warning: {case_id} reached deadlock")
                break

            transition = self.choose_transition(
                enabled, marking=marking, case_context=case_context
            )

            # --- CHANGE: Process the transition logic inside fire_transition ---
            yield self.env.process(
                self.fire_transition(case_id, transition, case_context)
            )

            marking = self.update_marking(transition, marking)

    def get_enabled_transitions(self, marking: Marking) -> Set[PetriNet.Transition]:
        enabled = set()
        for transition in self.net.transitions:
            if self.is_enabled(transition, marking):
                enabled.add(transition)
        return enabled

    def is_enabled(self, transition: PetriNet.Transition, marking: Marking) -> bool:
        for arc in transition.in_arcs:
            place = arc.source
            required = arc.weight if hasattr(arc, "weight") else 1
            if place not in marking or marking[place] < required:
                return False
        return True

    def choose_transition(
        self,
        enabled: Set[PetriNet.Transition],
        *,
        marking: Optional[Marking] = None,
        case_context: Optional[Dict] = None,
    ) -> PetriNet.Transition:
        """
        Select which enabled transition to fire

        Default: random choice

        If a branching model is configured (Task 1.4), we only intervene when:
        - at least two visible transitions (label not empty) are enabled,
        - and we have a trace history in case_context.

        Silent transitions (label None/"") are treated as routing and are only
        chosen if no visible transition is enabled.
        """
        enabled_list = list(enabled)
        if not enabled_list:
            raise ValueError("enabled must be non-empty")

        # Separate visible and silent transitions
        visible = [t for t in enabled_list if t.label not in (None, "")]
        silent = [t for t in enabled_list if t.label in (None, "")]

        # If only routing is possible, keep prior behavior
        if not visible:
            return random.choice(silent) if silent else random.choice(enabled_list)

        # If only one visible transition is possible, it is forced
        if len(visible) == 1:
            return visible[0]

        # If no branching model configured, keep random behavior among visible
        if self.branching_model is None or self.branching_mode == "none":
            return random.choice(visible)

        history = []
        if case_context and "history" in case_context:
            history = list(case_context["history"])

        # Advanced: model chooses transition directly, using marking signatures
        if (
            self.branching_mode == "advanced"
            and marking is not None
            and hasattr(self.branching_model, "choose_transition")
        ):
            try:
                return self.branching_model.choose_transition(
                    enabled=visible,
                    marking=marking,
                    history=history,
                    default_strategy="uniform",
                )
            except Exception:
                return random.choice(visible)

        # Basic: model chooses next activity label, then map to a transition
        if self.branching_mode == "basic" and hasattr(
            self.branching_model, "choose_next"
        ):
            try:
                enabled_labels = [str(t.label) for t in visible]
                chosen_label = self.branching_model.choose_next(
                    history=history,
                    enabled_next=enabled_labels,
                    default_strategy="uniform",
                )
                candidates = [t for t in visible if str(t.label) == str(chosen_label)]
                return (
                    random.choice(candidates) if candidates else random.choice(visible)
                )
            except Exception:
                return random.choice(visible)

        return random.choice(visible)

    def fire_transition(
        self, case_id: str, transition: PetriNet.Transition, case_context: Dict
    ):
        # 1. Handle Silent Transitions
        if transition.label is None or transition.label == "":
            yield self.env.timeout(0)
            return

        activity_name = transition.label

        # Update history for branching
        case_context.setdefault("history", []).append(activity_name)

        # 2. Calculate Processing Time FIRST (needed to book the resource)
        if self.use_advanced_model and self.quantile_models is not None:
            processing_time = self._get_processing_time_advanced(
                activity_name, case_context
            )
        else:
            processing_time = self._get_processing_time_basic(activity_name)

        # 3. Request Resource (Wait if busy/unavailable)
        resource = None
        arrival_seconds = float(self.env.now)

        if self.resource_manager:
            # Loop until we get a resource (Queueing behavior)
            while resource is None:
                # Ask manager: "Can anyone do 'activity_name' right now?"
                resource = self.resource_manager.request_resource(
                    activity=activity_name,
                    sim_time=self.env.now,
                    duration=processing_time,
                    case_id=case_id,  # important for CaseHandlingPlanner
                    amount=case_context["case_attributes"].get(
                        "RequestedAmount", 0
                    ),  # for Advanced Matcher
                )

                if resource is None:
                    # No one available (night time, or all busy)
                    # Wait 15 minutes (900 seconds) and check again
                    retry_interval = getattr(
                        self.resource_manager, "retry_interval", 900
                    )
                    yield self.env.timeout(retry_interval)
        else:
            # Fallback if no manager (old behavior)
            resource = random.choice(self.fallback_resources)

        start_seconds = float(self.env.now)
        wait_seconds = max(0.0, float(start_seconds - arrival_seconds))

        # Busy-until is the source of truth for allocation duration in the resource manager.
        end_seconds = None
        if self.resource_manager:
            busy_until = self.resource_manager.busy_until.get(resource)
            if busy_until is not None:
                end_seconds = float(
                    (busy_until - self.simulation_start_datetime).total_seconds()
                )

        if end_seconds is None or not math.isfinite(end_seconds):
            end_seconds = float(start_seconds + processing_time)

        service_seconds = max(0.0, float(end_seconds - start_seconds))
        is_system = (
            self.resource_manager is not None
            and resource in getattr(self.resource_manager, "system_resources", set())
        ) or (resource == "System")

        self.metric_records.append(
            {
                "case": case_id,
                "activity": activity_name,
                "resource": resource,
                "is_system": bool(is_system),
                "requested_amount": float(
                    case_context["case_attributes"].get("RequestedAmount", 1.0)
                ),
                "arrival_seconds": arrival_seconds,
                "start_seconds": start_seconds,
                "end_seconds": float(end_seconds),
                "wait_seconds": wait_seconds,
                "service_seconds": service_seconds,
                "timed_out": False,
            }
        )

        # 4. Log Start
        self.event_logger.log_event(
            case_id=case_id,
            activity=activity_name,
            timestamp=self.env.now,
            lifecycle="start",
            resource=resource,
        )

        # 5. Simulate Work
        yield self.env.timeout(service_seconds)

        # 6. Log Completion
        self.event_logger.log_event(
            case_id=case_id,
            activity=activity_name,
            timestamp=self.env.now,
            lifecycle="complete",
            resource=resource,
        )

        case_context["previous_activity"] = activity_name
        case_context["event_nr"] += 1

        # Simulate offer info logic
        if (
            activity_name == "O_Create Offer"
            and case_context["offer_info"]["CreditScore"] is None
        ):
            case_context["offer_info"] = {
                "CreditScore": random.randint(500, 1000),
                "OfferedAmount": case_context["case_attributes"]["RequestedAmount"]
                * random.uniform(0.8, 1.2),
                "NumberOfTerms": random.choice([12, 24, 36, 48, 60, 84, 120]),
                "MonthlyCost": random.uniform(100, 500),
            }

    def update_marking(
        self, transition: PetriNet.Transition, marking: Marking
    ) -> Marking:
        new_marking = Marking()
        for place, tokens in marking.items():
            new_marking[place] = tokens

        for arc in transition.in_arcs:
            place = arc.source
            tokens_to_remove = arc.weight if hasattr(arc, "weight") else 1
            new_marking[place] -= tokens_to_remove
            if new_marking[place] == 0:
                del new_marking[place]

        for arc in transition.out_arcs:
            place = arc.target
            tokens_to_add = arc.weight if hasattr(arc, "weight") else 1
            if place in new_marking:
                new_marking[place] += tokens_to_add
            else:
                new_marking[place] = tokens_to_add

        return new_marking

    def _get_processing_time_basic(self, activity: str) -> float:
        if activity not in self.fitted_distributions:
            return 10.0

        info = self.fitted_distributions[activity]
        dist_name = info["distribution"]
        params = info["params"]

        try:
            if dist_name == "lognorm":
                shape, loc, scale = params
                sample = stats.lognorm.rvs(shape, loc, scale)
            elif dist_name == "expon":
                loc, scale = params
                sample = stats.expon.rvs(loc, scale)
            elif dist_name == "gamma":
                shape, loc, scale = params
                sample = stats.gamma.rvs(shape, loc, scale)
            elif dist_name == "norm":
                loc, scale = params
                sample = stats.norm.rvs(loc, scale)
            else:
                sample = 10.0
            return max(0.01, sample)
        except Exception as e:
            # print(f"Error sampling for {activity}: {e}")
            return 10.0

    def _get_processing_time_advanced(self, activity: str, case_context: Dict) -> float:
        current_datetime = self._get_current_datetime()
        hour_of_day = current_datetime.hour
        day_of_week = current_datetime.weekday()
        elapsed_time = self.env.now - case_context["case_start_time"]
        case_attrs = case_context["case_attributes"]
        offer_info = case_context["offer_info"]

        try:
            predictions = predict_processing_time_distribution(
                activity=activity,
                previous_activity=case_context["previous_activity"],
                requested_amount=case_attrs.get("RequestedAmount", 15000.0),
                loan_goal=case_attrs.get("LoanGoal", "Unknown"),
                application_type=case_attrs.get("ApplicationType", "New credit"),
                event_nr=case_context["event_nr"],
                elapsed_time=elapsed_time,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                credit_score=offer_info.get("CreditScore"),
                offered_amount=offer_info.get("OfferedAmount"),
                number_of_terms=offer_info.get("NumberOfTerms"),
                monthly_cost=offer_info.get("MonthlyCost"),
                models_data=self.quantile_models,
            )
            return sample_from_quantiles(predictions)
        except Exception:
            return self._get_processing_time_basic(activity)

    def run(self, until: float):
        print(f"\nStarting simulation (until t={until})...")
        print(
            f"  Simulation datetime range: {self.simulation_start_datetime} to {self.simulation_start_datetime + timedelta(seconds=until)}"
        )
        self.env.run(until=until)
        print(f"Simulation completed. Processed {self.case_counter} cases.")
        self.event_logger.write_to_csv()
