"""
Discrete-Event Simulation Engine

Central orchestrator for a business process simulation built on Petri net
semantics and the SimPy discrete-event framework. This module does not
implement domain logic directly; instead it delegates to external modules
for instance spawning, processing-time estimation, routing/branching
decisions, and resource allocation.

Key responsibilities:
  - Managing Petri net token flow (markings, enabled transitions, firing).
  - Spawning and executing process instances (cases) over simulated time.
  - Coordinating with a resource manager to model staff availability and
    queueing behaviour.
  - Selecting transitions via pluggable branching strategies (none, basic,
    advanced).
  - Computing activity durations through either fitted statistical
    distributions or a quantile-regression model.
  - Emitting start/complete lifecycle events to the event logger.
"""

import simpy
import random
import pickle
import numpy as np
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

try:
    from ..timing.advanced_processing_time import (
        predict_processing_time_distribution,
        sample_from_quantiles,
        load_models as load_quantile_models,
    )
except ImportError:
    print(
        "Warning: Could not import advanced_processing_time. Advanced mode will fail if enabled."
    )


class SimulationEngine:
    """Core simulation engine that replays a business process over a Petri net.

    The engine advances a SimPy environment through simulated time while
    moving tokens through a Petri net that represents the process model.
    For each process instance (case) it:

      1. Initialises a marking from the Petri net's start place.
      2. Iteratively determines enabled transitions, selects one (via the
         configured branching strategy), fires it, and updates the marking.
      3. Delegates processing-time calculation to either a basic fitted-
         distribution sampler or an advanced quantile-regression model.
      4. Requests and waits for a resource through the attached resource
         manager (or falls back to a static pool).
      5. Logs start and complete events for every visible activity.

    Args:
        net: PM4Py Petri net defining the process structure.
        initial_marking: Token placement at the start of every case.
        final_marking: Token placement that signals case completion.
        event_log_path: Destination CSV path for the simulated event log.
        fitted_distributions_path: Pickle file mapping activity names to
            fitted scipy distributions (used in basic mode).
        quantile_models_path: Pickle file containing trained quantile-
            regression models (used in advanced mode).
        use_advanced_model: If True, use quantile-regression models for
            processing-time estimation instead of fitted distributions.
        branching_mode: Transition selection strategy -- one of "none"
            (uniform random), "basic" (label-level probabilities), or
            "advanced" (marking-aware classifier).
        branching_model_path: Path to a pickled branching model. If None
            and ``original_log_path`` is provided, a model is fitted on
            the fly.
        case_attributes: Default attribute dict attached to every new case
            (e.g. RequestedAmount, LoanGoal).
        simulation_start_datetime: Real-world timestamp corresponding to
            SimPy time 0. Defaults to 2016-01-04 08:00.
        simulation_end_datetime: Cutoff datetime after which no new cases
            are spawned. Defaults to 2016-06-04 20:00.
        original_log_path: Path to the original event log (XES/CSV) used
            to fit spawner and branching models when no pre-trained model
            is supplied.
        resource_manager: External resource-allocation component. When
            provided, the engine delegates all resource requests to it;
            otherwise a simple random fallback pool is used.
        spawner: (Reserved) Pre-configured spawner instance.
        spawner_advanced: If True, use KDE-based dynamic inter-arrival
            times; otherwise use a static distribution spawner.
        list_of_arrivals: Pre-generated list of case arrival datetimes
            (used primarily in evaluation mode).
        evaluation_flag: If True, the engine expects arrivals to be
            injected externally (e.g. by an evaluation harness) rather
            than generating them internally.
    """

    def __init__(
        self,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        event_log_path: str,
        fitted_distributions_path: str = "fitted_distributions.pkl",
        quantile_models_path: str = None,
        use_advanced_model: bool = False,
        branching_mode: str = "none",
        branching_model_path: str = None,
        case_attributes: Dict = None,
        simulation_start_datetime: datetime = None,
        simulation_end_datetime: datetime = None,
        original_log_path: str = "",
        resource_manager=None,
        spawner=None,
        spawner_advanced: bool = False,
        list_of_arrivals=None,
        evaluation_flag: bool = False,
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

        # Static fallback pool used when no resource manager is attached.
        self.fallback_resources = ["System", "User_1", "User_2", "User_3", "Manager"]

        # Anchor simulated time to a real-world calendar timestamp.
        self.simulation_start_datetime = simulation_start_datetime or datetime(
            2016, 1, 4, 8, 0, 0
        )

        self.simulation_end_datetime = simulation_end_datetime or datetime(
            2016, 6, 4, 20, 0, 0
        )

        self.use_advanced_model = use_advanced_model

        # --- Instance spawner configuration ---
        self.spawner_advanced = spawner_advanced

        self.evaluation_flag = evaluation_flag
        if self.evaluation_flag:
            # In evaluation mode arrivals are injected by the external harness.
            self.list_of_arrivals = []

        else:
            if self.spawner_advanced:
                self.spawner = DynamicSpawner_KDE()
                self.spawner.fit_with_event_log_path(original_log_path)
            else:
                self.spawner = StaticSpawner()
                self.spawner.fit_with_log_path(original_log_path)
            # Generate the full arrival schedule for the simulation window.
            self.list_of_arrivals = self.spawner.generate_arrivals(
                self.simulation_start_datetime, self.simulation_end_datetime
            )

        # --- Branching / routing strategy ---
        self.branching_mode = (branching_mode or "none").lower()
        self.branching_model = None

        if self.branching_mode in {"basic", "advanced"}:
            if branching_model_path:
                # Load a pre-trained branching model from disk.
                with open(branching_model_path, "rb") as f:
                    self.branching_model = pickle.load(f)
            elif original_log_path:
                # No saved model -- fit one from the original event log.
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

        # Compute a data-driven fallback from the median of all fitted distribution medians
        if self.fitted_distributions:
            medians = []
            for info in self.fitted_distributions.values():
                try:
                    dist = getattr(stats, info["distribution"])
                    medians.append(float(dist.median(*info["params"])))
                except Exception:
                    continue
            self.global_fallback_time = float(np.median(medians)) if medians else 10.0
        else:
            self.global_fallback_time = 10.0

        self.quantile_models = None
        if use_advanced_model:
            if quantile_models_path is None:
                quantile_models_path = "quantile_models.pkl"
            self.quantile_models = self._load_advanced_model(quantile_models_path)

        print(f"\nSimulation Engine initialized:")
        print(f"  Places: {len(net.places)}")
        print(f"  Transitions: {len(net.transitions)}")
        print(
            f"  Processing time model: {'Advanced' if self.use_advanced_model else 'Basic'}"
        )
        print(f"  Resource Manager attached: {self.resource_manager is not None}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_datetime(self) -> datetime:
        """Convert the current SimPy clock value to an absolute datetime."""
        return self.simulation_start_datetime + timedelta(seconds=self.env.now)

    def _load_basic_distributions(self, filepath: str) -> Dict:
        """Load per-activity fitted distribution parameters from a pickle file."""
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find {filepath}. Using defaults.")
            return {}

    def _load_advanced_model(self, filepath: str) -> Optional[Dict]:
        """Load quantile-regression models; disable advanced mode on failure."""
        try:
            return load_quantile_models(filepath)
        except Exception as e:
            # print(f"Error loading advanced models: {e}. Falling back to Basic.")
            self.use_advanced_model = False
            return None

    def _marking_to_dict(self, marking: Marking) -> Dict:
        """Convert a PM4Py Marking to a plain ``{place_name: token_count}`` dict."""
        return {place.name: tokens for place, tokens in marking.items()}

    def _markings_equal(self, m1: Marking, m2: Marking) -> bool:
        """Check structural equality of two markings by comparing their dict forms."""
        return self._marking_to_dict(m1) == self._marking_to_dict(m2)

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------

    def spawn_instance(self, case_attributes: Dict = None):
        """Create a new process instance and schedule its execution.

        Increments the global case counter, merges any per-case attributes
        with the defaults, and registers the ``execute_instance`` coroutine
        with the SimPy environment.

        Returns:
            The generated case identifier string (e.g. ``"case_42"``).
        """
        self.case_counter += 1
        case_id = f"case_{self.case_counter}"

        attrs = self.default_case_attributes.copy()
        if case_attributes:
            attrs.update(case_attributes)

        self.env.process(self.execute_instance(case_id, attrs))
        return case_id

    def spawn_at_time(self, delay: float, case_attributes: Dict = None):
        """SimPy process that waits ``delay`` seconds then spawns a case."""
        yield self.env.timeout(delay)
        self.spawn_instance(case_attributes)

    def execute_instance(self, case_id: str, case_attributes: Dict):
        """Main SimPy process driving a single case through the Petri net.

        Starting from the initial marking, the method repeatedly:
          1. Identifies enabled transitions.
          2. Selects one via ``choose_transition``.
          3. Fires it (logging events, waiting for resources, simulating
             work) via ``fire_transition``.
          4. Advances the marking via ``update_marking``.

        The loop terminates when the marking equals the final marking or
        a safety limit (1 000 iterations / deadlock) is reached.
        """
        marking = Marking()
        for place, tokens in self.initial_marking.items():
            marking[place] = tokens

        case_context = {
            "case_id": case_id,
            "case_start_time": self.env.now,
            "previous_activity": "START",
            "event_nr": 0,
            "history": [],
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

            yield self.env.process(
                self.fire_transition(case_id, transition, case_context)
            )

            marking = self.update_marking(transition, marking)

    # ------------------------------------------------------------------
    # Petri net mechanics
    # ------------------------------------------------------------------

    def get_enabled_transitions(self, marking: Marking) -> Set[PetriNet.Transition]:
        """Return the set of transitions whose input places hold enough tokens."""
        enabled = set()
        for transition in self.net.transitions:
            if self.is_enabled(transition, marking):
                enabled.add(transition)
        return enabled

    def is_enabled(self, transition: PetriNet.Transition, marking: Marking) -> bool:
        """Check whether every input arc of ``transition`` is satisfied by ``marking``."""
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
        """Select which enabled transition to fire next.

        The selection strategy depends on the configured branching mode:

        - **none**: Uniform random choice among visible transitions.
        - **basic**: The branching model predicts the next activity label
          from the case history, then the corresponding transition is
          selected. Falls back to uniform random on error.
        - **advanced**: The branching model receives the current marking
          and history and returns a transition directly. Falls back to
          uniform random on error.

        Silent (unlabelled) transitions are treated as internal routing
        and are only chosen when no visible transition is enabled.

        Raises:
            ValueError: If ``enabled`` is empty.
        """
        enabled_list = list(enabled)
        if not enabled_list:
            raise ValueError("enabled must be non-empty")

        # Partition into visible (labelled) and silent (routing) transitions.
        visible = [t for t in enabled_list if t.label not in (None, "")]
        silent = [t for t in enabled_list if t.label in (None, "")]

        # Only routing transitions available -- pick one at random.
        if not visible:
            return random.choice(silent) if silent else random.choice(enabled_list)

        # Single visible transition -- deterministic, no choice needed.
        if len(visible) == 1:
            return visible[0]

        # No branching model -- fall back to uniform random.
        if self.branching_model is None or self.branching_mode == "none":
            return random.choice(visible)

        history = []
        if case_context and "history" in case_context:
            history = list(case_context["history"])

        # --- Advanced branching: marking-aware classifier ---
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

        # --- Basic branching: history-based label prediction ---
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

    # ------------------------------------------------------------------
    # Transition firing (activity execution)
    # ------------------------------------------------------------------

    def fire_transition(
        self, case_id: str, transition: PetriNet.Transition, case_context: Dict
    ):
        """SimPy process that executes a single transition firing.

        For silent transitions the method yields immediately with zero
        delay. For visible (labelled) transitions the sequence is:

          1. Compute the processing time (basic or advanced model).
          2. Request a resource from the resource manager, retrying at a
             configurable interval until one becomes available.
          3. Log a *start* lifecycle event.
          4. Yield for the computed service duration (simulating work).
          5. Log a *complete* lifecycle event.
          6. Update case context (previous activity, event counter, and
             domain-specific offer information where applicable).
        """
        # Silent transitions carry no activity -- skip immediately.
        if transition.label is None or transition.label == "":
            yield self.env.timeout(0)
            return

        activity_name = transition.label

        # Append to the case trace so branching models can use history.
        case_context.setdefault("history", []).append(activity_name)

        # Step 1: Determine how long this activity will take.
        if self.use_advanced_model and self.quantile_models is not None:
            processing_time = self._get_processing_time_advanced(
                activity_name, case_context
            )
        else:
            processing_time = self._get_processing_time_basic(activity_name)

        # Step 2: Acquire a resource (blocks until one is available).
        resource = None
        arrival_seconds = float(self.env.now)

        if self.resource_manager:
            while resource is None:
                resource = self.resource_manager.request_resource(
                    activity=activity_name,
                    sim_time=self.env.now,
                    duration=processing_time,
                    case_id=case_id,
                    amount=case_context["case_attributes"].get("RequestedAmount", 0),
                )

                if resource is None:
                    # No resource available -- wait and retry.
                    retry_interval = getattr(
                        self.resource_manager, "retry_interval", 900
                    )
                    yield self.env.timeout(retry_interval)
        else:
            resource = random.choice(self.fallback_resources)

        start_seconds = float(self.env.now)
        wait_seconds = max(0.0, float(start_seconds - arrival_seconds))

        # Derive actual service duration from the resource manager's
        # busy-until timestamp when available; otherwise use the raw
        # processing time estimate.
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

        # Record per-activity metrics for downstream evaluation.
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

        # Step 3: Log the activity start event.
        self.event_logger.log_event(
            case_id=case_id,
            activity=activity_name,
            timestamp=self.env.now,
            lifecycle="start",
            resource=resource,
        )

        # Step 4: Simulate the activity's service duration.
        yield self.env.timeout(service_seconds)

        # Step 5: Log the activity completion event.
        self.event_logger.log_event(
            case_id=case_id,
            activity=activity_name,
            timestamp=self.env.now,
            lifecycle="complete",
            resource=resource,
        )

        case_context["previous_activity"] = activity_name
        case_context["event_nr"] += 1

        # Populate synthetic offer details on the first "O_Create Offer" event.
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
        """Apply a transition's token effects and return the new marking.

        Consumes tokens from each input place and produces tokens in each
        output place according to the arc weights defined in the Petri net.
        Places whose token count drops to zero are removed from the marking.
        """
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

    # ------------------------------------------------------------------
    # Processing-time estimation
    # ------------------------------------------------------------------

    def _get_processing_time_basic(self, activity: str) -> float:
        """Sample a processing time from the activity's fitted distribution.

        Falls back to a dynamically computed default when the activity has no
        fitted distribution or sampling fails. The result is clamped to a
        minimum of 0.01 seconds to avoid zero-duration events.
        """
        if activity not in self.fitted_distributions:
            return self.global_fallback_time

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
                sample = self.global_fallback_time
            return max(0.01, sample)
        except Exception as e:
            # print(f"Error sampling for {activity}: {e}")
            return self.global_fallback_time

    def _get_processing_time_advanced(self, activity: str, case_context: Dict) -> float:
        """Predict processing time using the quantile-regression model.

        Builds a feature vector from the current simulation state (time of
        day, day of week, elapsed case time, case attributes, and offer
        details), passes it to the quantile-regression predictor, and
        samples a duration from the predicted quantile distribution.

        Falls back to ``_get_processing_time_basic`` on any prediction
        error.
        """
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

    # ------------------------------------------------------------------
    # Simulation entry point
    # ------------------------------------------------------------------

    def run(self, until: float):
        """Execute the simulation up to the given SimPy time limit.

        After the SimPy environment finishes, the accumulated event log
        is flushed to CSV.
        """
        print(f"\nStarting simulation (until t={until})...")
        print(
            f"  Simulation datetime range: {self.simulation_start_datetime} to {self.simulation_start_datetime + timedelta(seconds=until)}"
        )
        self.env.run(until=until)
        print(f"Simulation completed. Processed {self.case_counter} cases.")
        self.event_logger.write_to_csv()
