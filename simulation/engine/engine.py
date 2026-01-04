"""
Simulation Engine Module
Core discrete event simulation engine using Petri Net semantics

Supports two modes for processing time prediction:
- Basic: Static distributions per activity (Task 1.3 Basic)
- Advanced: Context-aware quantile regression (Task 1.3 Advanced)
"""

import simpy
import random
from typing import Set, Dict, Optional
from pm4py.objects.petri_net.obj import PetriNet, Marking
from .logger import EventLogger
from datetime import datetime, timedelta

# from .resource_manager import ResourceManager
import pickle
from scipy import stats

# Import inference functions from advanced_processing_time module
from timing.advanced_processing_time import (
    predict_processing_time_distribution,
    sample_from_quantiles,
    load_models as load_quantile_models,
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
        case_attributes: Dict = None,
        simulation_start_datetime: datetime = None,
    ):
        """
        Initialize simulation engine with Petri Net.

        Args:
            net: Petri Net object from pm4py
            initial_marking: Initial marking (token positions)
            final_marking: Final marking (simulation end condition)
            event_log_path: Path for output CSV event log
            fitted_distributions_path: Path to fitted distributions file (Basic)
            quantile_models_path: Path to quantile models file (Advanced)
            use_advanced_model: If True, use context-aware quantile regression
            case_attributes: Default case attributes for simulation
                             (RequestedAmount, LoanGoal, ApplicationType)
            simulation_start_datetime: Real-world datetime when simulation starts.
                                       Required for correct temporal feature calculation.
                                       Default: 2016-01-04 08:00:00 (Monday, business hours)
        """
        self.env = simpy.Environment()
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.event_logger = EventLogger(event_log_path)
        # self.resource_manager = ResourceManager()
        self.case_counter = 0

        # Default to Monday 2016-01-04 08:00 (matches BPI 2017 dataset timeframe)
        self.simulation_start_datetime = simulation_start_datetime or datetime(
            2016, 1, 4, 8, 0, 0
        )

        # Processing time model selection
        self.use_advanced_model = use_advanced_model

        # Default case attributes (can be overridden per case)
        self.default_case_attributes = case_attributes or {
            "RequestedAmount": 15000.0,
            "LoanGoal": "Home improvement",
            "ApplicationType": "New credit",
        }

        # Load Basic model (static distributions)
        self.fitted_distributions = self._load_basic_distributions(
            fitted_distributions_path
        )

        # Load Advanced model (quantile regression) if requested
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
            f"  Processing time model: {'Advanced (Quantile Regression)' if self.use_advanced_model else 'Basic (Static Distributions)'}"
        )
        if not self.use_advanced_model:
            print(f"  Fitted distributions loaded: {len(self.fitted_distributions)}")
        else:
            print(
                f"  Quantile models loaded: {len(self.quantile_models['models']) if self.quantile_models else 0} quantiles"
            )
        print(f"  Simulation start datetime: {self.simulation_start_datetime}")
        print(f"  Initial marking: {self._marking_to_dict(initial_marking)}")
        print(f"  Final marking: {self._marking_to_dict(final_marking)}")

    def _get_current_datetime(self) -> datetime:
        """
        Convert simulation time (seconds) to actual datetime.

        Returns:
            datetime object representing current simulation time in real-world terms
        """
        return self.simulation_start_datetime + timedelta(seconds=self.env.now)

    def _load_basic_distributions(self, filepath: str) -> Dict:
        """Load fitted distributions for Basic approach."""
        try:
            with open(filepath, "rb") as f:
                distributions = pickle.load(f)
            print(f"Loaded {len(distributions)} fitted distributions (Basic)")
            return distributions
        except FileNotFoundError:
            print(f"Warning: Could not find {filepath}")
            return {}

    def _load_advanced_model(self, filepath: str) -> Optional[Dict]:
        """Load quantile regression models for Advanced approach."""
        try:
            models_data = load_quantile_models(filepath)
            print(f"Loaded quantile models (Advanced): {models_data['quantiles']}")
            return models_data
        except FileNotFoundError:
            print(f"Warning: Could not find {filepath}, falling back to Basic")
            self.use_advanced_model = False
            return None

    def _marking_to_dict(self, marking: Marking) -> Dict:
        """Convert pm4py Marking to simple dict for comparison."""
        return {place.name: tokens for place, tokens in marking.items()}

    def _markings_equal(self, m1: Marking, m2: Marking) -> bool:
        """Compare two markings."""
        d1 = self._marking_to_dict(m1)
        d2 = self._marking_to_dict(m2)
        return d1 == d2

    def spawn_instance(self, case_attributes: Dict = None):
        """
        Spawn a new process instance.

        Args:
            case_attributes: Optional dict with case-specific attributes
                            (RequestedAmount, LoanGoal, ApplicationType)
        """
        self.case_counter += 1
        case_id = f"case_{self.case_counter}"

        # Merge default and provided case attributes
        attrs = self.default_case_attributes.copy()
        if case_attributes:
            attrs.update(case_attributes)

        self.env.process(self.execute_instance(case_id, attrs))
        return case_id

    def spawn_at_time(self, delay: float, case_attributes: Dict = None):
        """Spawn a process instance after a delay."""
        yield self.env.timeout(delay)
        self.spawn_instance(case_attributes)

    def execute_instance(self, case_id: str, case_attributes: Dict):
        """
        Execute a process instance using token-based Petri Net semantics.

        Tracks context for Advanced processing time prediction:
        - previous_activity: Last executed activity
        - event_nr: Number of events in this case
        - case_start_time: When the case started
        - offer_info: Offer attributes (filled when O_Create Offer executes)
        """
        # Create working copy of marking
        marking = Marking()
        for place, tokens in self.initial_marking.items():
            marking[place] = tokens

        # Initialize case context for Advanced model
        case_context = {
            "case_id": case_id,
            "case_start_time": self.env.now,
            "previous_activity": "START",
            "event_nr": 0,
            "case_attributes": case_attributes,
            "offer_info": {
                "CreditScore": None,
                "OfferedAmount": None,
                "NumberOfTerms": None,
                "MonthlyCost": None,
            },
        }

        iteration = 0
        max_iterations = 1000

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

            transition = self.choose_transition(enabled)

            # Fire transition with context
            yield self.env.process(
                self.fire_transition(case_id, transition, case_context)
            )

            # Update marking
            marking = self.update_marking(transition, marking)

    def get_enabled_transitions(self, marking: Marking) -> Set[PetriNet.Transition]:
        """Get all enabled transitions for current marking."""
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

    def fire_transition(
        self, case_id: str, transition: PetriNet.Transition, case_context: Dict
    ):
        """
        Fire a transition: execute activity with processing time and logging.

        Updates case_context for Advanced model tracking.
        """
        # Skip invisible/silent transitions
        if transition.label is None or transition.label == "":
            yield self.env.timeout(0)
            return

        activity_name = transition.label

        # Log activity start
        self.event_logger.log_event(
            case_id=case_id,
            activity=activity_name,
            timestamp=self.env.now,
            lifecycle="start",
        )

        # Get processing time (Basic or Advanced)
        if self.use_advanced_model and self.quantile_models is not None:
            processing_time = self._get_processing_time_advanced(
                activity_name, case_context
            )
        else:
            processing_time = self._get_processing_time_basic(activity_name)

        # Simulate processing time
        yield self.env.timeout(processing_time)

        # Log activity completion
        self.event_logger.log_event(
            case_id=case_id,
            activity=activity_name,
            timestamp=self.env.now,
            lifecycle="complete",
        )

        # Update context for next activity
        case_context["previous_activity"] = activity_name
        case_context["event_nr"] += 1

        # Simulate offer info being set (for demo purposes)
        # In a real simulation, this would come from the process logic
        if (
            activity_name == "O_Create Offer"
            and case_context["offer_info"]["CreditScore"] is None
        ):
            case_context["offer_info"] = {
                "CreditScore": random.randint(500, 100000),
                "OfferedAmount": case_context["case_attributes"]["RequestedAmount"]
                * random.uniform(0.8, 1.2),
                "NumberOfTerms": random.choice([12, 24, 36, 48, 60, 84, 120]),
                "MonthlyCost": random.uniform(100, 500),
            }

    def update_marking(
        self, transition: PetriNet.Transition, marking: Marking
    ) -> Marking:
        """Execute Petri Net semantics: consume/produce tokens."""
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

    def _get_processing_time_basic(self, activity: str) -> float:
        """Get processing time using Basic approach (static distributions)."""
        if activity not in self.fitted_distributions:
            print(f"Warning: No distribution for '{activity}', using default 10.0s")
            return 10.0

        info = self.fitted_distributions[activity]
        dist_name = info["distribution"]
        params = info["params"]

        # Sample from distribution
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

            # Ensure positive (in case of normal distribution)
            return max(0.01, sample)

        except Exception as e:
            print(f"Error sampling for {activity}: {e}")
            return 10.0

    def _get_processing_time_advanced(self, activity: str, case_context: Dict) -> float:
        """
        Get processing time using Advanced approach (context-aware quantile regression).

        Uses the trained LightGBM quantile models to predict a distribution,
        then samples from that distribution.
        """
        # Get actual datetime for temporal features
        current_datetime = self._get_current_datetime()
        hour_of_day = current_datetime.hour
        day_of_week = current_datetime.weekday()  # 0=Monday, 6=Sunday

        # Calculate elapsed time since case start
        elapsed_time = self.env.now - case_context["case_start_time"]

        # Get case attributes and offer info
        case_attrs = case_context["case_attributes"]
        offer_info = case_context["offer_info"]

        # Predict quantile distribution using imported function
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

        # Sample from the predicted distribution using imported function
        return sample_from_quantiles(predictions)

    def get_processing_time(self, activity: str) -> float:
        return self._get_processing_time_basic(activity)

    def run(self, until: float):
        """Run simulation until specified time."""
        print(f"\nStarting simulation (until t={until})...")
        print(
            f"  Simulation datetime range: {self.simulation_start_datetime} to {self.simulation_start_datetime + timedelta(seconds=until)}"
        )
        self.env.run(until=until)
        print(f"Simulation completed. Processed {self.case_counter} cases.")

        # Write event log to CSV
        self.event_logger.write_to_csv()
