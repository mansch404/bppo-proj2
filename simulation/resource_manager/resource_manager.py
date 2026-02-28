import pandas as pd
import numpy as np
import pm4py
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set, Any
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =================================================================
# PART 1: RESOURCE PLANNER STRATEGIES (Task 1.1, 1.7 & Paper [4])
# =================================================================

class ResourcePlanner(ABC):
    @abstractmethod
    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs) -> Optional[str]:
        pass


# --- 1.1 & 1.7 BASIC HEURISTICS ---

class RandomPlanner(ResourcePlanner):
    """TASK 1.7 BASIC: Random allocation among authorized & available resources."""

    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs):
        real_time = manager.simulation_start_time + timedelta(seconds=current_sim_time)
        candidates = manager.get_available_authorized_candidates(activity, real_time, duration)
        return random.choice(candidates) if candidates else None


class RoundRobinPlanner(ResourcePlanner):
    """HEURISTIC: R-RRA (Round Robin). Ensures weighted fairness."""

    def __init__(self):
        self.last_index = {}

    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs):
        real_time = manager.simulation_start_time + timedelta(seconds=current_sim_time)
        candidates = manager.get_available_authorized_candidates(activity, real_time, duration)
        if not candidates: return None
        idx = (self.last_index.get(activity, -1) + 1) % len(candidates)
        self.last_index[activity] = idx
        return candidates[idx]


class ShortestQueuePlanner(ResourcePlanner):
    """HEURISTIC: R-SHQ (Shortest Queue First).
    Decision: We use 'busy_until' to represent the virtual queue length."""

    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs):
        real_time = manager.simulation_start_time + timedelta(seconds=current_sim_time)
        candidates = manager.get_available_authorized_candidates(activity, real_time, duration)
        if not candidates: return None
        return min(candidates, key=lambda r: manager.busy_until.get(r, manager.simulation_start_time))


# --- RUSSELL ET AL. WORKFLOW PATTERNS [4] ---

class DirectAllocationPlanner(ResourcePlanner):
    """PATTERN 1: Design Decision - Returns None if specific resource is unavailable (Wait)."""

    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs):
        target = kwargs.get('target_resource')
        real_time = manager.simulation_start_time + timedelta(seconds=current_sim_time)
        if target and manager.is_resource_available(target, real_time, duration):
            return target
        return None


class CaseHandlingPlanner(ResourcePlanner):
    """PATTERN 4: Ensures the same resource handles the whole case for context preservation."""

    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs):
        case_id = kwargs.get('case_id')
        real_time = manager.simulation_start_time + timedelta(seconds=current_sim_time)
        previous_res = manager.case_assignments.get(case_id)

        if previous_res:
            if manager.is_resource_available(previous_res, real_time, duration):
                return previous_res
            return None  # Force wait for the case owner to maintain continuity

        # If first task in case, pick best available
        return manager._greedy_fallback(activity, real_time, duration)


# --- ADVANCED OPTIMIZATION ---

class AdvancedOptimizationPlanner(ResourcePlanner):
    """THE CORE OPTIMIZER: Implementation of System-First + Strategic Fallback."""

    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs):
        return manager._perform_advanced_allocation(activity, current_sim_time, duration, **kwargs)


# =================================================================
# PART 2: RESOURCE MANAGEMENT (Tasks 1.5, 1.6 & OR Logic)
# =================================================================

class BaseResourceManager:
    """
    IMPLEMENTS BASIC TASKS:
    1.5 Basic: Simple interval-based availability.
    1.6 Basic: History-based permissions.
    """

    def __init__(self, simulation_start_time: datetime, strategy: ResourcePlanner):
        self.simulation_start_time = simulation_start_time
        self.strategy = strategy
        self.busy_until: Dict[str, datetime] = {}
        self.activity_permissions: Dict[str, Set[int]] = {}
        self.roles: Dict[int, List[str]] = {}
        self.case_assignments: Dict[str, str] = {}
        self.history_competence: Dict[str, Set[str]] = {}  # Basic 1.6: simple set of tasks done

    def is_resource_available(self, resource: str, current_time: datetime, duration: float) -> bool:
        # Task 1.5 Basic: Simple 9-to-5 working window
        if not (8 <= current_time.hour <= 18): return False
        return self.busy_until.get(resource, self.simulation_start_time) <= current_time

    def get_available_authorized_candidates(self, activity, real_time, duration) -> List[str]:
        allowed_roles = self.activity_permissions.get(activity, set())
        candidates = [res for rid in allowed_roles for res in self.roles.get(rid, [])]
        return [r for r in candidates if self.is_resource_available(r, real_time, duration)]


class AdvancedResourceManager(BaseResourceManager):
    """
    IMPLEMENTS ADVANCED TASKS & OR CONSTRAINTS:
    1.5 Advanced: 7x24 Probability Heatmaps (Monte Carlo).
    1.6 Advanced: K-Means Role Discovery.
    Operations Research: Setup Penalties, System-First, Capacity Budgets.
    """

    def __init__(self, simulation_start_time: datetime, strategy: ResourcePlanner):
        super().__init__(simulation_start_time, strategy)
        self.system_resources = {"User_1"}
        self.availability_matrix: Dict[str, Any] = {}
        self.daily_effort_capacities: Dict[str, float] = {}
        self.daily_work_seconds: Dict[str, Dict[str, float]] = {}
        self.last_activity: Dict[str, str] = {}
        self.competence_scores: Dict[str, Dict[str, int]] = {}
        self.senior_role_id: Optional[int] = None
        self.setup_penalty_multiplier = 0.15

    def mine_organizational_model(self, log_df: pd.DataFrame):
        """
        PHASE A: Gaussian-Informed Clustering & Advanced Mining.
        Distinguishes between Systems (User_1) and Humans.
        """
        res_col, act_col, time_col = 'org:resource', 'concept:name', 'time:timestamp'
        log_df[time_col] = pd.to_datetime(log_df[time_col], utc=True)

        # 1.6 Basic Permissions & Competence
        counts = log_df.groupby([res_col, act_col]).size().unstack(fill_value=0)
        self.competence_scores = counts.to_dict('index')

        # 1.5 Advanced: Heatmap Mining (7 Days x 24 Hours)
        log_df['weekday'] = log_df[time_col].dt.weekday
        log_df['hour'] = log_df[time_col].dt.hour

        # Identify System Resources (High frequency outliers)
        freq = log_df[res_col].value_counts()
        self.system_resources.update(freq[freq > freq.mean() + 3 * freq.std()].index)

        human_df = log_df[~log_df[res_col].isin(self.system_resources)].copy()

        # Build Probabilistic Heatmaps
        for res in human_df[res_col].unique():
            res_str = str(res)
            rd = human_df[human_df[res_col] == res]
            self.availability_matrix[res_str] = {
                d: {h: min((len(rd[(rd['weekday'] == d) & (rd['hour'] == h)]) / (len(rd) / 168)) * 0.4, 0.98)
                    for h in range(24)} for d in range(7)
            }
            # OR Capacity: 90th percentile of daily output
            daily_dur = rd.groupby(rd[time_col].dt.date).size() * 600  # Proxy for duration
            self.daily_effort_capacities[res_str] = float(daily_dur.quantile(0.9)) * 1.2

        # 1.6 Advanced: Role Discovery (K-Means)
        scaled_features = StandardScaler().fit_transform(counts)
        kmeans = KMeans(n_clusters=min(len(counts), 5), random_state=42).fit(scaled_features)

        for i, label in enumerate(kmeans.labels_):
            rid, rname = int(label), str(counts.index[i])
            self.roles.setdefault(rid, []).append(rname)
            for act in counts.columns[counts.loc[rname] > 0]:
                self.activity_permissions.setdefault(act, set()).add(rid)

        # System Role Mapping (ID: -1)
        self.roles[-1] = list(self.system_resources)
        for act in log_df[log_df[res_col].isin(self.system_resources)][act_col].unique():
            self.activity_permissions.setdefault(act, set()).add(-1)

    def is_resource_available(self, resource: str, current_time: datetime, duration: float) -> bool:
        """ADVANCED CHECK: Monte Carlo Heatmap + Daily Work Budget."""
        if resource in self.system_resources: return True

        # 1. Heatmap Probability
        prob = self.availability_matrix.get(resource, {}).get(current_time.weekday(), {}).get(current_time.hour, 0.0)
        if random.random() >= prob: return False

        # 2. Capacity Constraint
        date_key = current_time.strftime("%Y-%m-%d")
        used = self.daily_work_seconds.get(resource, {}).get(date_key, 0)
        return (used + duration) <= self.daily_effort_capacities.get(resource, 28800)

    def request_resource(self, activity, sim_time, duration, **kwargs) -> Optional[str]:
        """Technically executes the assignment with OR Setup Penalties."""
        selected = self.strategy.select_resource(self, activity, sim_time, duration, **kwargs)

        if selected:
            # Pattern 4 Tracking
            case_id = kwargs.get('case_id')
            if case_id: self.case_assignments[case_id] = selected

            real_time = self.simulation_start_time + timedelta(seconds=sim_time)

            # OR Penalty: Context Switching (15% duration increase)
            penalty = 1.15 if self.last_activity.get(selected) != activity else 1.0
            actual_duration = duration * penalty

            self.busy_until[selected] = real_time + timedelta(seconds=actual_duration)
            self.last_activity[selected] = activity

            # Log effort
            date_key = real_time.strftime("%Y-%m-%d")
            self.daily_work_seconds.setdefault(selected, {})
            self.daily_work_seconds[selected][date_key] = self.daily_work_seconds[selected].get(date_key,
                                                                                                0) + actual_duration

        return selected

    def _perform_advanced_allocation(self, activity, sim_time, duration, **kwargs):
        """THE OPTIMIZER: System-First + Weighted Suitability Score."""
        allowed_roles = self.activity_permissions.get(activity, set())

        # 1. SYSTEM-FIRST (Infinite capacity bots)
        if -1 in allowed_roles: return random.choice(self.roles[-1])

        # 2. STRATEGIC HUMAN MATCHING
        real_time = self.simulation_start_time + timedelta(seconds=sim_time)
        candidates = self.get_available_authorized_candidates(activity, real_time, duration)

        if not candidates: return None

        scored_candidates = []
        for c in candidates:
            # Multiplier logic for high-value cases and seniority
            is_senior = (self.senior_role_id is not None and c in self.roles[self.senior_role_id])
            score = 1.0
            if kwargs.get('amount', 0) > 20000 and is_senior: score *= 2.5
            if self.last_activity.get(c) == activity: score *= 1.2  # Bonus for staying in context

            # Suitability = historical competence * strategic weights
            suitability = self.competence_scores.get(c, {}).get(activity, 1) * score
            scored_candidates.append((c, suitability))

        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)[0][0]

    def _greedy_fallback(self, activity, real_time, duration):
        """Helper for patterns to find the first available person if no owner exists."""
        candidates = self.get_available_authorized_candidates(activity, real_time, duration)
        return candidates[0] if candidates else None



