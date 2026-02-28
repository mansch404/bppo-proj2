import pandas as pd
import numpy as np
import pm4py
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =================================================================
# RESOURCE PLANNER STRATEGIES (STRICTLY SEPARATED)
# =================================================================

class ResourcePlanner(ABC):
    @abstractmethod
    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs) -> Optional[str]:
        pass


class RandomPlanner(ResourcePlanner):
    """TRADITIONAL BASELINE: Simply picks any available authorized resource."""

    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs):
        real_time = manager.simulation_start_time + timedelta(seconds=current_sim_time)
        allowed_role_ids = manager.activity_permissions.get(activity, set())

        # No System-First logic here. Just gather all available authorized resources.
        candidates = [res for rid in allowed_role_ids for res in manager.roles.get(rid, [])
                      if manager.busy_until.get(res, manager.simulation_start_time) <= real_time
                      and manager.check_availability(res, real_time, duration)]

        return random.choice(candidates) if candidates else None


class GreedyPlanner(ResourcePlanner):
    """TRADITIONAL BASELINE: Assigns the first available authorized resource found."""

    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs):
        real_time = manager.simulation_start_time + timedelta(seconds=current_sim_time)
        allowed_role_ids = manager.activity_permissions.get(activity, set())

        # No System-First logic. Scan roles and assign immediately.
        for rid in allowed_role_ids:
            for res in manager.roles.get(rid, []):
                if manager.busy_until.get(res, manager.simulation_start_time) <= real_time:
                    if manager.check_availability(res, real_time, duration):
                        return res
        return None


class AdvancedOptimizationPlanner(ResourcePlanner):
    """ADVANCED OR MODEL: System-First + Seniority Match + Idling."""

    def select_resource(self, manager, activity, current_sim_time, duration, **kwargs):
        return manager._perform_advanced_allocation(activity, current_sim_time, duration, **kwargs)


# =================================================================
# ADVANCED RESOURCE MANAGER
# =================================================================

class AdvancedResourceManager:
    def __init__(self, simulation_start_time: datetime, strategy: ResourcePlanner):
        self.simulation_start_time = simulation_start_time
        self.strategy = strategy
        self.roles, self.activity_permissions, self.role_profiles = {}, {}, {}
        self.senior_role_id = None
        self.system_resources = {"User_1"}
        self.availability_matrix = {}
        self.daily_effort_capacities, self.daily_work_seconds = {}, {}
        self.last_activity, self.busy_until = {}, {}
        self.setup_penalty_multiplier = 0.15
        self.competence, self.predicted_demand = {}, []

    def load_log_and_mine_profiles(self, input_data):
        """Mines logs for work-second budgets, role permissions, and heatmaps."""
        df = input_data if isinstance(input_data, pd.DataFrame) else pm4py.convert_to_dataframe(
            pm4py.read_xes(input_data))
        res_col, act_col, time_col = 'org:resource', 'concept:name', 'time:timestamp'
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df['date'] = df[time_col].dt.date

        # Mine Durations & Capacities
        df = df.sort_values(by=['case:concept:name', time_col])
        df['duration'] = (df.groupby('case:concept:name')[time_col].shift(-1) - df[time_col]).dt.total_seconds().fillna(
            600).clip(60, 14400)
        effort_caps = df.groupby([res_col, 'date'])['duration'].sum().reset_index().groupby(res_col)[
            'duration'].quantile(0.9)

        # Separate Systems and Humans
        res_counts = df[res_col].value_counts()
        self.system_resources.update(res_counts[res_counts > res_counts.mean() + 3 * res_counts.std()].index)
        human_df = df[~df[res_col].isin(self.system_resources)].copy()

        for res in human_df[res_col].unique():
            res_str = str(res)
            self.daily_effort_capacities[res_str] = float(effort_caps.get(res, 14400)) * 1.2  # Flex budget
            rd = human_df[human_df[res_col] == res]
            self.availability_matrix[res_str] = {d: {
                h: min((len(rd[(rd[time_col].dt.weekday == d) & (rd[time_col].dt.hour == h)]) / len(rd)) * 20.0,
                       0.95) if len(rd) > 0 else 0.0 for h in range(24)} for d in range(7)}

        # Role Discovery
        res_act_counts = human_df.groupby([res_col, act_col]).size().unstack(fill_value=0)
        scaled = StandardScaler().fit_transform(res_act_counts)
        kmeans = KMeans(n_clusters=min(len(res_act_counts), 5), random_state=42, n_init=10).fit(scaled)
        for i, label in enumerate(kmeans.labels_):
            rid, rname = int(label), str(res_act_counts.index[i])
            self.roles.setdefault(rid, []).append(rname)
            for act in res_act_counts.columns[res_act_counts.loc[rname] > 0]:
                self.activity_permissions.setdefault(act, set()).add(rid)

        # Strict System Mapping
        self.roles[-1] = list(self.system_resources)
        for sr in self.system_resources:
            for act in df[df[res_col] == sr][act_col].unique():
                self.activity_permissions.setdefault(act, set()).add(-1)

        if len(self.roles) > 1:  # Find Seniority role if humans exist
            self.senior_role_id = max([k for k in self.roles.keys() if k != -1],
                                      key=lambda k: human_df[human_df[res_col].isin(self.roles[k])][
                                          'case:RequestedAmount'].mean())

    def check_availability(self, resource, current_time, task_duration):
        if resource in self.system_resources: return True
        prob = self.availability_matrix.get(resource, {}).get(current_time.weekday(), {}).get(current_time.hour, 0.0)
        if random.random() >= prob: return False
        date_key = current_time.strftime("%Y-%m-%d")
        return (self.daily_work_seconds.get(resource, {}).get(date_key,
                                                              0) + task_duration) <= self.daily_effort_capacities.get(
            resource, 14400)

    def update_predictions(self, tasks):
        self.predicted_demand = tasks

    def request_resource(self, activity, sim_time, duration, **kwargs) -> Optional[str]:
        selected = self.strategy.select_resource(self, activity, sim_time, duration, **kwargs)
        if selected and selected not in self.system_resources:
            real_time = self.simulation_start_time + timedelta(seconds=sim_time)

            # 15% Setup Penalty for switching activities
            actual_dur = duration * (1.15 if self.last_activity.get(selected) != activity else 1.0)

            self.busy_until[selected] = real_time + timedelta(seconds=actual_dur)
            self.last_activity[selected] = activity
            date_key = real_time.strftime("%Y-%m-%d")

            # SAFE DICTIONARY UPDATE: Prevents KeyError for new users like 'User_30'
            if selected not in self.daily_work_seconds:
                self.daily_work_seconds[selected] = {}

            # Update the daily effort budget tracker
            current_day_effort = self.daily_work_seconds[selected].get(date_key, 0)
            self.daily_work_seconds[selected][date_key] = current_day_effort + actual_dur

        return selected

    def _perform_advanced_allocation(self, activity, sim_time, duration, **kwargs):
        """THE CORE OPTIMIZER: System-First + Strategic Human Fallback."""
        allowed = self.activity_permissions.get(activity, set())

        # 1. SYSTEM-FIRST (The Optimized Path)
        if -1 in allowed: return random.choice(self.roles[-1])

        # 2. STRATEGIC HUMAN MATCHING (The Fallback)
        real_time = self.simulation_start_time + timedelta(seconds=sim_time)
        humans = [res for rid in allowed if rid != -1 for res in self.roles.get(rid, [])]
        avail = [h for h in humans if self.busy_until.get(h, self.simulation_start_time) <= real_time
                 and self.check_availability(h, real_time, duration)]

        if avail:
            scored = []
            for h in avail:
                is_senior = h in self.roles.get(self.senior_role_id, [])
                match = 2.0 if (kwargs.get('case_amount', 0) > 25000 and is_senior) else 1.0
                if self.last_activity.get(h) == activity: match *= 1.2  # Context Bonus
                scored.append((h, match))
            return sorted(scored, key=lambda x: x[1], reverse=True)[0][0]
        return None