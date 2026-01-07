import pandas as pd
import numpy as np
import pm4py
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer


class AdvancedResourceManager:
    def __init__(self, simulation_start_time: datetime):
        self.simulation_start_time = simulation_start_time

        # 1.6 Advanced: Role Discovery
        # RoleID -> List of Resource Names
        self.roles: Dict[int, List[str]] = {}
        # Activity -> Set of RoleIDs allowed to perform it
        self.activity_permissions: Dict[str, Set[int]] = {}

        # 1.5 Advanced: Availability Heatmap
        # Resource -> Weekday (0-6) -> Hour (0-23) -> Probability (0.0 - 1.0)
        self.availability_matrix: Dict[str, Dict[int, Dict[int, float]]] = {}

        # 1.7 Metric: Competence (Frequency of performing activity)
        self.competence: Dict[str, Dict[str, int]] = {}

        # Tracking state
        self.busy_until: Dict[str, datetime] = {}

    def load_log_and_mine_profiles(self, log_path: str):
        """
        Main mining function.
        Combines Availability Heatmapping (1.5) AND Role Clustering (1.6).
        """
        print(f"--- MINING RESOURCES FROM {log_path} ---")

        # 1. Load Data
        if log_path.endswith('.xes'):
            log = pm4py.read_xes(log_path)
            df = pm4py.convert_to_dataframe(log)
        else:
            df = pd.read_csv(log_path)

        # Standardize column names
        res_col = 'org:resource' if 'org:resource' in df.columns else 'resource'
        act_col = 'concept:name' if 'concept:name' in df.columns else 'activity'
        time_col = 'time:timestamp' if 'time:timestamp' in df.columns else 'timestamp'

        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.dropna(subset=[res_col])

        # Pre-calculate datetime features
        df['weekday'] = df[time_col].dt.weekday
        df['hour'] = df[time_col].dt.hour

        # ---------------------------------------------------------
        # TASK 1.5 ADVANCED: Mining Availability Heatmaps
        # ---------------------------------------------------------
        print("Mining Probabilistic Availability Profiles...")
        resources = df[res_col].unique()

        for res in resources:
            res_str = str(res)
            res_data = df[df[res_col] == res]
            self.availability_matrix[res_str] = {}

            # Competence mining (for 1.7)
            activities = res_data[act_col].unique()
            for act in activities:
                count = len(res_data[res_data[act_col] == act])
                if act not in self.competence: self.competence[act] = {}
                self.competence[act][res_str] = count

            # Availability Heatmap Calculation
            # We count events per hour slot relative to total activity
            # This captures lunch breaks and shifts automatically.
            total_events = len(res_data)

            for day in range(7):
                self.availability_matrix[res_str][day] = {}
                for hour in range(24):
                    # Count events in this specific slot (e.g., Mondays at 10am)
                    slot_events = len(res_data[(res_data['weekday'] == day) & (res_data['hour'] == hour)])

                    if total_events > 0:
                        # Normalize probability.
                        # We multiply by a factor (e.g. 20) because events are sparse.
                        # This estimates "If I need a resource at this hour, how likely are they active?"
                        prob = (slot_events / total_events) * 20.0
                        prob = min(prob, 0.95)  # Cap at 95% (always chance of illness)
                    else:
                        prob = 0.0

                    self.availability_matrix[res_str][day][hour] = prob

        # ---------------------------------------------------------
        # TASK 1.6 ADVANCED: Role Discovery via K-Means Clustering
        # ---------------------------------------------------------
        print("Performing Role Discovery (K-Means Clustering)...")

        # 1. Create Resource-Activity Matrix (Who does what?)
        # Group by resource and get list of unique activities
        resource_activities = df.groupby(res_col)[act_col].apply(list).to_dict()

        res_list = list(resource_activities.keys())
        # Convert activity lists to binary matrix (One-Hot Encoding)
        mlb = MultiLabelBinarizer()
        activity_matrix = mlb.fit_transform([resource_activities[r] for r in res_list])

        # 2. Apply K-Means
        # Estimate clusters: If < 5 resources, use 2 roles. Else, try 5.
        n_clusters = 5 if len(res_list) > 10 else min(len(res_list), 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(activity_matrix)

        # 3. Store Learned Roles
        for i, label in enumerate(labels):
            role_id = int(label)
            res_name = str(res_list[i])

            if role_id not in self.roles:
                self.roles[role_id] = []
            self.roles[role_id].append(res_name)

            # Assign permissions: If this resource does an activity, their Role allows it
            for act in resource_activities[res_name]:
                if act not in self.activity_permissions:
                    self.activity_permissions[act] = set()
                self.activity_permissions[act].add(role_id)

        print(f"--- MINING COMPLETE. Discovered {n_clusters} Roles. ---")

    def check_availability(self, resource: str, current_time: datetime) -> bool:
        """
        TASK 1.5 ADVANCED: Stochastic Availability Logic
        Uses the mined Heatmap to determine availability probabilistically.
        """
        if resource not in self.availability_matrix:
            return False

        weekday = current_time.weekday()
        hour = current_time.hour

        # 1. Lookup Mined Probability
        # (e.g., "User_1 is 80% active on Mondays at 10am")
        prob = self.availability_matrix[resource].get(weekday, {}).get(hour, 0.0)

        # 2. Apply Stochastic Check (Bernoulli Trial)
        # This naturally handles lunch (prob drops), night (prob is 0), and sick days (random chance)
        return random.random() < prob

    def request_resource(self, activity: str, current_sim_time: float, duration: float) -> Optional[str]:
        """
        TASK 1.7: Resource Allocation
        """
        real_time = self.simulation_start_time + timedelta(seconds=current_sim_time)

        # 1. Filter by Permission (Task 1.6)
        # Which roles can perform this activity?
        allowed_role_ids = self.activity_permissions.get(activity, set())

        candidates = []
        for role_id in allowed_role_ids:
            candidates.extend(self.roles.get(role_id, []))

        if not candidates: return "System"  # Fallback for automated tasks

        # 2. Filter by Availability (Task 1.5)
        available_candidates = []
        weights = []

        for res in candidates:
            # Check if busy with another simulation task
            if self.busy_until.get(res, self.simulation_start_time) > real_time:
                continue

            # Check probabilistic availability (Heatmap)
            if self.check_availability(res, real_time):
                available_candidates.append(res)
                # Competence weight (Frequency of doing this task)
                weights.append(self.competence.get(activity, {}).get(res, 1))

        if not available_candidates:
            return None  # Nobody available right now

        # 3. Random Allocation (Task 1.7)
        # We use weighted random choices to prefer "experts", but it is still stochastic.
        selected = random.choices(available_candidates, weights=weights, k=1)[0]

        # Book the resource
        self.busy_until[selected] = real_time + timedelta(seconds=duration)
        return selected