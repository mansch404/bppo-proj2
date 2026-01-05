import pandas as pd
import numpy as np
import random
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set, Tuple


class ResourceProfile:
    def __init__(self, name, avg_start_hour, avg_end_hour, work_days):
        self.name = name
        self.start_hour = int(avg_start_hour)
        self.end_hour = int(avg_end_hour)
        self.work_days = work_days

        # Advanced 1.5: Konfiguration
        self.lunch_window_start = 12
        self.lunch_window_end = 14
        # Chance, dass diese Person Überstunden macht (z.B. basierend auf Seniorität im Log)
        self.overtime_probability = 0.3


class AdvancedResourceManager:
    def __init__(self, simulation_start_time: datetime):
        self.simulation_start_time = simulation_start_time

        # 1.6 Advanced++: Clustering Data
        self.roles: Dict[str, List[str]] = {}  # RoleID -> List[ResourceNames]
        self.activity_permissions: Dict[str, Set[str]] = {}  # Activity -> Set[RoleIDs]

        # 1.7 Metric
        self.competence: Dict[str, Dict[str, int]] = {}

        self.profiles: Dict[str, ResourceProfile] = {}
        self.busy_until: Dict[str, datetime] = {}
        self.default_profile = ResourceProfile("default", 9, 17, {0, 1, 2, 3, 4})

    def load_log_and_mine_profiles(self, log_path: str):
        print(f"Mining High-Level Profiles & Roles from {log_path}...")

        if log_path.endswith('.xes'):
            import pm4py
            log = pm4py.read_xes(log_path)
            df = pm4py.convert_to_dataframe(log)
        else:
            df = pd.read_csv(log_path)

        res_col = 'org:resource' if 'org:resource' in df.columns else 'resource'
        act_col = 'concept:name' if 'concept:name' in df.columns else 'activity'
        time_col = 'time:timestamp' if 'time:timestamp' in df.columns else 'timestamp'

        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.dropna(subset=[res_col])

        # PREPARATION
        df['date'] = df[time_col].dt.date
        df['hour'] = df[time_col].dt.hour

        resource_activities: Dict[str, Set[str]] = {}
        resources = df[res_col].unique()

        for res in resources:
            res_str = str(res)
            res_data = df[df[res_col] == res]

            # A) Aktivitäten sammeln
            acts = set(res_data[act_col].unique())
            resource_activities[res_str] = acts

            # B) Kompetenz zählen
            for act in acts:
                count = len(res_data[res_data[act_col] == act])
                if act not in self.competence: self.competence[act] = {}
                self.competence[act][res_str] = count

            # C) Zeit-Profile minen
            daily_stats = res_data.groupby('date')['hour'].agg(['min', 'max'])
            if daily_stats.empty: continue

            avg_start = int(daily_stats['min'].quantile(0.25))
            avg_end = int(daily_stats['max'].quantile(0.75))
            if avg_end <= avg_start: avg_end = avg_start + 8
            active_days = set(res_data[time_col].dt.weekday.unique())

            prof = ResourceProfile(res_str, avg_start, avg_end, active_days)
            # Extra: Wer oft spät arbeitet, hat höhere Overtime-Chance
            late_work_count = len(res_data[res_data['hour'] > 18])
            if late_work_count > 5:
                prof.overtime_probability = 0.8

            self.profiles[res_str] = prof
            self.busy_until[res_str] = self.simulation_start_time

        #1.6 ADVANCED: CLUSTERING (Role Discovery)
        self._discover_roles_via_clustering(resource_activities)

        print(f"Mined {len(self.profiles)} profiles. Discovered {len(self.roles)} roles via clustering.")

    def _discover_roles_via_clustering(self, resource_activities: Dict[str, Set[str]]):
        """
        Gruppiert Ressourcen basierend auf Jaccard-Similarity (Clustering).
        Das erlaubt 'unscharfe' Rollen (z.B. Junior vs Senior).
        """
        # Wir bauen Cluster auf. Ein Cluster ist definiert durch ein Set an Activities.
        # Structure: list of {'prototype_acts': Set, 'members': List[res]}
        clusters = []

        # Schwellenwert: 70% Übereinstimmung reicht für gleiche Rolle
        SIMILARITY_THRESHOLD = 0.7

        for res, acts in resource_activities.items():
            best_cluster_idx = -1
            best_similarity = -1.0

            # Suche passendes Cluster
            for idx, cluster in enumerate(clusters):
                # Jaccard Index: Intersection / Union
                intersection = len(acts.intersection(cluster['prototype_acts']))
                union = len(acts.union(cluster['prototype_acts']))
                similarity = intersection / union if union > 0 else 0

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_idx = idx

            # Entscheidung: Hinzufügen oder neu erstellen
            if best_similarity >= SIMILARITY_THRESHOLD:
                clusters[best_cluster_idx]['members'].append(res)
                # Update des Prototyps (wir lernen neue Activities dazu -> Generalisierung!)
                clusters[best_cluster_idx]['prototype_acts'].update(acts)
            else:
                clusters.append({'prototype_acts': set(acts), 'members': [res]})

        # Cluster in finale Struktur übertragen
        for idx, cluster in enumerate(clusters):
            role_name = f"Role_Cluster_{idx}"
            self.roles[role_name] = cluster['members']

            # Alle Mitglieder erben ALLE Rechte des Clusters (RBAC Generalization)
            for act in cluster['prototype_acts']:
                if act not in self.activity_permissions:
                    self.activity_permissions[act] = set()
                self.activity_permissions[act].add(role_name)

    def check_availability(self, resource: str, current_time: datetime) -> bool:
        """
        1.5 Super Advanced: Krankheit, Overtime, Micro-Interruptions
        """
        profile = self.profiles.get(resource, self.default_profile)

        # 1. Tag-Check
        if current_time.weekday() not in profile.work_days: return False

        current_h = current_time.hour

        # 2. Uhrzeit mit OVERTIME Logik
        # Normalerweise Ende: profile.end_hour.
        # Overtime möglich bis +2 Stunden, wenn profile.overtime_probability hoch ist.
        is_working_hours = profile.start_hour <= current_h < profile.end_hour

        is_overtime = False
        if not is_working_hours:
            # Check ob wir in der Overtime-Zone sind (bis 2h nach Schichtende)
            if profile.end_hour <= current_h < (profile.end_hour + 2):
                # Zufällige Entscheidung pro Stunde (deterministisch für Konsistenz innerhalb der Stunde)
                hour_seed = f"{resource}_{current_time.date()}_{current_h}"
                if random.Random(hour_seed).random() < profile.overtime_probability:
                    is_overtime = True

        if not (is_working_hours or is_overtime):
            return False

        # 3. Krankheit (Daily Stochastic)
        day_seed = f"{resource}_{current_time.date()}"
        if random.Random(day_seed).random() < 0.02: return False

        # 4. Micro-Interruptions (Meetings/Phone)
        # Jede Stunde gibt es eine 10% Chance, dass man für 15 Min blockiert ist
        # Wir simulieren das einfach: Zufallswurf bei jedem Request
        if random.random() < 0.05:  # 5% Chance kurz weg zu sein
            return False

        # 5. Mittagspause
        if profile.lunch_window_start <= current_h < profile.lunch_window_end:
            if random.random() < 0.3: return False

        return True

    def request_resource(self, activity: str, current_sim_time: float, duration: float) -> Optional[str]:
        real_time = self.simulation_start_time + timedelta(seconds=current_sim_time)

        # A) Roles (Clustering-Based)
        allowed_roles = self.activity_permissions.get(activity, set())
        candidates = []
        for role in allowed_roles:
            candidates.extend(self.roles[role])

        if not candidates: return "System"

        # B) Availability (Overtime/Interruptions)
        available = []
        weights = []

        for res in candidates:
            if not self.check_availability(res, real_time): continue

            if self.busy_until.get(res, self.simulation_start_time) <= real_time:
                available.append(res)
                # C) Competence Weighting
                weights.append(self.competence.get(activity, {}).get(res, 1))

        if not available: return None

        selected = random.choices(available, weights=weights, k=1)[0]
        self.busy_until[selected] = real_time + timedelta(seconds=duration)
        return selected