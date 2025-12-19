import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Set


class ResourceManager:
    def __init__(self, simulation_start_time: datetime):
        self.simulation_start_time = simulation_start_time

        # 1.6 Permissions: Map Activity -> Set of allowed Resources
        self.permissions: Dict[str, Set[str]] = {}

        # 1.5 Availability: Map Resource -> Timestamp when they become free
        self.busy_until: Dict[str, datetime] = {}

        # Konfiguration für 1.5 (Interval / Schichten)
        # Beispiel: Mo-Fr, 09:00 - 17:00 Uhr
        self.work_start_hour = 9
        self.work_end_hour = 17
        self.work_days = [0, 1, 2, 3, 4]  # 0=Monday, 4=Friday

    def load_permissions_from_log(self, csv_path: str):
        """
        Task 1.6 (Basic): Implement permissions based on historical data.
        Liest das Log und merkt sich, wer welche Aktivität schon mal gemacht hat.
        """
        print(f"Loading Resource Permissions from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)

            # Wir brauchen nur Zeilen, wo eine Resource eingetragen ist
            if 'resource' in df.columns:
                df_clean = df.dropna(subset=['resource'])

                # Gruppieren: Welche Resource hat welche Activity gemacht?
                for _, row in df_clean.iterrows():
                    act = row['activity']
                    res = str(row['resource'])  # Sicherstellen, dass es ein String ist

                    if act not in self.permissions:
                        self.permissions[act] = set()
                    self.permissions[act].add(res)

                # Alle gefundenen Ressourcen initialisieren (sie sind am Anfang frei)
                all_resources = set().union(*self.permissions.values())
                for res in all_resources:
                    self.busy_until[res] = self.simulation_start_time

                print(f"Permissions loaded. Found {len(all_resources)} resources.")
            else:
                print("Warning: No 'resource' column found in log.")

        except FileNotFoundError:
            print("Warning: Log file not found. No permissions loaded.")

    def is_on_shift(self, current_time: datetime) -> bool:
        """
        Task 1.5 (Basic): Resource availabilities based on an interval.
        Prüft, ob der Zeitpunkt innerhalb der Arbeitszeiten liegt.
        """
        # 1. Ist es Wochenende?
        if current_time.weekday() not in self.work_days:
            return False

        # 2. Ist es außerhalb der Uhrzeiten (z.B. vor 9 oder nach 17 Uhr)?
        if not (self.work_start_hour <= current_time.hour < self.work_end_hour):
            return False

        return True

    def request_resource(self, activity: str, current_sim_time: float, duration: float) -> Optional[str]:
        """
        Task 1.7: Random Resource Allocation.
        Versucht, eine Ressource für eine bestimmte Dauer zu buchen.

        Returns:
            Name der Ressource (str), wenn erfolgreich.
            None, wenn niemand verfügbar ist.
        """
        # Umrechnung: SimPy Float-Zeit -> Echtes Datum
        real_time = self.simulation_start_time + timedelta(seconds=current_sim_time)

        # Schritt 1: Prüfen, ob wir überhaupt im "Dienst" sind (1.5)
        if not self.is_on_shift(real_time):
            return None  # Nachts/Wochenende arbeitet niemand

        # Schritt 2: Wer darf das machen? (1.6)
        candidates = list(self.permissions.get(activity, []))
        if not candidates:
            return "System"  # Fallback für Aktivitäten ohne Ressourcen-Daten (z.B. automatisierte Mails)

        # Schritt 3: Wer von denen ist gerade frei? (1.5)
        available_candidates = []
        for res in candidates:
            # Ressource ist frei, wenn ihr "busy_until" Zeitstempel in der Vergangenheit liegt
            if self.busy_until.get(res, self.simulation_start_time) <= real_time:
                available_candidates.append(res)

        if not available_candidates:
            return None  # Alle berechtigten Mitarbeiter sind beschäftigt

        # Schritt 4: Zufällige Zuweisung (1.7)
        selected_resource = random.choice(available_candidates)

        # Schritt 5: Ressource blockieren
        finish_time = real_time + timedelta(seconds=duration)
        self.busy_until[selected_resource] = finish_time

        return selected_resource