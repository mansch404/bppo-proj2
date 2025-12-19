import pandas as pd
import numpy as np
import pm4py
from scipy import stats
from typing import Dict, Tuple
import random


class DistributionMiner:
    def __init__(self, log_path: str, percentile_cutoff: float = 0.25):
        """
        Args:
            log_path: Pfad zur XES Datei
            percentile_cutoff: 0.25 bedeutet, wir nutzen nur die schnellsten 25% der Daten
                               für das Lernen der Bearbeitungszeit (um Wartezeiten rauszufiltern).
        """
        self.log_path = log_path
        self.cutoff = percentile_cutoff
        self.activity_distributions = {}
        # Fallback für Aktivitäten ohne Daten (Standard: 10 Minuten)
        self.default_distribution = {"type": "fixed", "value": 600}

    def analyze_log(self):
        """Lädt das Log und lernt die Verteilungen"""
        print(f"Loading log from {self.log_path}...")

        # Log in DataFrame konvertieren
        log = pm4py.read_xes(self.log_path)
        df = pm4py.convert_to_dataframe(log)

        # Sortieren
        df = df.sort_values(['case:concept:name', 'time:timestamp'])

        # Dauer zum VORGÄNGER-Event berechnen (Cycle Time)
        df['prev_timestamp'] = df.groupby('case:concept:name')['time:timestamp'].shift(1)
        df['duration'] = (df['time:timestamp'] - df['prev_timestamp']).dt.total_seconds()

        # Bereinigung: Keine NaNs, keine negativen Werte
        df = df.dropna(subset=['duration'])

        # Grobes Data Cleaning: Alles über 2 Tage raus (Wochenende/Systemfehler)
        df = df[df['duration'] < 172800]

        print(f"Fitting distributions using the fastest {self.cutoff * 100}% of cases (Best-Case approach)...")
        self._fit_distributions(df)

    def _fit_distributions(self, df: pd.DataFrame):
        """Fittet eine Verteilung basierend auf den schnellsten Ausführungen"""
        activities = df['concept:name'].unique()

        for activity in activities:
            # 1. Alle Dauern für diese Aktivität holen
            raw_durations = df[df['concept:name'] == activity]['duration'].values

            # Einfaches Cleaning: Alles unter 1 Sekunde ist oft technisches Logging -> raus
            raw_durations = raw_durations[raw_durations > 1.0]

            if len(raw_durations) < 5:
                self.activity_distributions[activity] = self.default_distribution
                continue

            # Wir berechnen den Grenzwert (z.B. das 25. Perzentil)
            # Das bedeutet: 25% der Fälle waren schneller als dieser Wert.
            # Wir nehmen an: Diese Fälle hatten KEINE Wartezeit.
            limit = np.quantile(raw_durations, self.cutoff)  # z.B. 0.25

            # Wir filtern die Daten: Nur die Werte behalten, die <= Limit sind
            pure_processing_times = raw_durations[raw_durations <= limit]

            # Sicherheitscheck, falls durch das Filtern nichts übrig bleibt (unwahrscheinlich)
            if len(pure_processing_times) < 2:
                pure_processing_times = raw_durations  # Fallback auf alle Daten

            # Jetzt fitten wir die Verteilung NUR auf diese "bereinigten" Zeiten
            mu, std = stats.norm.fit(pure_processing_times)

            # Das Minimum der beobachteten "schnellen" Zeiten speichern wir als Untergrenze
            min_observed = np.min(pure_processing_times)

            self.activity_distributions[activity] = {
                "type": "normal",
                "mean": mu,
                "std": std,
                "min": min_observed
            }

            # Optional: Print zur Kontrolle, wie stark sich die Zeiten unterscheiden
            # full_mean = np.mean(raw_durations)
            # print(f"  {activity}: Raw Mean={full_mean:.1f}s -> Pure Processing Mean={mu:.1f}s")

        print(f"Learned distributions for {len(self.activity_distributions)} activities.")

    def get_processing_time(self, activity_name: str) -> float:
        """Gibt eine Zeit zurück (wird von Engine genutzt)"""
        dist = self.activity_distributions.get(activity_name, self.default_distribution)

        if dist["type"] == "fixed":
            return dist["value"]

        if dist["type"] == "normal":
            # Ziehe aus der Verteilung der "reinen Bearbeitungszeit"
            val = random.gauss(dist["mean"], dist["std"])

            # Wichtig: Nicht unter das absolute Minimum fallen
            return max(dist["min"], val)

        return 10.0