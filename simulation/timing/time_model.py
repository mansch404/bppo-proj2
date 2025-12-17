import pandas as pd
import numpy as np
import random
from typing import Dict, Tuple, Optional


class TimeModel:
    def __init__(self):
        # Stores (mean, std_dev) for each activity
        # Example: {'A_Create Application': (120.5, 30.2)}
        self.activity_distributions: Dict[str, Tuple[float, float]] = {}
        self.default_mean = 60.0
        self.default_std = 10.0

    def train_on_log(self, csv_path: str):
        """
        Fits probability distributions on historical data.
        Expects a CSV with columns: 'case_id', 'activity', 'timestamp', 'lifecycle'
        """
        print(f"Training Time Model from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)

            # Ensure timestamps are datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # We need to calculate duration.
            # If log has 'start' and 'complete' rows:
            if 'lifecycle' in df.columns:
                starts = df[df['lifecycle'].str.lower() == 'start'].set_index(['case_id', 'activity'])['timestamp']
                ends = df[df['lifecycle'].str.lower() == 'complete'].set_index(['case_id', 'activity'])['timestamp']
                durations = (ends - starts).dt.total_seconds()

                # Group by activity and calculate stats
                stats = durations.groupby('activity').agg(['mean', 'std'])

                for activity, row in stats.iterrows():
                    mean = row['mean'] if not pd.isna(row['mean']) else self.default_mean
                    std = row['std'] if not pd.isna(row['std']) else self.default_std
                    self.activity_distributions[activity] = (mean, std)

            else:
                # Fallback if log only has 'complete' timestamps (Estimate based on previous event)
                print("Warning: Log lacks lifecycle info. Using default distributions.")

            print(f"Time Model trained on {len(self.activity_distributions)} activities.")

        except FileNotFoundError:
            print("Warning: Log file not found. Using default distributions.")
        except Exception as e:
            print(f"Error training time model: {e}")

    def get_predicted_time(self, activity_name: str, context_data: Optional[dict] = None) -> float:
        """
        Returns a random sample from the learned distribution.
        """
        if activity_name in self.activity_distributions:
            mean, std = self.activity_distributions[activity_name]
            # Log-Normal is often better for process times (no negative values),
            # but Normal is acceptable for Basic requirements.
            # We ensure non-negative by taking max(0.1, value)
            duration = random.gauss(mean, std)
        else:
            duration = random.gauss(self.default_mean, self.default_std)

        return max(1.0, duration)  # Duration must be positive