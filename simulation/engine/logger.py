"""
Event Logger Module
Handles logging of simulation events to CSV format
"""

import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

import pm4py
from pm4py.objects.log.util import dataframe_utils


class EventLogger:
    """Logs simulation events to CSV or XES format"""

    def __init__(self, filepath: str, start_time: Optional[datetime] = None):
        self.filepath = filepath
        self.start_time = start_time or datetime.now()
        self.events = []

    def log_event(
        self,
        case_id: str,
        activity: str,
        timestamp: float,
        resource: Optional[str] = None,
        lifecycle: str = "complete",
    ):
        """
        Log a single event

        Args:
            case_id: Unique identifier for the process instance
            activity: Name of the activity
            timestamp: Simulation time when event occurred
            resource: Resource that executed the activity (optional)
            lifecycle: Event lifecycle state (start, complete, etc.)
        """
        real_timestamp = self.start_time + timedelta(seconds=timestamp)
        self.events.append(
            {
                "case_id": case_id,
                "activity": activity,
                "timestamp": real_timestamp,
                "resource": resource,
                "lifecycle": lifecycle,
            }
        )

    def write_to_csv(self):
        """Write all logged events to CSV file"""
        if not self.events:
            print("Warning: No events to write")
            return

        df = pd.DataFrame(self.events)
        df = df.sort_values("timestamp")  # Chronological order
        df.to_csv(self.filepath, index=False)
        print(f"Event log written to {self.filepath} ({len(self.events)} events)")

    def write_to_xes(self, xes_path: Optional[str] = None):
        """Write all logged events to XES file"""
        if not self.events:
            print("Warning: No events to write")
            return
        df = pd.DataFrame(self.events)
        df = df.sort_values("timestamp")  # Chronological order

        # Map your columns to XES standard attribute keys
        df_xes = df.rename(
            columns={
                "case_id": "case:concept:name",
                "activity": "concept:name",
                "timestamp": "time:timestamp",
                "resource": "org:resource",
                "lifecycle": "lifecycle:transition",
            }
        )
        # Ensure timestamp column is proper datetime for pm4py
        df_xes = dataframe_utils.convert_timestamp_columns_in_df(df_xes)

        # Convert to pm4py event log
        event_log = pm4py.convert_to_event_log(df_xes)

        if xes_path is None:
            xes_path = self.filepath.replace(".csv", ".xes")

        # Write XES file
        pm4py.write_xes(event_log, xes_path)
        print(f"Event log written to {xes_path} ({len(self.events)} events)")

    def clear(self):
        """Clear all logged events"""
        self.events = []
