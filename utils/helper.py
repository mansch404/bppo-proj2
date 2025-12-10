from collections import defaultdict
import pandas as pd

def transform_to_float(arrival_times):
    """
    Transforms arrival timestamps into floats representing hours since midnight,
    grouped by date. Prints a message if any time float exceeds 24 hours.

    Parameters:
    -----------
    arrival_times : list of pd.Timestamp
        List of arrival timestamps.

    Returns:
    --------
    grouped_timestamps_floats : list of lists
        List of lists where each sublist contains time floats for a specific date.
    """
    # Ensure all timestamps are timezone-aware and in UTC
    arrival_times = [
        ts.tz_convert('UTC') if ts.tzinfo else ts.tz_localize('UTC')
        for ts in arrival_times
    ]

    # Dictionary to hold lists of timestamps grouped by day
    grouped_by_day = defaultdict(list)

    for timestamp in arrival_times:
        # Extract date in UTC
        date_str = timestamp.strftime('%Y-%m-%d')
        grouped_by_day[date_str].append(timestamp)

    # Convert timestamps to floats representing hours since midnight
    grouped_timestamps_floats = []
    for date, times in grouped_by_day.items():
        time_floats = []
        for time_obj in times:
            # Calculate time difference from midnight
            midnight = pd.Timestamp(date + ' 00:00:00', tz='UTC')
            time_delta = (time_obj - midnight).total_seconds() / 3600  # Convert seconds to hours

            # Handle negative time differences (if any)
            if time_delta < 0:
                time_delta += 24  # Adjust for times after midnight

            # Print statement if time_delta >= 24
            if time_delta >= 24:
                print(f"Time float exceeds 24 hours: {time_delta} for timestamp {time_obj}")
                # Cap time_float at maximum valid time before midnight
                time_delta = 23 + 59 / 60 + 59 / 3600 + 999999 / 1e6  # Approximately 23.9999997222 hours

            time_floats.append(time_delta)
        grouped_timestamps_floats.append(time_floats)

    return grouped_timestamps_floats
