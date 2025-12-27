from collections import defaultdict
import pandas as pd
import fitter
import warnings
import numpy as np


# Spawner helper functions
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

def delta_timestamps_in_seconds(list_of_arrivals):

    df_arrivals = pd.DataFrame()
    df_arrivals.insert(0, 'time:timestamp', list_of_arrivals)
    df_arrivals['time:timestamp'] = pd.to_datetime(df_arrivals['time:timestamp'])

    deltas = df_arrivals.diff()

    deltas_in_seconds = deltas['time:timestamp'].dt.total_seconds()
    deltas_in_seconds = deltas_in_seconds.dropna()

    return deltas_in_seconds

def get_inter_arrival_times_from_list_of_timestamps(arrival_times):
    current_day = arrival_times[0].strftime('%Y-%m-%d')
    # Compute durations between one arrival and the next one (inter-arrival durations)
    new_day = []
    inter_arrival_durations = []
    last_arrival = None
    for arrival in arrival_times:
        if last_arrival:
            if arrival.strftime('%Y-%m-%d') == current_day:
                inter_arrival_durations += [(arrival - last_arrival).total_seconds()]
            else:
                new_day.append(arrival)
        last_arrival = arrival
        current_day = arrival.strftime('%Y-%m-%d')

    return inter_arrival_durations

def find_best_fitting_distribution(deltas_in_minutes):

    list_of_distibutions = ['gamma',
                            'lognorm',
                            'expon',
                            'norm'
                            ]

    f = fitter.Fitter(deltas_in_minutes, distributions=list_of_distibutions)
    f.fit()

    best_dist = f.get_best(method='sumsquare_error')

    return best_dist

def extract_timestamps_per_case(df):
    """
    Args:
        df: event-log as pandas DataFrame

    Returns: list of first timestamp for each case
    """
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
    arrival_times = []
    for _, events in df.groupby('case:concept:name'):
        arrival_times += [events['time:timestamp'].min()]

    arrival_times.sort()
    return arrival_times


def transform_to_float(arrival_times):
    """
    Transforms arrival timestamps into floats representing hours since midnight,
    grouped by date. Prints a message if any time float exceeds 24 hours.

    Args:
    arrival_times : list of pd.Timestamp. List of arrival timestamps.
    Returns:
    grouped_timestamps_floats : List of lists where each sublist contains time floats for a specific date.
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

def silvermans_rule(data, weights=None):
    """
    Returns optimal smoothing (standard deviation) if the data is close to
    normal.

    Examples
    --------
    data = np.arange(9).reshape(-1, 1)
    ans = silvermans_rule(data)
    assert np.allclose(ans, 1.8692607078355594)
    """
    if not len(data.shape) == 2:
        raise ValueError("Data must be of shape (obs, dims).")
    obs, dims = data.shape
    if not dims == 1:
        raise ValueError("Silverman's rule is only available for 1D data.")

    if weights is not None:
        warnings.warn("Silverman's rule currently ignores all weights")

    if obs == 1:
        return 1
    if obs < 1:
        raise ValueError("Data must be of length > 0.")

    sigma = np.std(data, ddof=1)
    # scipy.stats.norm.ppf(.75) - scipy.stats.norm.ppf(.25) -> 1.3489795003921634
    IQR = (np.percentile(data, q=75) - np.percentile(data, q=25)) / 1.3489795003921634

    sigma = min(sigma, IQR)

    # The logic below is not related to silverman's rule, but if the data is constant
    # it's nice to return a value instead of getting an error. A warning will be raised.
    if sigma > 0:
        return sigma * (obs * 3 / 4.0) ** (-1 / 5)
    else:
        # stats.norm.ppf(.99) - stats.norm.ppf(.01) = 4.6526957480816815
        IQR = (np.percentile(data, q=99) - np.percentile(data, q=1)) / 4.6526957480816815
        if IQR > 0:
            bw = IQR * (obs * 3 / 4.0) ** (-1 / 5)
            warnings.warn(
                "Silverman's rule failed. Too many idential values. \
Setting bw = {}".format(
                    bw
                )
            )
            return bw

        # Here, all values are basically constant
        warnings.warn("Silverman's rule failed. Too many idential values. Setting bw = 1.0")
        return 1.0