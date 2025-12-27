import datetime
import math
import pickle

import pandas as pd
from scipy.stats import wasserstein_distance

from simulation.spawner.config import EventLogIDs, DistanceMetric, discretize_to_hour
from utils.earth_movers_distance import earth_movers_distance
from utils.helper import extract_timestamps_per_case

'''
Acknowledgements: https://github.com/AutomatedProcessImprovement/log-distance-measures.git

Benchmarking different simulated arrivals with AT-KDE 
'''

def case_arrival_distribution_distance(
        original_arrivals_list,
        simulated_arrivals_list,
        discretize_event=discretize_to_hour,  # function to discretize a total amount of seconds into bins
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of case arrival of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize_instance] (default by hour).


    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param discretize_event: function to discretize the total amount of seconds each timestamp represents, default to hour.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the case arrival distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get arrival events of each log
    original_arrivals = pd.DataFrame({'Timestamp': original_arrivals_list})
    simulated_arrivals = pd.DataFrame({'Timestamp': simulated_arrivals_list})

    original_arrivals['Timestamp'] = pd.to_datetime(original_arrivals['Timestamp'])
    simulated_arrivals['Timestamp'] = pd.to_datetime(simulated_arrivals['Timestamp'], format='%d.%m.%Y %H:%M:%S')

    simulated_arrivals['Timestamp'] = pd.to_datetime(simulated_arrivals['Timestamp']).dt.tz_localize(None)
    original_arrivals['Timestamp'] = pd.to_datetime(original_arrivals['Timestamp']).dt.tz_localize(None)
    # Get the first arrival to normalize
    first_arrival = min(
        original_arrivals['Timestamp'].min(),
        simulated_arrivals['Timestamp'].min()
    ).floor(freq='h')
    # Discretize each event to its corresponding "bin"
    original_discrete_arrivals = [
        discretize_event(difference.total_seconds())
        for difference in (original_arrivals['Timestamp'] - first_arrival)
    ]
    simulated_discrete_arrivals = [
        discretize_event(difference.total_seconds())
        for difference in (simulated_arrivals['Timestamp'] - first_arrival)
    ]
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(original_discrete_arrivals, simulated_discrete_arrivals) / len(original_discrete_arrivals)
    else:
        distance = wasserstein_distance(original_discrete_arrivals, simulated_discrete_arrivals)
    # Normalize if needed
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(original_discrete_arrivals), max(simulated_discrete_arrivals))
        distance = distance / max_value if max_value > 0 else distance
    # Return metric
    return distance

if __name__ == '__main__':
    # 1. Load log
    file_name = r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\data\event_log\data_log.pkl"
    log = 0
    with open(file_name, "rb") as f:
        log = pickle.load(f)

    original_timestamps_list = extract_timestamps_per_case(log)

    #print(original_timestamps_list)
    # Convert simulation json files to list
    # lstm
    simulated_timestamps_list_lstm = pd.read_json(r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\event_log_simulations_from_atkde\lstm_prob\bpi17\simulated_run_1.json")
    simulated_timestamps_list_lstm.columns = ['Timestamp']

    simulated_arrival_times = []
    for tm in simulated_timestamps_list_lstm['Timestamp']:
        simulated_arrival_times.append(tm)

    simulated_arrival_times.sort()

    # atkde
    simulated_timestamps_list_kde = pd.read_json(r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\event_log_simulations_from_atkde\at_kde_prob\bpi17\simulated_run_2.json")
    simulated_timestamps_list_kde.columns = ['Timestamp']

    simulated_arrival_times_kde = []
    for tm in simulated_timestamps_list_kde['Timestamp']:
        simulated_arrival_times_kde.append(tm)

    simulated_arrival_times_kde.sort()

    # exponential
    simulated_timestamps_list_expon = pd.read_json(r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\event_log_simulations_from_atkde\exponential_prob\bpi17\simulated_run_1.json")
    simulated_timestamps_list_expon.columns = ['Timestamp']

    simulated_arrival_times_expon = []
    for tm in simulated_timestamps_list_expon['Timestamp']:
        simulated_arrival_times_expon.append(tm)
    simulated_arrival_times_expon.sort()

    # prophet
    simulated_timestamps_list_prophet = pd.read_json(r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\event_log_simulations_from_atkde\prophet\bpi17\simulated_run_1.json")
    simulated_timestamps_list_prophet.columns = ['Timestamp']

    simulated_arrival_times_prophet = []
    for tm in simulated_timestamps_list_prophet['Timestamp']:
        simulated_arrival_times_prophet.append(tm)
    simulated_arrival_times_prophet.sort()

    #print(simulated_arrival_times)
    metric1 = case_arrival_distribution_distance(original_timestamps_list, simulated_arrival_times)
    metric2 = case_arrival_distribution_distance(original_timestamps_list, simulated_arrival_times_kde)
    metric3 = case_arrival_distribution_distance(original_timestamps_list, simulated_arrival_times_expon)
    metric4 = case_arrival_distribution_distance(original_timestamps_list, simulated_arrival_times_prophet)

    print('Comparison with lstm: ', metric1)
    print('Comparison with kde: ', metric2)
    print('Comparison with exponential: ', metric3)
    print('Comparison with prophet: ', metric4)





