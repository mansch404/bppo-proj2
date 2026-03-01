import logging
from pathlib import Path
from markdown_it.rules_block import reference
from simulation.spawner.arrivals_segmentation import tune_sensitivity, extend_pattern, get_timeframe_years
import pandas as pd
import numpy as np
import pickle
import copy
import math
from simulation.spawner.KDE_DataSimulator import KDE_DataSimulator
from utils.helper import delta_timestamps_in_seconds, extract_timestamps_per_case, transform_to_float, get_inter_arrival_times_from_list_of_timestamps, _read_event_log
from scipy.stats import wasserstein_distance



class DynamicSpawner_KDE():
    def __init__(self, arrival_times=None, float_format=False):
        logging.basicConfig(level=logging.INFO, format='%(filename)s:%(lineno)d - %(message)s')
        self.logger = logging.getLogger(__name__)
        if self.train_set != None:
            self.train_set = arrival_times.copy()  # Train set or whole set
        self.float_format = float_format

    def generate_next(self):
        return None

    def generate_arrivals(self, start_time, end_time):

        ## -- AT_KDE Segmentation -- ##
        def _setup_clustered_train_dict(train, prediction_start_t, prediction_end_t, verbose=None):
            """
            in:
                train: list[Timestamp]
                prediction_start_t: [Timstamp] start timestamp of domain that bw is validated on
                prediction_end_t: [Timstamp] end timestamp of domain that bw is validated on
            out:
                output_df: [pd.DataFrame] representing the predicted cluster for each date in the test set
                clustered_train_dict: constructs a dict of data, key: global cluster (int), value: corresponding data as list[Timestamp(...)]

            some additional info:
            segments_tuned is a list of lists, where each sublist is a segment of consecutive days
            labels is a list of labels, where each label is the cluster number for the corresponding segment
            """

            years = get_timeframe_years(train)
            segments_tuned, status_finished, labels = tune_sensitivity(train)

            self.logger.info(f'status_finished: {status_finished}') if verbose is not None else None
            self.logger.info(f'labels: {labels}') if verbose is not None else None

            output_df, segment_flag = extend_pattern(
                train,
                prediction_start_t,  # comes from generate_arrivals
                prediction_end_t,  # comes from generate_arrivals
                segments_tuned,
                labels,
                years
            )

            if segment_flag:
                # segment_flag == True means: no trust in very last (too-short) segment; continue the previous cluster instead
                # last segment now becomes original last and the one preceeding merged together
                new_last_segment = segments_tuned[-2] + segments_tuned[-1]
                segments_tuned = segments_tuned[:-2]
                segments_tuned.append(new_last_segment)

                faulty_segment_cl = labels[-1]
                replacement_segment_cl = labels[-2]
                labels = labels[:-1]  # remove the faulty_segment_cluster_label for construction of clustered_train_dict

                # should typically only affect the inital test value since it is both shared by train and test
                output_df['predicted_cluster'] = (
                    output_df['predicted_cluster']
                    .apply(
                        lambda c: replacement_segment_cl if c == faulty_segment_cl else c
                    )
                )
                self.logger.info(f'output_df after segment_flag: {output_df}')

            clustered_train_dict = {}
            for label, corresponding_timestamps in zip(labels, segments_tuned):
                # we do not want to overwrite the timestamps of one cluster if multiple segments of that cluster exist
                if label in clustered_train_dict:
                    # Extend existing list with new timestamps
                    clustered_train_dict[label].extend(corresponding_timestamps)
                else:
                    # Create new entry
                    clustered_train_dict[label] = corresponding_timestamps
            return output_df, clustered_train_dict

        def run_bandwidth_optimisation(train, cluster_values):
            """
            Args:
                train: list of Timestamp (list[Timestamp(...)])
                cluster_values:
            """

            train_bw = train
            val_bw = train

            output_df = pd.DataFrame(
                                        data=cluster_values,
                                        index = pd.date_range(val_bw[0].date(), val_bw[-1].date()),
                                        columns = ['predicted_cluster']
                                    ).reset_index().rename(columns = {'index':'date'})

            clustered_train_dict = {}
            clustered_train_dict[cluster_values] = train_bw

            train_df_clustered = (
                pd.Series(clustered_train_dict)      # index = cluster, values = list of dates
                .explode()                         # one row per (cluster, date)
                .rename_axis("cluster")
                .reset_index(name="date")
                .sort_values("date", ignore_index=True)
            )

            # Run optimization of bandwidth parameter
            bw_factor_dict = {}
            bw_smooth_factors = [199, 149, 124, 99, 74, 49, 24, 9, 4, 2, 0.5, 0.0, -0.1, -0.25, -0.5, -0.75, -0.85, -0.99]

            best_emd = float('inf')
            best_bw_factor = None
            bw_emd_dict = {bw: 0 for bw in bw_smooth_factors}
            for factor in bw_smooth_factors:
                bw_factor_dict[list(clustered_train_dict.keys())[0]] = factor
                ds_class = KDE_DataSimulator(
                    reference_dataset=train_df_clustered,
                    train_clustered=clustered_train_dict,
                    test_cluster_estim=output_df,
                    bw_factor_dict=bw_factor_dict
                    )
                simulated_data, _ = ds_class.sample_kde(start_time=val_bw[0], end_time=val_bw[-1])

                # Evaluate validation performance
                validation_data = pd.to_datetime(val_bw)
                emd = evaluate_validation_performance(simulated_data, validation_data)
                # self.logger.info(f'EMD for bw_smooth_factor {factor}: {emd}')

                bw_emd_dict[factor] = emd
                if emd < best_emd:
                    best_bw_factor = factor
                    best_emd = emd

                # self.logger.info(f'best_emd: {best_emd}')
                # self.logger.info(f'best_bw_factor: {best_bw_factor}')
            return best_emd, bw_emd_dict, best_bw_factor

        output_df, clustered_train_dict = _setup_clustered_train_dict(self.train_set.copy(), start_time, end_time)

        # Uncomment for -> Diagnostics to access globally set clusters
        # self.logger.info('Saving output_df to csv...')
        # output_df.to_csv('output_df.csv')
        # self.logger.info('Complete.')

        self.logger.info(f'number of observations per cluster: {output_df.groupby("predicted_cluster").count()}')
        train_df_clustered = (
            pd.Series(clustered_train_dict)  # index = cluster, values = list of dates
            .explode()  # one row per (cluster, date)
            .rename_axis("cluster")
            .reset_index(name="date")
            .sort_values("date", ignore_index=True)
        )

        optimal_bandwidths_per_global_cluster = {}

        for gc in clustered_train_dict:
            self.logger.info(f'Optimize bandwidth for global cluster {gc}..')
            _, _, best_bw_factor_gc = run_bandwidth_optimisation(clustered_train_dict[gc].copy(), gc)
            optimal_bandwidths_per_global_cluster[gc] = best_bw_factor_gc

        self.logger.info(f'Optimization complete.')
        # simulate arrivals with
        # KDE
        self.logger.info('Generating Arrivals now...')
        ds_class = KDE_DataSimulator(
            reference_dataset=train_df_clustered,
            train_clustered=clustered_train_dict,
            test_cluster_estim=output_df,
            bw_factor_dict=optimal_bandwidths_per_global_cluster
        )

        simulated_data, _ = ds_class.sample_kde(start_time=start_time, end_time=end_time)
        self.logger.info('Complete.')
        return simulated_data

    def fit_with_event_log_path(self, path: str):
        path = Path(path)
        log = _read_event_log(path)
        timestamps_list = extract_timestamps_per_case(log)
        train, test = split_arrival_times(timestamps_list, threshold=0.8)

        self.train_set = train.copy()


def store_arrivals(self, sim_case_arrivals, start_time, end_time, train, test, logger, ref):
    full_dataset = train + test
    idx = pd.to_datetime(full_dataset, utc=True)
    start = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)
    sim_period = idx[(idx >= start) & (idx <= end)]

    logger.info(f"Number of simulated arrivals: {len(sim_case_arrivals)}")
    logger.info(f"Number of reference arrivals: {len(sim_period)}")

    filename = f"simulated_arrivals_{ref}.csv"

def evaluate_validation_performance(simulated_data, val_data):
    """
    Compute the wasserstein distance between the inter-arrival times of the validation set and the simulated data
    """

    if len(simulated_data) == 0:
        sim_data_for_distance = []
    else:
        sim_data_for_distance = get_inter_arrival_times_from_list_of_timestamps(simulated_data)
    test_data_for_distance = get_inter_arrival_times_from_list_of_timestamps(val_data)

    if len(sim_data_for_distance) == 0:
        emd_iat = np.inf
    else:
        emd_iat = wasserstein_distance(test_data_for_distance, sim_data_for_distance)

    return np.sqrt(emd_iat)

def split_arrival_times(list_of_timestamps, threshold=0.8):
    """
    Perform temporal hold out split with threshold % training and (1-threshold)% testing.

    Args:
      list_of_timestamps (list): The generated timestamps.

    Returns:
      train (list): The timestamps of the train set.
      test (list): The timestamps of the test set.
    """
    arrival_times = list_of_timestamps
    arrival_times.sort()

    number_times = (len(arrival_times))
    train_size = int(threshold * number_times)

    train = arrival_times[:train_size]
    test = arrival_times[train_size:]

    return train, test



def clustered_arrival_table(arrival_times):
    """
    Args:
        arrival_times: list of timestamps

    Returns: A table with timestamps segmented by day and hour

    """
    df = pd.DataFrame({'Timestamp': arrival_times})

    df['Timestamp'] = pd.to_datetime(df['Timestamp']) # Ensure column is datetime type

    df['Day_name'] = df['Timestamp'].dt.day_name() # Saves the name of the day of the week as a string
    df['Day_index'] = df['Timestamp'].dt.dayofweek # Saves the day of the week as integer range [0,7)

    df['Hour'] = df['Timestamp'].dt.hour

    return df



if __name__ == '__main__':
    ## Testing ##

    # 1. Load log
    file_name = r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\data\event_log\data_log.pkl"
    log = 0
    with open(file_name, "rb") as f:
        log = pickle.load(f)

    # 2. Split dataset and extract list of arrivals
    timestamps_list = extract_timestamps_per_case(log)
    train, test = split_arrival_times(timestamps_list, threshold=0.8)

    start_date = timestamps_list[0].date()
    end_date = timestamps_list[-1].date()

    kde_spawner = DynamicSpawner_KDE(train)

    print(timestamps_list[0])

    # Generate arrivals
    case_arrivals_times = kde_spawner.generate_arrivals(start_date, end_date) # List of timestamps
    simulated_delays = delta_timestamps_in_seconds(case_arrivals_times) # List of delays




"""
Acknowledgement:
    This implementation is based on/adapted from the research paper:
    "A Divide-and-Conquer Approach for Modeling Arrival Times in Business Process Simulation"
    by Lukas Kirchdorfer, Konrad Özdemir, Stjepan Kusenic, Han van der Aa, and Heiner Stuckenschmidt (2025).

    Original Code Repository: https://github.com/konradoezdemir/AT-KDE
    Paper DOI: https://doi.org/10.1007/978-3-032-02867-9_20
"""