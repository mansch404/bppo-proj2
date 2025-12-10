import logging
from simulation.spawner.arrivals_segmentation import tune_sensitivity, extend_pattern, get_timeframe_years
import pandas as pd
import fitter
import pickle
import scipy.stats as stats

"""
DynamicSpawner
------------------
This module implements some of the 'Divide-and-Conquer' strategy for dynamic arrival rate generation. 
Specifically on the divide phase, for global segmentation.

Acknowledgement:
    This implementation is based on/adapted from the research paper:
    "A Divide-and-Conquer Approach for Modeling Arrival Times in Business Process Simulation"
    by Lukas Kirchdorfer, Konrad Özdemir, Stjepan Kusenic, Han van der Aa, and Heiner Stuckenschmidt (2025).

    Original Code Repository: https://github.com/konradoezdemir/AT-KDE
    Paper DOI: https://doi.org/10.1007/978-3-032-02867-9_20
"""

class DynamicSpawner():

    def __init__(self, arrival_times):
        logging.basicConfig(level=logging.INFO, format='%(filename)s:%(lineno)d - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.train_set = arrival_times.copy() # Train set or whole set

    def generate_next(self):
        return None

    def generate_arrivals(self, start_time, end_time):

        ## -- AT_KDE -- ##
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

        output_df, clustered_train_dict = _setup_clustered_train_dict(self.train_set.copy(), start_time, end_time)

        ## Uncomment for seeing the global segmentation
        #self.logger.info('Saving output_df to csv...')
        #output_df.to_csv('output_df.csv')
        #self.logger.info('Complete.')
        #self.logger.info(f'number of observations per cluster:\n {output_df.groupby("predicted_cluster").count()}\n')
        #self.logger.info(f"number of clusters: {len(clustered_train_dict)}")

        train_df_clustered = (
            pd.Series(clustered_train_dict)  # index = cluster, values = list of dates
            .explode()  # one row per (cluster, date)
            .rename_axis("cluster")
            .reset_index(name="date")
            .sort_values("date", ignore_index=True)
        )



        ## TODO
        ## Here comes the training/optimisation part for the decided ML model either KDE or new ideas

        return 0

def extract_timestamps_per_case(df):
    '''
    :param df: event-log as pandas DataFrame
    :return: list of first timestamp for each case
    '''
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    arrival_times = []
    for _, events in df.groupby('case:concept:name'):  # CHANGED
        arrival_times += [events['time:timestamp'].min()]

    arrival_times.sort()
    return arrival_times

def delta_timestamps_in_minutes(list_of_arrivals):

    df_arrivals = pd.DataFrame()
    df_arrivals.insert(0, 'time:timestamp', list_of_arrivals)
    df_arrivals['time:timestamp'] = pd.to_datetime(df_arrivals['time:timestamp'])

    deltas = df_arrivals.diff()

    deltas_in_minutes = deltas['time:timestamp'].dt.total_seconds() / 60.0
    deltas_in_minutes = deltas_in_minutes.dropna()

    return deltas_in_minutes

## Functions for the segmentation process
#-- From AT-KDE -- #

if __name__ == '__main__':

    # 1. Load log
    file_name = r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\data\event_log\data_log.pkl"
    log = 0
    with open(file_name, "rb") as f:
        log = pickle.load(f)

    # 2. Extract list of arrivals
    timestamps_list = extract_timestamps_per_case(log)

    start_date = timestamps_list[0].date()
    end_date = timestamps_list[-1].date()


    print(start_date, end_date)

    dynamic_spawner = DynamicSpawner(arrival_times=timestamps_list)
    dynamic_spawner.generate_arrivals(start_date, end_date)
