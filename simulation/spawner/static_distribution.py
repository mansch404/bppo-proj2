import pandas as pd
import logging
import pickle
import scipy.stats as stats

from simulation.spawner.arrivals_segmentation import tune_sensitivity, extend_pattern, get_timeframe_years
from utils.helper import delta_timestamps_in_seconds, find_best_fitting_distribution, extract_timestamps_per_case
from datetime import datetime, timedelta


class StaticSpawner:
    def __init__(self):
        self.best_dist_name = []
        self.best_params = []
        self.dist_object = []
        self.segmentation = False
        self.train_df_clustered = None

    def fit(self, event_log, segmentation=False):
        """
        Args:
            event_log: Event log as a pandas DataFrame
            segmentation: use global segmentation for the event log [bool]
        Returns: None

        """

        self.segmentation = segmentation
        list_timestamps = extract_timestamps_per_case(event_log)


        if segmentation:
            start_date = timestamps_list[0].date()
            end_date = timestamps_list[-1].date()
            output_df, clustered_train_dict = _setup_clustered_train_dict(list_timestamps, start_date, end_date)

            self.train_df_clustered = (
                pd.Series(clustered_train_dict)  # index = cluster, values = list of dates
                .explode()  # one row per (cluster, date)
                .rename_axis("cluster")
                .reset_index(name="date")
                .sort_values("date", ignore_index=True)
            )
            best_dist = []
            for cluster in range(len(self.train_df_clustered['cluster'].unique())):
                current_deltas_seconds =  delta_timestamps_in_seconds(self.train_df_clustered[self.train_df_clustered['cluster'] == cluster]['date'])
                current_best_dist = find_best_fitting_distribution(current_deltas_seconds)
                best_dist.append(current_best_dist)

            for i, best_d in enumerate(best_dist):
                dist_name = list(best_d.keys())[0]
                self.best_dist_name.append(dist_name)
                self.best_params.append(best_d[dist_name])
                self.dist_object.append(getattr(stats, dist_name))


        else:

            deltas_seconds = delta_timestamps_in_seconds(list_timestamps)
            best_dist = find_best_fitting_distribution(deltas_seconds)

            self.best_dist_name = list(best_dist.keys())[0]
            self.best_params.append(best_dist[self.best_dist_name])
            self.dist_object = getattr(stats, self.best_dist_name)

            print(f"Best distribution: {self.best_dist_name}")

    def generate_next(self, current_dist: int = 0):
        """
        Compute arrivals based on selected distribution.

        Returns: [float] Time of the next arrival in minutes.
        """
        if self.dist_object is None:
            raise ValueError("Model not trained! Use fit() first.")

        if self.segmentation:
            val = self.dist_object[current_dist].rvs(**self.best_params[current_dist]) # Distanz zwischen nächstes Arrival
            return max(0.0, val)

        else:
            val = self.dist_object.rvs(**self.best_params[0]) # Distanz zwischen nächstes Arrival
            return max(0.0, val)

    def generate_arrivals(self, star_date, end_date):
        """
        Args:
            star_date: [datetime] the start date of the simulation
            end_date: [datetime] the end date of the simulation

        Returns: Generated arrival times as a list of [datetime].
        """

        generated_arrivals = [star_date + timedelta(seconds=self.generate_next())] # First arrival
        print("Generating arrivals...")

        if self.segmentation:

            clustered_df = self.train_df_clustered.copy()
            clustered_df['date'] = pd.to_datetime(clustered_df['date'])
            grouped_dates = clustered_df.groupby('cluster')['date'].agg(['min', 'max']).reset_index()
            grouped_dates.columns = ['cluster','start_date', 'end_date']

            for i in range(len(grouped_dates)):
                star_date = grouped_dates['start_date'][i]
                end_date = grouped_dates['end_date'][i]
                while generated_arrivals[-1] < end_date:
                    generated_arrivals.append(generated_arrivals[-1] + timedelta(seconds=self.generate_next(current_dist=i)))

            return generated_arrivals

        else:

            while generated_arrivals[-1] < end_date:
                generated_arrivals.append(generated_arrivals[-1] + timedelta(seconds=self.generate_next()))

            return generated_arrivals


''' Segmentation of arrivals '''
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
    logging.basicConfig(level=logging.INFO, format='%(filename)s:%(lineno)d - %(message)s')
    logger = logging.getLogger(__name__)
    years = get_timeframe_years(train)
    segments_tuned, status_finished, labels = tune_sensitivity(train)

    logger.info(f'status_finished: {status_finished}') if verbose is not None else None
    logger.info(f'labels: {labels}') if verbose is not None else None

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
        logger.info(f'output_df after segment_flag: {output_df}')

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

def testing():

    file_name = r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\data\event_log\data_log.pkl"
    log = 0
    with open(file_name, "rb") as f:
        log = pickle.load(f)
    log

    timestamps_list = extract_timestamps_per_case(log)
    #print(timestamps_list)

    deltas_timestamps_minutes_df = delta_timestamps_in_minutes(timestamps_list)
    #print(deltas_timestamps_minutes_df)

    print("Starting fitting...")
    f.fit()
    print(f.get_best(method='sumsquare_error'))




if __name__ == '__main__':

    ## TESTING ##

    # 1. Load log
    file_name = r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\data\event_log\data_log.pkl"
    log = 0
    with open(file_name, "rb") as f:
        log = pickle.load(f)
    log

    # 2. Extract list of arrivals
    timestamps_list = extract_timestamps_per_case(log)


    # 3. Start arrival generation
    # WITH SEGMENTATION
    arrival_generator_with_segmentation = StaticSpawner()
    arrival_generator_with_segmentation.fit(log, True)

    generated_arrivals_with_segmentation = arrival_generator_with_segmentation.generate_arrivals(timestamps_list[0], timestamps_list[-1])
    generated_arrivals_df_with_segmentation = pd.DataFrame()
    generated_arrivals_df_with_segmentation.insert(0, 'time:timestamp', generated_arrivals_with_segmentation)

    # WITHOUT SEGMENTATION
    arrival_generator_no_segmentation = StaticSpawner()
    arrival_generator_no_segmentation.fit(log, False)

    generated_arrivals_without_segmentation = arrival_generator_no_segmentation.generate_arrivals(timestamps_list[0], timestamps_list[-1])
    generated_arrivals_df_without_segmentation = pd.DataFrame()
    generated_arrivals_df_without_segmentation.insert(0, 'time:timestamp', generated_arrivals_without_segmentation)

    print(generated_arrivals_df_with_segmentation.head(50))
    print(generated_arrivals_df_without_segmentation.head(50))
    print(len(timestamps_list))
    print("With segmentation arrivals generated: ", len(generated_arrivals_with_segmentation), " Without segmentation generated arrivals: ", len(generated_arrivals_df_without_segmentation) )

