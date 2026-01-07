from pathlib import Path

import pandas as pd
import logging
import pickle
import scipy.stats as stats
from utils.helper import delta_timestamps_in_seconds, find_best_fitting_distribution, extract_timestamps_per_case, _setup_clustered_train_dict, _read_event_log
from datetime import datetime, timedelta


class StaticSpawner:
    def __init__(self):
        self.best_dist_name = [] # Saves the best distribution(s)
        self.best_params = [] # The parameters of the selected distribution(s)
        self.dist_object = []
        self.segmentation = False # If True, segmentation is executed and a the fitted distributions for each cluster are saved
        self.train_df_clustered = None # Training data set

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

            # Find the best fitting distribution for each found cluster
            for cluster in range(len(self.train_df_clustered['cluster'].unique())):
                current_deltas_seconds =  delta_timestamps_in_seconds(self.train_df_clustered[self.train_df_clustered['cluster'] == cluster]['date'])
                current_best_dist = find_best_fitting_distribution(current_deltas_seconds)
                best_dist.append(current_best_dist)

            for i, best_d in enumerate(best_dist):
                dist_name = list(best_d.keys())[0]
                self.best_dist_name.append(dist_name)
                self.best_params.append(best_d[dist_name])
                self.dist_object.append(getattr(stats, dist_name))

        # Find a fitting distribution for whole arrivals without segmentation.
        else:
            deltas_seconds = delta_timestamps_in_seconds(list_timestamps)
            best_dist = find_best_fitting_distribution(deltas_seconds)

            self.best_dist_name = list(best_dist.keys())[0]
            self.best_params.append(best_dist[self.best_dist_name])
            self.dist_object = getattr(stats, self.best_dist_name)

            print(f"Best distribution: {self.best_dist_name}")

    def fit_with_log_path(self, event_log_path, segmentation=False):
        """
        Args: event_log_path: Path to event log
        """
        # Read the event log from Path
        path = Path(event_log_path)
        event_log = _read_event_log(path)

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

            # Find the best fitting distribution for each found cluster
            for cluster in range(len(self.train_df_clustered['cluster'].unique())):
                current_deltas_seconds =  delta_timestamps_in_seconds(self.train_df_clustered[self.train_df_clustered['cluster'] == cluster]['date'])
                current_best_dist = find_best_fitting_distribution(current_deltas_seconds)
                best_dist.append(current_best_dist)

            for i, best_d in enumerate(best_dist):
                dist_name = list(best_d.keys())[0]
                self.best_dist_name.append(dist_name)
                self.best_params.append(best_d[dist_name])
                self.dist_object.append(getattr(stats, dist_name))

        # Find a fitting distribution for whole arrivals without segmentation.
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

