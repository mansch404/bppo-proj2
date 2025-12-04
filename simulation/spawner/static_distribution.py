import pandas as pd
import fitter
import pickle
import scipy.stats as stats

class StaticSpawner:
    def __init__(self):
        self.best_dist_name = None
        self.best_params = {}
        self.dist_object = None

    def fit(self, event_log):

        list_timestamps = extract_timestamps_per_case(event_log)
        deltas_minutes = delta_timestamps_in_minutes(list_timestamps)

        best_dist = find_best_fitting_distribution(deltas_minutes)

        self.best_dist_name = list(best_dist.keys())[0]
        self.best_params = best_dist[self.best_dist_name]
        self.dist_object = getattr(stats, self.best_dist_name)

        print(f"Best distribution: {self.best_dist_name}")
        print(f"Params: {self.best_params}")

    def generate_next(self):
        """
        Compute arrivals based on selected distribution.
        Returns: Float (time to wait in minutes)
        """
        if self.dist_object is None:
            raise ValueError("Model not trained!")

        val = self.dist_object.rvs(**self.best_params)


        return max(0.0, val)



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


    f = fitter.Fitter(deltas_timestamps_minutes_df, distributions=['gamma',
                                                                   'lognorm',
                                                                   'expon',
                                                                   'norm'
                                                                   ])
    print("Starting fitting...")
    f.fit()
    print(f.get_best(method='sumsquare_error'))




if __name__ == '__main__':

    # 1. Load log
    file_name = r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\data\event_log\data_log.pkl"
    log = 0
    with open(file_name, "rb") as f:
        log = pickle.load(f)
    log

    arrival_generator = StaticSpawner()
    arrival_generator.fit(log)

