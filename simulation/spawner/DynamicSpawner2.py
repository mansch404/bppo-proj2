import logging
import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from datetime import timedelta, datetime, time
from utils.helper import extract_timestamps_per_case
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import copy

class DynamicSpawner_LSTM():
    def __init__(self, train_arrival_times): # TODO explain the arguments to give
        logging.basicConfig(level=logging.INFO, format='%(filename)s:%(lineno)d - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.train_set = train_arrival_times.copy() # Train set or whole set
        self.simulated_arrivals = None # List of simulated arrivals
        self.lstm_model = None # The lstm model trained

    def create_and_train_model(self):
        if self.train_set is None:
            raise ValueError("No training data set was given. Initialize the spawner first.")
        else:
            self.lstm_model = LSTM_Generator(train_arrival_times=self.train_set) # Creates the model instance and the training takes place when initialized


    def generate_arrivals(self, start_date, number_of_days):
        if self.lstm_model is None:
            raise ValueError("No model is trained yet. Call function train_model().")

        self.simulated_arrivals = self.lstm_model.generate_arrivals(start_date, number_of_days)

        return self.simulated_arrivals




# TODO read the whole generator carefully and test the implementation. And check for correct variable names

class LSTM_Generator:
    def __init__(self, train_arrival_times, data_n_seqs: int=365):
        self.train_arrival_times = sorted(train_arrival_times)
        self.data_n_seqs = data_n_seqs # Number of days to simulate beginning with start time
        self.sequence_length = 5 # TODO Check meaning of var
        self.model = None
        # Switch to MinMaxScaler for better scaling of small values
        self.scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        # Train the model during initialization
        self._prepare_and_train_model()

    @staticmethod
    def _transform_features(df, scaler):
        """Transforms inter-arrival times and extracts temporal features."""
        df = df.copy()

        # Scale inter-arrival times
        df[['inter_time']] = scaler.transform(df[['inter_time']])

        # Extract additional temporal features
        timestamps = df['timestamp']

        # Hour of day (sine and cosine for cyclical nature)
        hours = timestamps.dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)  # 2*pi*hrs / 24
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        # Day of week (one-hot encoded)
        weekday = timestamps.dt.weekday
        weekday_ohe = to_categorical(weekday, num_classes=7)  # transform weekday into a categorical variable
        for i in range(7):
            df[f'day_{i}'] = weekday_ohe[:, i]

        # Week of month (1-5)
        df['week_of_month'] = timestamps.dt.day.apply(lambda x: (x - 1) // 7 + 1) / 5

        # Month (sine and cosine for cyclical nature)
        months = timestamps.dt.month
        df['month_sin'] = np.sin(2 * np.pi * months / 12)
        df['month_cos'] = np.cos(2 * np.pi * months / 12)

        # Quarter of year (1-4)
        df['quarter'] = timestamps.dt.quarter / 4

        # Is weekend
        df['is_weekend'] = timestamps.dt.weekday.isin([5, 6]).astype(float)

        # Select features for model
        features = ['inter_time', 'hour_sin', 'hour_cos',
                    'week_of_month', 'month_sin', 'month_cos', 'quarter',
                    'is_weekend'] + [f'day_{i}' for i in range(7)]

        return df[features]

    def _vectorize(self, log, ngram_size):
        """Converts the DataFrame into a sequence-based format for LSTM training."""
        num_samples = len(log) - ngram_size  # Adjust to avoid padding
        dt_prefixes, dt_expected = [], []

        # Remove padding approach and use sliding window
        for i in range(num_samples):
            dt_prefixes.append(log.iloc[i:i + ngram_size])
            dt_expected.append(log.iloc[i + ngram_size:i + ngram_size + 1][['inter_time']])

        dt_prefixes = pd.concat(dt_prefixes, axis=0, ignore_index=True)
        dt_expected = pd.concat(dt_expected, axis=0, ignore_index=True)

        # Reshape the data
        dt_prefixes = dt_prefixes.to_numpy().reshape(num_samples, ngram_size, -1)
        dt_expected = dt_expected.to_numpy().reshape((num_samples, 1))

        return dt_prefixes, dt_expected

    def _prepare_training_data(self):
        """Prepares training data with inter-arrival times and temporal features. Using the functions above."""
        # Convert list to DataFrame first
        df_arrivals = pd.DataFrame({'timestamp': self.train_arrival_times})
        daily_times = df_arrivals.to_dict('records')

        # Create DataFrame with timestamps and inter-arrival durations
        inter_arrival_times = []
        for i, event in enumerate(daily_times):
            delta = (daily_times[i]['timestamp'] -
                     daily_times[i - 1]['timestamp']).total_seconds() if i > 0 else 0
            inter_arrival_times.append({
                'inter_time': delta,
                'timestamp': daily_times[i]['timestamp']
            })

        train_data = pd.DataFrame(inter_arrival_times)

        # Fit and apply transformation
        self.scaler.fit(train_data[['inter_time']])
        train_data = self._transform_features(train_data, self.scaler)

        # Generate training sequences
        return self._vectorize(train_data, self.sequence_length)

    def _build_model(self, n_features):
        """Builds a LSTM model for inter-arrival time prediction."""
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, n_features),
                 return_sequences=True, dropout=0.1),
            LSTM(32, return_sequences=False, dropout=0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        # Use Huber loss which is less sensitive to outliers than MSE
        # and more sensitive to differences than MAE
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss=tf.keras.losses.Huber())
        return model

    def _prepare_and_train_model(self):
        """Prepares data and trains the LSTM model."""
        X_train, y_train = self._prepare_training_data()

        n_samples = len(X_train)
        batch_size = min(32, max(4, n_samples // 10))

        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]

        self.model = self._build_model(n_features=X_train.shape[2])
        self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    def generate_arrivals(self, start_time, num_days):
        """Generates arrival times based on learned patterns."""

        # Check if start_time is just a date, and convert it to midnight datetime
        if not hasattr(start_time, 'hour'):
            start_time = datetime.combine(start_time, time.min)

        generated_times = [start_time]
        current_sequence = []

        # Initialize sequence with first few actual times
        train_end_idx = min(self.sequence_length, len(self.train_arrival_times) - 1)
        for i in range(train_end_idx):
            timestamp = self.train_arrival_times[i]
            iat = (self.train_arrival_times[i + 1] - self.train_arrival_times[i]).total_seconds()

            # Use DataFrame to silence sklearn warning
            scaled_iat = self.scaler.transform(pd.DataFrame([[iat]], columns=['inter_time']))[0][0]

            # Calculate temporal features
            hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
            hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
            week_of_month = ((timestamp.day - 1) // 7 + 1) / 5
            month_sin = np.sin(2 * np.pi * timestamp.month / 12)
            month_cos = np.cos(2 * np.pi * timestamp.month / 12)

            # Quarter calculation
            quarter = ((timestamp.month - 1) // 3 + 1) / 4

            is_weekend = float(timestamp.weekday() in [5, 6])

            # One-hot encode weekday
            weekday_ohe = [0] * 7
            weekday_ohe[timestamp.weekday()] = 1

            features = [scaled_iat, hour_sin, hour_cos, week_of_month,
                        month_sin, month_cos, quarter, is_weekend] + weekday_ohe
            current_sequence.append(np.array(features))

        target_end_time = start_time + timedelta(days=num_days)
        recent_iats = []

        while generated_times[-1] < target_end_time:
            last_time = generated_times[-1]

            # Calculate temporal features for last time
            hour_sin = np.sin(2 * np.pi * last_time.hour / 24)
            hour_cos = np.cos(2 * np.pi * last_time.hour / 24)
            week_of_month = ((last_time.day - 1) // 7 + 1) / 5
            month_sin = np.sin(2 * np.pi * last_time.month / 12)
            month_cos = np.cos(2 * np.pi * last_time.month / 12)

            # Quarter calculation
            quarter = ((last_time.month - 1) // 3 + 1) / 4

            is_weekend = float(last_time.weekday() in [5, 6])

            # One-hot encode weekday
            weekday_ohe = [0] * 7
            weekday_ohe[last_time.weekday()] = 1

            # Prepare input sequence
            sequence = np.array(current_sequence[-self.sequence_length:])
            sequence[-1, 1:] = [hour_sin, hour_cos, week_of_month,
                                month_sin, month_cos, quarter, is_weekend] + weekday_ohe
            sequence = sequence.reshape(1, self.sequence_length, sequence.shape[1])

            # Predict next IAT
            predicted_scaled_iat = self.model.predict(sequence, verbose=0)[0][0]

            # Clip and Inverse Transform
            predicted_scaled_iat = np.clip(predicted_scaled_iat, 0.1, 0.9)
            predicted_iat = \
            self.scaler.inverse_transform(pd.DataFrame([[predicted_scaled_iat]], columns=['inter_time']))[0][0]

            # Add logic to ensure validity
            predicted_iat *= np.random.normal(1, 0.1)
            predicted_iat = max(1.0, predicted_iat)

            recent_iats.append(predicted_iat)
            if len(recent_iats) > 10: recent_iats.pop(0)
            if len(recent_iats) > 5 and predicted_iat < np.mean(recent_iats) * 0.1:
                predicted_iat = np.mean(recent_iats)

            # Generate next timestamp
            next_time = generated_times[-1] + timedelta(seconds=float(predicted_iat))

            if next_time < target_end_time:
                generated_times.append(next_time)

                # Calculate features for next time
                hour_sin = np.sin(2 * np.pi * next_time.hour / 24)
                hour_cos = np.cos(2 * np.pi * next_time.hour / 24)
                week_of_month = ((next_time.day - 1) // 7 + 1) / 5
                month_sin = np.sin(2 * np.pi * next_time.month / 12)
                month_cos = np.cos(2 * np.pi * next_time.month / 12)

                quarter = ((next_time.month - 1) // 3 + 1) / 4

                is_weekend = float(next_time.weekday() in [5, 6])
                weekday_ohe = [0] * 7
                weekday_ohe[next_time.weekday()] = 1

                features = [predicted_scaled_iat, hour_sin, hour_cos, week_of_month,
                            month_sin, month_cos, quarter, is_weekend] + weekday_ohe
                current_sequence.append(np.array(features))
            else:
                break

        return generated_times



if __name__ == '__main__':
    # 1. Load log
    file_name = r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\data\event_log\data_log.pkl"
    log = 0
    with open(file_name, "rb") as f:
        log = pickle.load(f)
    log

    # 2. Extract list of arrivals
    timestamps_list = extract_timestamps_per_case(pd.DataFrame(log))


    spawner = DynamicSpawner_LSTM(timestamps_list) # Initialize list

    print("Training model...")
    spawner.create_and_train_model() # Create model and train it with the data set

    start_date = timestamps_list[0]

    print("Generating arrivals...")
    spawner.generate_arrivals(start_date, 365) # Generate arrivals after training model

    sim_arrivals_df = pd.DataFrame(spawner.simulated_arrivals)
    print(sim_arrivals_df.head(10))
    print("Number of generated arrivals: ",len(spawner.simulated_arrivals))



