"""
Advanced Processing Time Prediction - Quantile Regression
Task 1.3 Advanced: Context-aware probabilistic processing time prediction

Uses LightGBM with Quantile Loss to predict processing time distributions
based on case context, trace history, and available attributes.

Key differences from Basic approach:
- No percentile filtering: The model learns the full distribution
- Context-aware: Uses trace history, case attributes, and temporal features
- Probabilistic output: Returns quantiles (q10, q25, q50, q75, q90) instead of point estimates
"""

import pm4py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import pickle
import json


# =============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# =============================================================================


def load_event_log(path: str) -> pd.DataFrame:
    """
    Load XES event log and convert to DataFrame.

    Args:
        path: Path to the XES file

    Returns:
        DataFrame with all events
    """
    print(f"Loading event log from {path}...")
    log = pm4py.read_xes(path)
    df = pm4py.convert_to_dataframe(log)
    print(f"  Total events: {len(df):,}")
    print(f"  Unique cases: {df['case:concept:name'].nunique():,}")
    print(f"  Unique activities: {df['concept:name'].nunique()}")
    return df


def extract_processing_times_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract processing times with full context for each event.

    Processing time calculation:
    - W_ activities: actual processing time (complete - start)
    - A_/O_ activities: time-to-next-event as proxy

    Args:
        df: DataFrame with all events

    Returns:
        DataFrame with one row per event that has a valid processing time,
        including all contextual features
    """
    print("\n" + "=" * 70)
    print("EXTRACTING PROCESSING TIMES WITH CONTEXT")
    print("=" * 70)

    # Sort by case and timestamp
    df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

    print("\nExtracting events with full context (no percentile filtering)...")
    training_data = []

    for case_id, case_df in df.groupby("case:concept:name"):
        case_df = case_df.reset_index(drop=True)

        # Case-level attributes (constant throughout the case)
        case_requested_amount = case_df["case:RequestedAmount"].iloc[0]
        case_loan_goal = case_df["case:LoanGoal"].iloc[0]
        case_application_type = case_df["case:ApplicationType"].iloc[0]
        case_start_time = case_df["time:timestamp"].iloc[0]

        # Offer attributes (become known after O_Create Offer)
        offer_info = {
            "CreditScore": np.nan,
            "OfferedAmount": np.nan,
            "NumberOfTerms": np.nan,
            "MonthlyCost": np.nan,
        }

        # Track trace history
        previous_activity = "START"
        event_nr = 0

        for idx, row in case_df.iterrows():
            activity = row["concept:name"]
            lifecycle = row.get("lifecycle:transition", "complete")
            timestamp = row["time:timestamp"]

            # Update offer info when O_Create Offer is completed
            if activity == "O_Create Offer" and lifecycle == "complete":
                if pd.notna(row.get("CreditScore")):
                    offer_info["CreditScore"] = row["CreditScore"]
                if pd.notna(row.get("OfferedAmount")):
                    offer_info["OfferedAmount"] = row["OfferedAmount"]
                if pd.notna(row.get("NumberOfTerms")):
                    offer_info["NumberOfTerms"] = row["NumberOfTerms"]
                if pd.notna(row.get("MonthlyCost")):
                    offer_info["MonthlyCost"] = row["MonthlyCost"]

            # Only process completion events for target calculation
            if lifecycle not in ["complete", "ate_abort", "withdraw"]:
                continue

            duration = None

            # W_ activities: actual processing time (start to complete)
            if activity.startswith("W_"):
                start_events = case_df[
                    (case_df["concept:name"] == activity)
                    & (case_df["lifecycle:transition"] == "start")
                    & (case_df.index < idx)
                ]
                if not start_events.empty:
                    start_time = start_events.iloc[-1]["time:timestamp"]
                    duration = (timestamp - start_time).total_seconds()

            # A_/O_ activities: time-to-next-event as proxy
            elif activity.startswith(("A_", "O_")):
                if idx < len(case_df) - 1:
                    next_time = case_df.iloc[idx + 1]["time:timestamp"]
                    duration = (next_time - timestamp).total_seconds()

            # Add sample if duration is valid
            if duration is not None and duration > 0:
                sample = {
                    # Target variable
                    "processing_time": duration,
                    # Activity context
                    "activity": activity,
                    "previous_activity": previous_activity,
                    # Case attributes
                    "RequestedAmount": case_requested_amount,
                    "LoanGoal": case_loan_goal,
                    "ApplicationType": case_application_type,
                    # Trace context
                    "event_nr": event_nr,
                    "elapsed_time": (timestamp - case_start_time).total_seconds(),
                    # Temporal features
                    "hour_of_day": timestamp.hour,
                    "day_of_week": timestamp.dayofweek,
                    # Offer attributes (NaN if not yet known in the process)
                    "CreditScore": offer_info["CreditScore"],
                    "OfferedAmount": offer_info["OfferedAmount"],
                    "NumberOfTerms": offer_info["NumberOfTerms"],
                    "MonthlyCost": offer_info["MonthlyCost"],
                    # Metadata (for splitting, not used as features)
                    "case_id": case_id,
                    "timestamp": timestamp,
                }
                training_data.append(sample)

            # Update trace history
            previous_activity = activity
            event_nr += 1

    print(f"\n  Total training samples: {len(training_data):,}")

    return pd.DataFrame(training_data)


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================


def prepare_features(
    df: pd.DataFrame,
    use_log_transform: bool = True,
    min_processing_time: float = 1.0,
    max_processing_time: float = 28800.0,
):
    """
    Prepare features for LightGBM training.

    Args:
        df: DataFrame with training data
        use_log_transform: If True, apply log(1 + y) transform to target
        min_processing_time: Minimum processing time in seconds (filters system events)
        max_processing_time: Maximum processing time in seconds (filters overnight/weekend waits)

    Returns:
        X: Feature DataFrame
        y: Target array (log-transformed if use_log_transform=True)
        feature_info: Dict with encoders, feature names, and transformation settings
        filtered_df: Filtered DataFrame (for splitting by case_id)
    """
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    df = df.copy()

    # Filter out very short processing times (likely system events)
    original_len = len(df)
    df = df[df["processing_time"] >= min_processing_time].reset_index(drop=True)
    after_min_filter = len(df)
    print(
        f"  Filtered samples < {min_processing_time}s: {original_len - after_min_filter:,} removed"
    )

    # Filter out very long processing times (likely overnight/weekend waits)
    df = df[df["processing_time"] <= max_processing_time].reset_index(drop=True)
    after_max_filter = len(df)
    print(
        f"  Filtered samples > {max_processing_time}s ({max_processing_time / 3600:.1f}h): {after_min_filter - after_max_filter:,} removed"
    )
    print(f"  Remaining samples: {after_max_filter:,}")

    # Extract and transform target
    y_raw = df["processing_time"].values
    if use_log_transform:
        y = np.log1p(y_raw)  # log(1 + y) to handle values close to 0
        print(f"  Applied log1p transformation to target")
        print(f"    Original range: [{y_raw.min():.2f}, {y_raw.max():.2f}]")
        print(f"    Transformed range: [{y.min():.2f}, {y.max():.2f}]")
    else:
        y = y_raw

    # Initialize feature info
    feature_info = {
        "categorical_features": [],
        "numerical_features": [],
        "label_encoders": {},
        "use_log_transform": use_log_transform,
        "min_processing_time": min_processing_time,
        "max_processing_time": max_processing_time,
    }

    # Encode categorical features
    categorical_cols = [
        "activity",
        "previous_activity",
        "LoanGoal",
        "ApplicationType",
        "day_of_week",
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna("UNKNOWN")
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        feature_info["label_encoders"][col] = le
        feature_info["categorical_features"].append(col + "_encoded")

    print(f"  Categorical features: {len(categorical_cols)}")
    for col in categorical_cols:
        print(f"    {col}: {df[col].nunique()} unique values")

    # Numerical features
    numerical_cols = [
        "RequestedAmount",
        "event_nr",
        "elapsed_time",
        "hour_of_day",
        "CreditScore",
        "OfferedAmount",
        "NumberOfTerms",
        "MonthlyCost",
    ]
    feature_info["numerical_features"] = numerical_cols

    print(f"  Numerical features: {len(numerical_cols)}")
    for col in numerical_cols:
        nan_pct = df[col].isna().mean() * 100
        print(f"    {col}: {nan_pct:.1f}% missing")

    # Build feature matrix
    feature_cols = feature_info["categorical_features"] + numerical_cols
    X = df[feature_cols].copy()
    feature_info["feature_names"] = feature_cols

    print(f"\n  Total features: {len(feature_cols)}")
    print(f"  Training samples: {len(X):,}")

    return X, y, feature_info, df


# =============================================================================
# STEP 3: TRAIN/TEST SPLIT
# =============================================================================


def temporal_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Temporal split: Use early cases for training, later cases for testing.

    Args:
        df: DataFrame with 'case_id' and 'timestamp' columns
        test_ratio: Fraction of cases to use for testing

    Returns:
        train_mask: Boolean mask for training samples
        test_mask: Boolean mask for test samples
    """
    print("\n" + "=" * 70)
    print("TEMPORAL TRAIN/TEST SPLIT")
    print("=" * 70)

    # Sort cases by their first event timestamp
    case_start_times = df.groupby("case_id")["timestamp"].min().sort_values()

    n_cases = len(case_start_times)
    n_train = int(n_cases * (1 - test_ratio))

    train_cases = set(case_start_times.iloc[:n_train].index)
    test_cases = set(case_start_times.iloc[n_train:].index)

    train_mask = df["case_id"].isin(train_cases)
    test_mask = df["case_id"].isin(test_cases)

    print(f"  Total cases: {n_cases:,}")
    print(f"  Train cases: {len(train_cases):,} ({100 * (1 - test_ratio):.0f}%)")
    print(f"  Test cases: {len(test_cases):,} ({100 * test_ratio:.0f}%)")
    print(f"  Train samples: {train_mask.sum():,}")
    print(f"  Test samples: {test_mask.sum():,}")

    return train_mask, test_mask


# =============================================================================
# STEP 4: MODEL TRAINING
# =============================================================================


def train_quantile_models(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    categorical_features=None,
):
    """
    Train separate LightGBM models for each quantile.

    Args:
        X_train: Training features
        y_train: Training targets (log-transformed)
        X_val: Validation features (for early stopping)
        y_val: Validation targets
        quantiles: List of quantiles to predict
        categorical_features: List of categorical feature column names

    Returns:
        models: Dict {quantile: trained_model}
    """
    print("\n" + "=" * 70)
    print("TRAINING QUANTILE REGRESSION MODELS")
    print("=" * 70)

    models = {}

    # LightGBM hyperparameters
    base_params = {
        "objective": "quantile",
        "metric": "quantile",
        "boosting_type": "gbdt",
        "num_leaves": 20,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 100,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # Get categorical feature indices
    cat_feature_indices = []
    if categorical_features:
        for i, col in enumerate(X_train.columns):
            if col in categorical_features:
                cat_feature_indices.append(i)

    for q in quantiles:
        print(f"\n  Training model for quantile {q:.2f}...")

        params = base_params.copy()
        params["alpha"] = q

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=cat_feature_indices if cat_feature_indices else "auto",
        )

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            print(f"    Best iteration: {model.best_iteration}")
        else:
            model = lgb.train(params, train_data, num_boost_round=300)
            print(f"    Trained for 300 rounds")

        models[q] = model

    print(f"\n  Trained {len(models)} quantile models")

    return models


# =============================================================================
# STEP 5: EVALUATION
# =============================================================================


def evaluate_models(models, X_test, y_test, quantiles):
    """
    Evaluate quantile models using multiple metrics.

    Metrics:
    1. Pinball Loss (quantile-specific loss function)
    2. Coverage (prediction interval accuracy)
    3. MAE/RMSE for median predictions

    Args:
        models: Dict {quantile: model}
        X_test: Test features
        y_test: Test targets (log-transformed)
        quantiles: List of quantiles

    Returns:
        Dict with evaluation results
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Get raw predictions
    raw_predictions = {}
    for q, model in models.items():
        raw_predictions[q] = model.predict(X_test, num_iteration=model.best_iteration)

    # Apply quantile crossing correction
    print("\n  Applying quantile crossing correction...")
    predictions = {q: np.zeros(len(y_test)) for q in quantiles}

    n_crossings = 0
    for i in range(len(y_test)):
        sample_preds = {q: raw_predictions[q][i] for q in quantiles}

        # Check for crossing
        sorted_q = sorted(sample_preds.keys())
        values = [sample_preds[q] for q in sorted_q]
        if any(values[j] > values[j + 1] for j in range(len(values) - 1)):
            n_crossings += 1

        # Fix crossing
        fixed = fix_quantile_crossing(sample_preds)
        for q in quantiles:
            predictions[q][i] = fixed[q]

    print(
        f"    Samples with crossing: {n_crossings:,} ({100 * n_crossings / len(y_test):.1f}%)"
    )

    # 1. Pinball Loss
    print("\n  Pinball Loss per Quantile:")
    pinball_losses = {}
    for q in quantiles:
        error = y_test - predictions[q]
        loss = np.where(error >= 0, q * error, (q - 1) * error)
        pinball_losses[q] = np.mean(loss)
        print(f"    q{int(q * 100):02d}: {pinball_losses[q]:.4f}")

    avg_pinball = np.mean(list(pinball_losses.values()))
    print(f"    Average: {avg_pinball:.4f}")

    # 2. Coverage
    print("\n  Prediction Interval Coverage:")
    if 0.1 in predictions and 0.9 in predictions:
        coverage_80 = (
            np.mean((y_test >= predictions[0.1]) & (y_test <= predictions[0.9])) * 100
        )
        print(f"    10-90 interval (expected 80%): {coverage_80:.1f}%")

    if 0.25 in predictions and 0.75 in predictions:
        coverage_50 = (
            np.mean((y_test >= predictions[0.25]) & (y_test <= predictions[0.75])) * 100
        )
        print(f"    25-75 interval (expected 50%): {coverage_50:.1f}%")

    # 3. Median accuracy (in both spaces)
    if 0.5 in predictions:
        median_pred = predictions[0.5]
        mae_log = np.mean(np.abs(y_test - median_pred))
        rmse_log = np.sqrt(np.mean((y_test - median_pred) ** 2))
        print(f"\n  Median Prediction Accuracy (log space):")
        print(f"    MAE: {mae_log:.4f}")
        print(f"    RMSE: {rmse_log:.4f}")

        # Back-transform to original scale
        y_orig = np.expm1(y_test)
        pred_orig = np.expm1(median_pred)
        mae_orig = np.mean(np.abs(y_orig - pred_orig))
        rmse_orig = np.sqrt(np.mean((y_orig - pred_orig) ** 2))
        print(f"  Median Prediction Accuracy (original scale):")
        print(f"    MAE: {mae_orig:.2f}s ({mae_orig / 60:.2f} min)")
        print(f"    RMSE: {rmse_orig:.2f}s ({rmse_orig / 60:.2f} min)")

    # 4. Target distribution
    y_orig = np.expm1(y_test)
    print(f"\n  Target Distribution (original scale):")
    print(f"    Median: {np.median(y_orig):.2f}s ({np.median(y_orig) / 60:.2f} min)")
    print(
        f"    90th percentile: {np.percentile(y_orig, 90):.2f}s ({np.percentile(y_orig, 90) / 60:.2f} min)"
    )
    print(f"    Max: {np.max(y_orig):.2f}s ({np.max(y_orig) / 3600:.2f} hours)")

    # 5. Feature importance
    if 0.5 in models:
        print(f"\n  Feature Importance (top 10):")
        importance = models[0.5].feature_importance(importance_type="gain")
        feature_names = models[0.5].feature_name()
        sorted_idx = np.argsort(importance)[::-1]
        for i in sorted_idx[:10]:
            print(f"    {feature_names[i]}: {importance[i]:.2f}")

    return {
        "pinball_losses": pinball_losses,
        "avg_pinball": avg_pinball,
        "predictions": predictions,
    }


# =============================================================================
# STEP 6: SAVE/LOAD MODELS
# =============================================================================


def save_models(models, feature_info, filepath="quantile_models.pkl"):
    """Save trained models and feature info to disk."""
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    save_data = {
        "models": models,
        "feature_info": feature_info,
        "quantiles": list(models.keys()),
    }

    with open(filepath, "wb") as f:
        pickle.dump(save_data, f)
    print(f"  Saved to {filepath}")

    # Also save metadata as JSON
    json_info = {
        "quantiles": list(models.keys()),
        "feature_names": feature_info["feature_names"],
        "categorical_features": feature_info["categorical_features"],
        "numerical_features": feature_info["numerical_features"],
        "use_log_transform": feature_info["use_log_transform"],
        "min_processing_time": feature_info["min_processing_time"],
        "max_processing_time": feature_info["max_processing_time"],
    }

    json_path = filepath.replace(".pkl", "_info.json")
    with open(json_path, "w") as f:
        json.dump(json_info, f, indent=2)
    print(f"  Saved metadata to {json_path}")


def load_models(filepath="quantile_models.pkl"):
    """Load trained models from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


# =============================================================================
# STEP 7: INFERENCE (for integration with simulation engine)
# =============================================================================


def predict_processing_time_distribution(
    activity: str,
    previous_activity: str,
    requested_amount: float,
    loan_goal: str,
    application_type: str,
    event_nr: int,
    elapsed_time: float,
    hour_of_day: int,
    day_of_week: int,
    credit_score: float = None,
    offered_amount: float = None,
    number_of_terms: float = None,
    monthly_cost: float = None,
    models_data: dict = None,
) -> dict:
    """
    Predict processing time quantiles for a single event.

    Args:
        activity: Current activity name
        previous_activity: Previous activity in the trace
        requested_amount: Requested loan amount
        loan_goal: Purpose of the loan
        application_type: Type of application
        event_nr: Position in the trace (0-indexed)
        elapsed_time: Time since case start in seconds
        hour_of_day: Hour of the event (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        credit_score: Credit score (None if not yet known)
        offered_amount: Offered amount (None if not yet known)
        number_of_terms: Number of terms (None if not yet known)
        monthly_cost: Monthly cost (None if not yet known)
        models_data: Loaded model data from load_models()

    Returns:
        Dict {quantile: predicted_time_in_seconds}
    """
    if models_data is None:
        raise ValueError("models_data must be provided")

    models = models_data["models"]
    feature_info = models_data["feature_info"]
    encoders = feature_info["label_encoders"]
    use_log_transform = feature_info.get("use_log_transform", False)

    def safe_encode(encoder, value):
        """Encode value, returning 0 for unknown values."""
        try:
            return encoder.transform([str(value)])[0]
        except ValueError:
            return 0

    # Build feature vector
    features = {
        "activity_encoded": safe_encode(encoders["activity"], activity),
        "previous_activity_encoded": safe_encode(
            encoders["previous_activity"], previous_activity
        ),
        "LoanGoal_encoded": safe_encode(encoders["LoanGoal"], loan_goal),
        "ApplicationType_encoded": safe_encode(
            encoders["ApplicationType"], application_type
        ),
        "day_of_week_encoded": safe_encode(encoders["day_of_week"], day_of_week),
        "RequestedAmount": requested_amount,
        "event_nr": event_nr,
        "elapsed_time": elapsed_time,
        "hour_of_day": hour_of_day,
        "CreditScore": credit_score if credit_score is not None else np.nan,
        "OfferedAmount": offered_amount if offered_amount is not None else np.nan,
        "NumberOfTerms": number_of_terms if number_of_terms is not None else np.nan,
        "MonthlyCost": monthly_cost if monthly_cost is not None else np.nan,
    }

    X = pd.DataFrame([features])[feature_info["feature_names"]]

    # Predict quantiles
    raw_predictions = {}
    for q, model in models.items():
        pred = model.predict(X, num_iteration=model.best_iteration)[0]
        if use_log_transform:
            pred = np.expm1(pred)  # Back-transform from log space
        raw_predictions[q] = pred

    # Fix quantile crossing and ensure positive values
    return fix_quantile_crossing(raw_predictions)


def fix_quantile_crossing(predictions: dict, min_value: float = 0.01) -> dict:
    """
    Fix quantile crossing using isotonic regression.

    Ensures that q10 <= q25 <= q50 <= q75 <= q90.

    Args:
        predictions: Dict {quantile: value}
        min_value: Minimum allowed value

    Returns:
        Fixed predictions with monotonically increasing quantiles
    """
    sorted_q = sorted(predictions.keys())
    values = np.array([predictions[q] for q in sorted_q])

    # Ensure minimum value
    values = np.maximum(values, min_value)

    # Check if already monotonic
    if all(values[i] <= values[i + 1] for i in range(len(values) - 1)):
        return {q: float(v) for q, v in zip(sorted_q, values)}

    # Apply pool adjacent violators algorithm
    result = values.copy().astype(float)
    i = 0
    while i < len(result) - 1:
        if result[i] > result[i + 1]:
            # Find block of violators
            j = i + 1
            while j < len(result) - 1 and result[j] > result[j + 1]:
                j += 1
            # Average the block
            block_mean = np.mean(result[i : j + 1])
            result[i : j + 1] = block_mean
            # Go back to check for new violations
            if i > 0:
                i -= 1
            else:
                i += 1
        else:
            i += 1

    return {q: float(v) for q, v in zip(sorted_q, result)}


def sample_from_quantiles(quantile_predictions: dict) -> float:
    """
    Sample a processing time from the predicted quantile distribution.

    Uses linear interpolation between quantiles.

    Args:
        quantile_predictions: Dict {quantile: value}

    Returns:
        Sampled processing time in seconds
    """
    # Ensure monotonicity
    fixed = fix_quantile_crossing(quantile_predictions)

    sorted_q = sorted(fixed.keys())
    sorted_v = [fixed[q] for q in sorted_q]

    # Draw uniform random number
    u = np.random.uniform(0, 1)

    # Handle edge cases
    if u <= sorted_q[0]:
        return sorted_v[0]
    if u >= sorted_q[-1]:
        return sorted_v[-1]

    # Linear interpolation
    for i in range(len(sorted_q) - 1):
        if sorted_q[i] <= u <= sorted_q[i + 1]:
            t = (u - sorted_q[i]) / (sorted_q[i + 1] - sorted_q[i])
            return sorted_v[i] + t * (sorted_v[i + 1] - sorted_v[i])

    # Fallback
    return fixed.get(0.5, np.median(sorted_v))


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================


def main():
    """Main training pipeline."""

    # 1. Load data
    df = load_event_log("BPI Challenge 2017.xes")

    # 2. Extract processing times with context
    training_df = extract_processing_times_for_training(df)

    # 3. Prepare features
    X, y, feature_info, filtered_df = prepare_features(
        training_df,
        use_log_transform=True,
        min_processing_time=1.0,  # Filter system events (< 1s)
        max_processing_time=28800.0,  # Filter overnight waits (> 8h)
    )

    # 4. Temporal split
    train_mask, test_mask = temporal_split(filtered_df, test_ratio=0.2)

    # Create validation set from training data
    train_indices = np.where(train_mask)[0]
    n_val = int(len(train_indices) * 0.125)
    val_indices = train_indices[-n_val:]
    actual_train_indices = train_indices[:-n_val]

    X_train = X.iloc[actual_train_indices]
    X_val = X.iloc[val_indices]
    X_test = X[test_mask]
    y_train = y[actual_train_indices]
    y_val = y[val_indices]
    y_test = y[test_mask]

    print(f"\n  Final split:")
    print(f"    Train: {len(X_train):,}")
    print(f"    Validation: {len(X_val):,}")
    print(f"    Test: {len(X_test):,}")

    # 5. Train models
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    models = train_quantile_models(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        quantiles=quantiles,
        categorical_features=feature_info["categorical_features"],
    )

    # 6. Evaluate
    eval_results = evaluate_models(models, X_test, y_test, quantiles)

    # 7. Save
    save_models(models, feature_info, "quantile_models.pkl")

    # 8. Test inference
    print("\n" + "=" * 70)
    print("TESTING INFERENCE")
    print("=" * 70)

    models_data = load_models("quantile_models.pkl")

    test_pred = predict_processing_time_distribution(
        activity="W_Complete application",
        previous_activity="A_Concept",
        requested_amount=20000.0,
        loan_goal="Home improvement",
        application_type="New credit",
        event_nr=5,
        elapsed_time=3600.0,
        hour_of_day=10,
        day_of_week=1,
        models_data=models_data,
    )

    print("\n  Predictions for 'W_Complete application':")
    for q, v in sorted(test_pred.items()):
        print(f"    q{int(q * 100):02d}: {v:.2f}s ({v / 60:.2f} min)")

    print("\n  5 samples:")
    for i in range(5):
        sample = sample_from_quantiles(test_pred)
        print(f"    {sample:.2f}s")

    return models, feature_info, eval_results


if __name__ == "__main__":
    models, feature_info, eval_results = main()
