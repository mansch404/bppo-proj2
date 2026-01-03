"""
Processing Time Analysis
Extract and analyze processing times from historical event log
Hybrid approach: Actual times for W_ activities, 25th percentile for A_/O_
"""

import pm4py
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kstest
import pickle
import json


def load_event_log(path: str):
    """Load XES event log"""
    log = pm4py.read_xes(path)
    df = pm4py.convert_to_dataframe(log)
    return df


def extract_processing_times_hybrid(df: pd.DataFrame):
    """
    Extract processing times using hybrid approach:
    1. W_ Activities with start/complete: actual processing time
    2. A_/O_ Activities: time-to-next-event (will use 25th percentile later)

    Returns:
        actual_times: dict {activity: [times]} for W_ activities with start/complete
        proxy_times: dict {activity: [times]} for A_/O_ activities (time-to-next)
    """
    actual_times = {}  # W_ activities with start/complete
    proxy_times = {}  # A_/O_ activities (time-to-next-event)

    # Sort by case and timestamp
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    # Group by case
    for case_id, case_df in df.groupby("case:concept:name"):
        case_df = case_df.reset_index(drop=True)

        for idx, row in case_df.iterrows():
            activity = row["concept:name"]
            lifecycle = row.get("lifecycle:transition", "complete")

            # Skip start events (we'll handle them with complete)
            if lifecycle == "start":
                continue

            # Check if activity starts with W_ and has start event
            if activity.startswith("W_"):
                # Look for corresponding start event
                start_event = case_df[
                    (case_df["concept:name"] == activity)
                    & (case_df["lifecycle:transition"] == "start")
                    & (case_df.index < idx)
                ]

                if not start_event.empty:
                    # Use actual processing time (start to complete)
                    start_time = start_event.iloc[-1]["time:timestamp"]
                    complete_time = row["time:timestamp"]
                    duration = (complete_time - start_time).total_seconds()

                    if activity not in actual_times:
                        actual_times[activity] = []

                    if duration > 0:
                        actual_times[activity].append(duration)
                    continue

            # For A_/O_ activities: use time-to-next-event
            if activity.startswith("W_"):
                continue

            if idx < len(case_df) - 1:
                next_time = case_df.iloc[idx + 1]["time:timestamp"]
                current_time = row["time:timestamp"]
                duration = (next_time - current_time).total_seconds()

                if activity not in proxy_times:
                    proxy_times[activity] = []

                if duration > 0:
                    proxy_times[activity].append(duration)

    return actual_times, proxy_times


def compute_final_processing_times(
    actual_times: dict, proxy_times: dict, percentile: float = 25.0
):
    """
    Compute final processing times:
    - W_ activities: use 25th percentile (to avoid overnight/weekend waiting times)
    - A_/O_ activities: use 25th percentile of time-to-next-event

    Returns: dict {activity: [times]}
    """
    final_times = {}

    print("\n=== Computing Final Processing Times ===")

    # W_ Activities: Use 25th percentile to remove waiting times
    print("\n--- W_ Activities (25th Percentile to Remove Waiting Times) ---")
    for activity, times in actual_times.items():
        if len(times) == 0:
            continue

        # Take 25th percentile to avoid overnight/weekend waiting
        p25 = np.percentile(times, percentile)

        # Use values <= 25th percentile
        filtered = [t for t in times if t <= p25]

        if len(filtered) > 0:
            final_times[activity] = filtered
            print(
                f"{activity}: 25th percentile = {p25:.2f}s ({p25 / 3600:.2f}h), {len(filtered)} samples used"
            )

    # A_/O_ Activities: Use 25th percentile
    # Skip activities that are already in final_times
    print("\n--- A_/O_ Activities (25th Percentile of Time-to-Next) ---")
    for activity, times in proxy_times.items():
        # Skip if already processed in actual_times
        if activity in final_times:
            print(f"{activity}: Skipped (using actual times instead)")
            continue

        if len(times) == 0:
            continue

        # Take 25th percentile to avoid waiting times
        p25 = np.percentile(times, percentile)

        # Use values <= 25th percentile
        filtered = [t for t in times if t <= p25]

        if len(filtered) > 0:
            final_times[activity] = filtered
            print(
                f"{activity}: 25th percentile = {p25:.2f}s, {len(filtered)} samples used"
            )

    return final_times


def show_statistics(times: dict):
    """Display statistics for processing times"""
    print("\n" + "=" * 70)
    print("FINAL PROCESSING TIME STATISTICS")
    print("=" * 70)

    for activity, durations in sorted(times.items()):
        print(f"\n{activity}:")
        print(f"  Count: {len(durations)}")
        print(f"  Mean: {np.mean(durations):.2f}s ({np.mean(durations) / 60:.2f}min)")
        print(f"  Median: {np.median(durations):.2f}s")
        print(f"  Std: {np.std(durations):.2f}s")
        print(f"  Min: {np.min(durations):.2f}s")
        print(f"  Max: {np.max(durations):.2f}s")


def fit_distributions(processing_times: dict):
    """
    Fit multiple distributions to processing times and select best fit

    Candidates:
    - Lognormal (best for right-skewed data)
    - Exponential (memoryless)
    - Gamma (flexible)
    - Normal (symmetric)

    Returns: dict {activity: {'distribution': name, 'params': tuple}}
    """
    fitted_distributions = {}

    print("\n" + "=" * 70)
    print("FITTING PROBABILITY DISTRIBUTIONS")
    print("=" * 70)

    for activity, times in sorted(processing_times.items()):
        if len(times) < 10:  # Need enough samples
            print(f"\n{activity}: Too few samples ({len(times)}), skipping")
            continue

        print(f"\n{activity}:")
        print(f"  Samples: {len(times)}")

        data = np.array(times)

        # Remove zeros for log-based distributions
        data_positive = data[data > 0]
        if len(data_positive) < len(data):
            print(f"  Warning: Removed {len(data) - len(data_positive)} zero values")
            data = data_positive

        if len(data) < 10:
            print(f"  Too few positive samples, skipping")
            continue

        # Test different distributions
        results = {}

        # 1. Lognormal
        try:
            shape, loc, scale = stats.lognorm.fit(data, floc=0)
            ks_stat, p_value = kstest(
                data, lambda x: stats.lognorm.cdf(x, shape, loc, scale)
            )
            results["lognorm"] = {
                "params": (shape, loc, scale),
                "ks_stat": ks_stat,
                "p_value": p_value,
            }
        except:
            pass

        # 2. Exponential
        try:
            loc, scale = stats.expon.fit(data, floc=0)
            ks_stat, p_value = kstest(data, lambda x: stats.expon.cdf(x, loc, scale))
            results["expon"] = {
                "params": (loc, scale),
                "ks_stat": ks_stat,
                "p_value": p_value,
            }
        except:
            pass

        # 3. Gamma
        try:
            shape, loc, scale = stats.gamma.fit(data, floc=0)
            ks_stat, p_value = kstest(
                data, lambda x: stats.gamma.cdf(x, shape, loc, scale)
            )
            results["gamma"] = {
                "params": (shape, loc, scale),
                "ks_stat": ks_stat,
                "p_value": p_value,
            }
        except:
            pass

        # 4. Normal
        try:
            loc, scale = stats.norm.fit(data)
            ks_stat, p_value = kstest(data, lambda x: stats.norm.cdf(x, loc, scale))
            results["norm"] = {
                "params": (loc, scale),
                "ks_stat": ks_stat,
                "p_value": p_value,
            }
        except:
            pass

        # Select best distribution (lowest KS statistic = best fit)
        if not results:
            print(f"  ERROR: Could not fit any distribution")
            continue

        best_dist = min(results.items(), key=lambda x: x[1]["ks_stat"])
        dist_name = best_dist[0]
        dist_info = best_dist[1]

        fitted_distributions[activity] = {
            "distribution": dist_name,
            "params": dist_info["params"],
            "ks_stat": dist_info["ks_stat"],
            "p_value": dist_info["p_value"],
        }

        # Show results
        print(f"  Best fit: {dist_name}")
        print(f"    KS statistic: {dist_info['ks_stat']:.4f}")
        print(f"    p-value: {dist_info['p_value']:.4f}")
        print(f"    params: {dist_info['params']}")

        # Show all tested distributions
        print(f"  All candidates:")
        for name, info in sorted(results.items(), key=lambda x: x[1]["ks_stat"]):
            print(f"    {name:12s}: KS={info['ks_stat']:.4f}, p={info['p_value']:.4f}")

    return fitted_distributions


def save_distributions(
    fitted_distributions: dict, filepath: str = "fitted_distributions.pkl"
):
    """Save fitted distributions to file"""
    with open(filepath, "wb") as f:
        pickle.dump(fitted_distributions, f)
    print(f"\nSaved fitted distributions to {filepath}")

    # Also save as JSON for readability
    json_data = {}
    for activity, info in fitted_distributions.items():
        json_data[activity] = {
            "distribution": info["distribution"],
            "params": [float(p) for p in info["params"]],
            "ks_stat": float(info["ks_stat"]),
            "p_value": float(info["p_value"]),
        }

    json_path = filepath.replace(".pkl", ".json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved fitted distributions to {json_path} (human-readable)")


def load_distributions(filepath: str = "fitted_distributions.pkl"):
    """Load fitted distributions from file"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def sample_processing_time(activity: str, fitted_distributions: dict) -> float:
    """
    Sample a processing time from the fitted distribution

    Args:
        activity: Activity name
        fitted_distributions: Dictionary from fit_distributions()

    Returns:
        Processing time in seconds (always positive)
    """
    if activity not in fitted_distributions:
        # Fallback: return default time
        return 10.0

    info = fitted_distributions[activity]
    dist_name = info["distribution"]
    params = info["params"]

    # Sample from distribution
    if dist_name == "lognorm":
        shape, loc, scale = params
        sample = stats.lognorm.rvs(shape, loc, scale)
    elif dist_name == "expon":
        loc, scale = params
        sample = stats.expon.rvs(loc, scale)
    elif dist_name == "gamma":
        shape, loc, scale = params
        sample = stats.gamma.rvs(shape, loc, scale)
    elif dist_name == "norm":
        loc, scale = params
        sample = stats.norm.rvs(loc, scale)
    else:
        sample = 10.0

    # Ensure positive (in case of normal distribution)
    return max(0.01, sample)


def main():
    # Load log
    print("Loading event log...")
    df = load_event_log("BPI Challenge 2017.xes")

    print(f"Total events: {len(df)}")
    print(f"Unique activities: {df['concept:name'].nunique()}")
    print(f"Activities: {sorted(df['concept:name'].unique())}")

    # Extract processing times (hybrid approach)
    print("\n" + "=" * 70)
    print("EXTRACTING PROCESSING TIMES")
    print("=" * 70)
    actual_times, proxy_times = extract_processing_times_hybrid(df)

    print(f"\nW_ Activities with actual times: {len(actual_times)}")
    print(f"A_/O_ Activities with proxy times: {len(proxy_times)}")

    # Compute final processing times
    final_times = compute_final_processing_times(
        actual_times, proxy_times, percentile=25.0
    )

    # Show statistics
    show_statistics(final_times)

    # Fit distributions
    fitted_distributions = fit_distributions(final_times)

    # Save to file
    save_distributions(fitted_distributions)

    # Test sampling
    print("\n" + "=" * 70)
    print("TESTING SAMPLING")
    print("=" * 70)
    for activity in list(fitted_distributions.keys())[:5]:  # Test first 5
        samples = [
            sample_processing_time(activity, fitted_distributions) for _ in range(5)
        ]
        print(f"\n{activity}:")
        print(f"  Samples: {[f'{s:.2f}s' for s in samples]}")

    return fitted_distributions


if __name__ == "__main__":
    fitted_distributions = main()
