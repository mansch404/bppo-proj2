"""
Empirical Evaluation of Basic Processing Time Distributions (Task 1.3 Basic)

Generates publication-ready plots and tables comparing fitted parametric
distributions against empirical (filtered) processing times from BPIC-2017.

Outputs (saved to simulation/timing/evaluation_outputs/):
  - basic_eval_histograms.png   Histogram + fitted PDF overlay per activity
  - basic_eval_qqplots.png      QQ-plots per activity
  - basic_eval_summary_table.csv  Summary statistics comparison + coverage
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest

# ---------------------------------------------------------------------------
# Data loading — replicated from basic_processing_time.py to avoid import
# issues (pm4py prints to stdout, interactive prompts, etc.)
# ---------------------------------------------------------------------------
import pm4py


def load_event_log(path: str) -> pd.DataFrame:
    log = pm4py.read_xes(path)
    return pm4py.convert_to_dataframe(log)


def extract_processing_times_hybrid(df: pd.DataFrame):
    actual_times: dict[str, list[float]] = {}
    proxy_times: dict[str, list[float]] = {}

    df = df.sort_values(["case:concept:name", "time:timestamp"])

    for _, case_df in df.groupby("case:concept:name"):
        case_df = case_df.reset_index(drop=True)
        for idx, row in case_df.iterrows():
            activity = row["concept:name"]
            lifecycle = row.get("lifecycle:transition", "complete")

            if lifecycle == "start":
                continue

            if activity.startswith("W_"):
                start_event = case_df[
                    (case_df["concept:name"] == activity)
                    & (case_df["lifecycle:transition"] == "start")
                    & (case_df.index < idx)
                ]
                if not start_event.empty:
                    start_time = start_event.iloc[-1]["time:timestamp"]
                    complete_time = row["time:timestamp"]
                    duration = (complete_time - start_time).total_seconds()
                    if duration > 0:
                        actual_times.setdefault(activity, []).append(duration)
                    continue

            if activity.startswith("W_"):
                continue

            if idx < len(case_df) - 1:
                next_time = case_df.iloc[idx + 1]["time:timestamp"]
                current_time = row["time:timestamp"]
                duration = (next_time - current_time).total_seconds()
                if duration > 0:
                    proxy_times.setdefault(activity, []).append(duration)

    return actual_times, proxy_times


def compute_final_processing_times(
    actual_times: dict, proxy_times: dict, percentile: float = 25.0
) -> dict[str, list[float]]:
    final_times: dict[str, list[float]] = {}

    for activity, times in actual_times.items():
        if not times:
            continue
        p25 = np.percentile(times, percentile)
        filtered = [t for t in times if t <= p25]
        if filtered:
            final_times[activity] = filtered

    for activity, times in proxy_times.items():
        if activity in final_times or not times:
            continue
        p25 = np.percentile(times, percentile)
        filtered = [t for t in times if t <= p25]
        if filtered:
            final_times[activity] = filtered

    return final_times


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------
DIST_MAP = {
    "lognorm": stats.lognorm,
    "expon": stats.expon,
    "gamma": stats.gamma,
    "norm": stats.norm,
}

# Representative activities (covering different distribution types)
SELECTED_ACTIVITIES = [
    "W_Validate application",   # gamma
    "W_Call after offers",      # lognorm
    "W_Complete application",   # gamma
    "A_Accepted",               # norm
    "A_Complete",               # norm
    "O_Created",                # norm
    "W_Handle leads",           # norm
    "W_Call incomplete files",  # lognorm
]

MIN_SAMPLES = 30  # skip activities with fewer samples


def _get_dist_obj(name: str, params: tuple):
    """Return a frozen scipy distribution object."""
    return DIST_MAP[name](*params)


def _smart_unit(data: np.ndarray):
    """Choose seconds or minutes for axis labels based on data magnitude."""
    median = np.median(data)
    if median > 120:
        return data / 60.0, "minutes"
    return data, "seconds"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _set_style():
    for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]:
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


def plot_histograms(activities, empirical, fitted, out_path: Path):
    n = len(activities)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for i, act in enumerate(activities):
        ax = axes[i // ncols, i % ncols]
        data = np.array(empirical[act])
        info = fitted[act]
        dist_name = info["distribution"]
        params = tuple(info["params"])

        data_plot, unit = _smart_unit(data)
        scale_factor = 60.0 if unit == "minutes" else 1.0

        # Histogram
        ax.hist(data_plot, bins="auto", density=True, alpha=0.6,
                color="steelblue", edgecolor="white", label="Empirical")

        # Fitted PDF
        frozen = _get_dist_obj(dist_name, params)
        x_min, x_max = data_plot.min(), data_plot.max()
        x = np.linspace(max(0, x_min - 0.1 * (x_max - x_min)),
                        x_max + 0.1 * (x_max - x_min), 300)
        # pdf in original seconds, then convert
        pdf_vals = frozen.pdf(x * scale_factor) * scale_factor
        ax.plot(x, pdf_vals, "r-", lw=2, label=f"Fitted ({dist_name})")

        ks_stat = info["ks_stat"]
        ax.set_title(f"{act}\n{dist_name}, KS={ks_stat:.4f}", fontsize=10)
        ax.set_xlabel(f"Processing time ({unit})", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram overlay plot to {out_path}")


def plot_qqplots(activities, empirical, fitted, out_path: Path):
    n = len(activities)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for i, act in enumerate(activities):
        ax = axes[i // ncols, i % ncols]
        data = np.sort(np.array(empirical[act]))
        info = fitted[act]
        dist_name = info["distribution"]
        params = tuple(info["params"])
        frozen = _get_dist_obj(dist_name, params)

        n_pts = len(data)
        theoretical_q = frozen.ppf(np.linspace(1 / (n_pts + 1),
                                               n_pts / (n_pts + 1), n_pts))

        data_plot, unit = _smart_unit(data)
        scale_factor = 60.0 if unit == "minutes" else 1.0
        theo_plot = theoretical_q / scale_factor

        ax.scatter(theo_plot, data_plot, s=8, alpha=0.5, color="steelblue")
        all_vals = np.concatenate([theo_plot, data_plot])
        lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
        margin = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", lw=1.5, label="Perfect fit")

        ax.set_title(f"{act}\n({dist_name})", fontsize=10)
        ax.set_xlabel(f"Theoretical quantiles ({unit})", fontsize=9)
        ax.set_ylabel(f"Empirical quantiles ({unit})", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved QQ-plot to {out_path}")


# ---------------------------------------------------------------------------
# Summary statistics + coverage
# ---------------------------------------------------------------------------

def build_summary_table(activities, empirical, fitted) -> pd.DataFrame:
    rows = []
    np.random.seed(42)

    for act in activities:
        data = np.array(empirical[act])
        info = fitted[act]
        dist_name = info["distribution"]
        params = tuple(info["params"])
        frozen = _get_dist_obj(dist_name, params)

        samples = frozen.rvs(size=10_000)
        samples = samples[samples > 0]  # drop negatives from norm

        # Coverage: % of empirical data in [P5, P95] of fitted distribution
        p5 = frozen.ppf(0.05)
        p95 = frozen.ppf(0.95)
        coverage = np.mean((data >= p5) & (data <= p95)) * 100

        ks_stat, _ = kstest(data, lambda x: frozen.cdf(x))

        rows.append({
            "Activity": act,
            "Distribution": dist_name,
            "N_samples": len(data),
            "Emp_Mean": np.mean(data),
            "Fit_Mean": np.mean(samples),
            "Emp_Median": np.median(data),
            "Fit_Median": np.median(samples),
            "Emp_Std": np.std(data),
            "Fit_Std": np.std(samples),
            "Emp_P25": np.percentile(data, 25),
            "Fit_P25": np.percentile(samples, 25),
            "Emp_P75": np.percentile(data, 75),
            "Fit_P75": np.percentile(samples, 75),
            "Coverage_90pct": round(coverage, 1),
            "KS_Statistic": round(ks_stat, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "bpi-chall.xes"
    pkl_path = project_root / "fitted_distributions.pkl"
    out_dir = Path(__file__).resolve().parent / "evaluation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load fitted distributions
    print("Loading fitted distributions...")
    with open(pkl_path, "rb") as f:
        fitted = pickle.load(f)

    # 2. Load and process event log (same pipeline as basic_processing_time.py)
    print("Loading event log (this may take a minute)...")
    df = load_event_log(str(data_path))
    actual_times, proxy_times = extract_processing_times_hybrid(df)
    final_times = compute_final_processing_times(actual_times, proxy_times, percentile=25.0)

    # 3. Select activities that exist in both empirical data and fitted dists
    activities = [
        a for a in SELECTED_ACTIVITIES
        if a in final_times and a in fitted and len(final_times[a]) >= MIN_SAMPLES
    ]
    print(f"\nEvaluating {len(activities)} activities: {activities}")

    if not activities:
        print("ERROR: No activities available for evaluation.")
        return

    _set_style()

    # 4. Generate outputs
    plot_histograms(activities, final_times, fitted,
                    out_dir / "basic_eval_histograms.png")

    plot_qqplots(activities, final_times, fitted,
                 out_dir / "basic_eval_qqplots.png")

    summary = build_summary_table(activities, final_times, fitted)

    # Print to console
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS: Empirical vs Fitted Distributions")
    print("=" * 120)
    with pd.option_context("display.max_columns", None, "display.width", 120,
                           "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))

    # Save CSV
    csv_path = out_dir / "basic_eval_summary_table.csv"
    summary.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved summary table to {csv_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
