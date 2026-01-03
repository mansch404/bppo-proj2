import pm4py
import pandas as pd

from pm4py.objects.conversion.log import converter as log_converter
from branching_basic import BranchingBasic


def load_bpic17_dataframe():
    print("\n[LOAD] Loading BPIC-17 XES log")
    log = pm4py.read_xes("../../data/bpi-chall.xes")

    print("[LOAD] Converting log to dataframe")
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

    print("[LOAD] Dataframe shape:", df.shape)
    print("[LOAD] Dataframe columns:", list(df.columns))
    print("[LOAD] Head of dataframe:")
    print(df.head())

    return df


def inspect_basic_statistics(df):
    print("\n[STATS] Basic log statistics")

    print("[STATS] Number of cases:",
          df["case:concept:name"].nunique())

    print("[STATS] Number of events:",
          len(df))

    print("[STATS] Number of unique activities:",
          df["concept:name"].nunique())

    print("[STATS] Most frequent activities:")
    print(df["concept:name"].value_counts().head(10))


def fit_branching_model(df):
    print("\n[MODEL] Initializing BranchingBasic")
    model = BranchingBasic(max_context=1, seed=42)

    print("[MODEL] Fitting model from dataframe")
    model = model.fit_from_dataframe(df)

    print("[MODEL] Model fitting complete")
    return model


def inspect_xor_branching_points(
    model,
    df,
    min_successors=2,
    min_prob=0.01,
    max_points=20,
    skip_self_loops=True,
):
    print("\n[INSPECT] Detecting XOR branching points automatically")

    # Step 1: compute empirical successor sets
    df_sorted = df.sort_values(
        ["case:concept:name", "time:timestamp"]
    )

    df_sorted["next_activity"] = (
        df_sorted
        .groupby("case:concept:name")["concept:name"]
        .shift(-1)
    )

    successor_table = (
        df_sorted
        .dropna(subset=["next_activity"])
        .groupby("concept:name")["next_activity"]
        .unique()
    )

    xor_candidates = {
        act: succs.tolist()
        for act, succs in successor_table.items()
        if len(succs) >= min_successors
    }

    print("[INFO] XOR candidates found:", len(xor_candidates))

    printed = 0

    for activity, successors in xor_candidates.items():
        if printed >= max_points:
            break

        if skip_self_loops:
            successors = [s for s in successors if s != activity]

        if len(successors) < min_successors:
            continue

        print("\n[BRANCH POINT]")
        print("  Activity:", activity)
        print("  Enabled successors:", successors)

        dist = model.get_distribution(
            history=[activity],
            enabled_next=successors
        )

        if dist is None:
            print("  [SKIP] Model returned None")
            continue

        # Filter negligible probabilities
        filtered = {
            k: v for k, v in dist.items()
            if v >= min_prob
        }

        if len(filtered) < 2:
            print("  [SKIP] Degenerate distribution")
            continue

        print("  [DISTRIBUTION]")
        for k, v in sorted(
            filtered.items(),
            key=lambda x: -x[1]
        ):
            print("   ", k, "->", round(v, 4))

        printed += 1


def inspect_xor_decisions(model, df, max_cases=5):
    print("\n[INSPECT] Inspecting XOR-style branching decisions")

    cases = df["case:concept:name"].unique()[:max_cases]

    for case_id in cases:
        print("\n[CASE]", case_id)

        trace = (
            df[df["case:concept:name"] == case_id]
            .sort_values("time:timestamp")
        )

        activities = trace["concept:name"].tolist()
        print("[TRACE] Activities:", activities)

        for i in range(len(activities) - 1):
            history = [activities[i]]

            enabled_next = list(
                df[df["concept:name"] == activities[i + 1]]
                ["concept:name"]
                .unique()
            )

            print("\n[QUERY]")
            print("  history =", history)
            print("  enabled_next =", enabled_next)

            try:
                dist = model.get_distribution(
                    history=history,
                    enabled_next=enabled_next
                )
            except Exception as e:
                print("[ERROR] Exception during get_distribution:", e)
                continue

            print("[RESULT] distribution =", dist)


def inspect_common_branch_points(model, df, top_k=10):
    print("\n[INSPECT] Inspecting most common branching points")

    next_counts = (
        df.groupby("concept:name")["case:concept:name"]
        .count()
        .sort_values(ascending=False)
        .head(top_k)
    )

    for activity in next_counts.index:
        print("\n[ACTIVITY]", activity)

        enabled_next = (
            df[df["concept:name"] != activity]["concept:name"]
            .unique()
            .tolist()
        )

        history = [activity]

        print("[QUERY]")
        print("  history =", history)
        print("  enabled_next (sample size) =", len(enabled_next))

        dist = model.get_distribution(
            history=history,
            enabled_next=enabled_next
        )

        print("[RESULT] distribution (truncated):")
        if dist is None:
            print("  None")
        else:
            for k, v in list(dist.items())[:10]:
                print(" ", k, "->", v)


def main():
    print("\n[START] BPIC-17 BranchingBasic exploratory testing")

    df = load_bpic17_dataframe()
    inspect_basic_statistics(df)

    model = fit_branching_model(df)

    inspect_xor_decisions(model, df, max_cases=3)
    inspect_common_branch_points(model, df, top_k=5)

    inspect_xor_branching_points(
        model,
        df,
        min_successors=2,
        min_prob=0.01,
        max_points=15
    )

    print("\n[END] Exploratory testing completed")


if __name__ == "__main__":
    main()
