import pm4py
import pandas as pd

from pm4py.objects.conversion.log import converter as log_converter

from branching_advanced import BranchingAdvanced, FitConfig


BPIC17_XES_PATH = "../../data/bpi-chall.xes"


def load_bpic17_log_and_dataframe():
    print("\n[LOAD] Loading BPIC-17 XES log")
    log = pm4py.read_xes(BPIC17_XES_PATH)

    print("[LOAD] Converting log to dataframe")
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

    print("[LOAD] Dataframe shape:", df.shape)
    print("[LOAD] Dataframe columns:", list(df.columns))
    print("[LOAD] Head of dataframe:")
    print(df.head())

    return log, df


def discover_petri_net(log):
    print("\n[DISCOVER] Discovering Petri net with Inductive Miner")
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log)

    print("[DISCOVER] Net elements:",
          "places =", len(net.places),
          "| transitions =", len(net.transitions),
          "| arcs =", len(net.arcs))
    print("[DISCOVER] Initial marking tokens:", len(initial_marking))
    if final_marking is not None:
        print("[DISCOVER] Final marking tokens:", len(final_marking))

    return net, initial_marking, final_marking


def inspect_basic_statistics(df):
    print("\n[STATS] Basic log statistics")

    print("[STATS] Number of cases:",
          df["case:concept:name"].nunique())

    print("[STATS] Number of events:",
          len(df))

    if "concept:name" in df.columns:
        print("[STATS] Unique activities:",
              df["concept:name"].nunique())
        print("[STATS] Top 10 activities:")
        print(df["concept:name"].value_counts().head(10))

    if "org:resource" in df.columns:
        print("[STATS] Unique resources:",
              df["org:resource"].nunique())


def fit_branching_model(df, net, initial_marking, final_marking):
    print("\n[MODEL] Initializing BranchingAdvanced")

    # Keep configuration explicit so results are reproducible and easy to justify in the report
    config = FitConfig(
        case_id_col="case:concept:name",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        lifecycle_col="lifecycle:transition" if "lifecycle:transition" in df.columns else None,
        keep_only_complete=True,
        max_tau_steps_per_event=50,
        max_total_steps_per_case=5000,
        use_bag_of_activities=True,
        use_last_activity=True,
        use_last_bigram=True,
        alpha=1.0,
    )

    model = BranchingAdvanced(seed=42, config=config)

    print("[MODEL] Fitting model from dataframe using token replay")
    model = model.fit_from_dataframe(
        df,
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        config=config,
    )

    print("[MODEL] Model fitting complete")
    return model


def inspect_model_overview(model, top_k=10):
    print("\n[MODEL] Overview")

    # These are implementation internals; they are useful for debugging and reporting,
    # and are stable enough for this uni project testing file.
    n_decisions = len(getattr(model, "_models", {}))
    n_features = len(getattr(model, "_idx2feat", []))

    print("[MODEL] Decision points learned:", n_decisions)
    print("[MODEL] Feature vocabulary size:", n_features)

    empirical = getattr(model, "_empirical", {})
    if empirical:
        totals = []
        for key, cnts in empirical.items():
            totals.append((key, sum(cnts.values())))
        totals.sort(key=lambda x: -x[1])

        print(f"[MODEL] Top {min(top_k, len(totals))} decision points by training examples:")
        for key, n in totals[:top_k]:
            enabled = list(key[1])
            marking_sig = key[0]
            ms_str = str(marking_sig)
            print("  - examples =", n, "| enabled =", enabled, "| marking_sig =", ms_str[:60] + ("..." if len(ms_str) > 60 else ""))



def _predict_distribution_for_example(model, key, history):
    model_entry = model._models.get(key)  # pylint: disable=protected-access
    if model_entry is None:
        return None

    classes = model_entry["classes"]
    nb = model_entry["model"]

    x = model._vectorize(history, config=model.config)  # pylint: disable=protected-access
    probs = nb.predict_proba(x)
    row = probs[0] if hasattr(probs, "__len__") and len(probs) > 0 else probs

    return {cls: float(p) for cls, p in zip(classes, row)}
def _fmt_any(x, max_len=80):
    s = x if isinstance(x, str) else repr(x)
    return s[:max_len] + ("..." if len(s) > max_len else "")


def inspect_replay_decisions(model, df, net, initial_marking, max_cases=3, max_points_per_case=8, min_prob=0.01):
    print("\n[INSPECT] Inspecting decision points via replay (sample cases)")

    # Reuse the model's preprocessing so this testing script stays consistent with training
    work = model._prepare_dataframe(df, model.config)  # pylint: disable=protected-access

    case_col = model.config.case_id_col
    act_col = model.config.activity_col

    for ci, (case_id, g) in enumerate(work.groupby(case_col, sort=False), start=1):
        if ci > max_cases:
            break

        trace = g[act_col].tolist()
        print(f"\n[CASE] {ci} / {max_cases} | case_id = {case_id} | events = {len(trace)}")

        examples = model._replay_and_extract_examples(  # pylint: disable=protected-access
            trace,
            net=net,
            initial_marking=initial_marking,
            config=model.config,
        )

        if not examples:
            print("  [INFO] No decision points found in this case.")
            continue

        shown = 0
        for key, enabled_labels, history, chosen in examples:
            if shown >= max_points_per_case:
                print(f"  [INFO] Reached max_points_per_case = {max_points_per_case}")
                break

            dist = _predict_distribution_for_example(model, key, history)
            if dist is None:
                # Not all decision keys encountered in replay will necessarily be trained
                continue

            filtered = {k: v for k, v in dist.items() if v >= min_prob}
            if len(filtered) < 2:
                continue

            shown += 1
            print("\n  [DECISION POINT]")
            print("   marking_sig:", _fmt_any(key[0], max_len=80))
            print("   enabled:", list(enabled_labels))
            print("   history_tail:", list(history[-6:]))
            print("   actual_next:", chosen)

            print("   [DISTRIBUTION]")
            for k, v in sorted(filtered.items(), key=lambda x: -x[1]):
                print("    ", k, "->", round(v, 4))

        if shown == 0:
            print("  [INFO] No multi-option distributions above threshold in this case.")

def main():
    print("\n[START] BPIC-17 BranchingAdvanced exploratory testing")

    log, df = load_bpic17_log_and_dataframe()
    inspect_basic_statistics(df)

    net, initial_marking, final_marking = discover_petri_net(log)

    model = fit_branching_model(df, net, initial_marking, final_marking)

    inspect_model_overview(model, top_k=10)
    inspect_replay_decisions(model, df, net, initial_marking, max_cases=3, max_points_per_case=8, min_prob=0.01)
    #inspect_global_examples(model, df, net, initial_marking, max_points=15, min_prob=0.01)

    print("\n[END] Exploratory testing completed")


if __name__ == "__main__":
    main()