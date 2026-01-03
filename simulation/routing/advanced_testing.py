"""
advanced_testing.py

Exploratory test harness for BranchingAdvanced on the BPIC-17 log.

Design goals:
- Same overall "console report" style as basic_testing.py
- End-to-end: load XES -> dataframe -> discover Petri net -> token-replay fit -> inspect learned decision points
- Robust against small API differences (e.g., trace representation)

Notes:
- This script is intentionally defensive: it cross-checks enabled sets derived from the
  decision key against enabled sets returned by replay extraction.
"""

import pm4py
import pandas as pd

from pm4py.objects.conversion.log import converter as log_converter

from pm4py.algo.discovery.inductive import algorithm as inductive_miner

from branching_advanced import BranchingAdvanced, FitConfig


BPIC17_XES_PATH = "../../data/bpi-chall.xes"


# -----------------------------
# Load / prep
# -----------------------------
def load_bpic17_log_and_dataframe():
    print("\n[LOAD] Loading BPIC-17 XES log")
    log = pm4py.read_xes(BPIC17_XES_PATH)

    print("[LOAD] Converting log to dataframe")
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

    print("[LOAD] Dataframe shape:", df.shape)
    print("[LOAD] Dataframe columns:", list(df.columns))
    print("[LOAD] Head of dataframe:")
    print(df.head(5))

    return log, df


def print_basic_stats(df: pd.DataFrame):
    print("\n[STATS] Basic log statistics")

    case_col = "case:concept:name" if "case:concept:name" in df.columns else None
    act_col = "concept:name" if "concept:name" in df.columns else None

    if case_col:
        print("[STATS] Number of cases:", df[case_col].nunique())
    print("[STATS] Number of events:", len(df))

    if act_col:
        print("[STATS] Unique activities:", df[act_col].nunique())
        print("[STATS] Top 10 activities:")
        print(df[act_col].value_counts().head(10))

    if "org:resource" in df.columns:
        print("[STATS] Unique resources:", df["org:resource"].nunique())


# -----------------------------
# Petri net discovery
# -----------------------------
def discover_petri_net(log):
    print("\n[DISCOVER] Discovering Petri net with Inductive Miner")

    # Preferred: returns (net, im, fm) in many PM4Py versions
    try:
        net, im, fm = pm4py.discover_petri_net_inductive(log)
    except Exception:
        # Fallback: inductive miner returns a ProcessTree, then convert to Petri net
        tree = inductive_miner.apply(log)
        from pm4py.objects.conversion.process_tree import converter as pt_converter
        net, im, fm = pt_converter.apply(tree, variant=pt_converter.Variants.TO_PETRI_NET)

    print(
        "[DISCOVER] Net elements: places =",
        len(net.places),
        "| transitions =",
        len(net.transitions),
        "| arcs =",
        len(net.arcs),
    )
    print("[DISCOVER] Initial marking tokens:", sum(im.values()) if hasattr(im, "values") else 0)
    print("[DISCOVER] Final marking tokens:", sum(fm.values()) if hasattr(fm, "values") else 0)

    return net, im, fm



# -----------------------------
# Model fit
# -----------------------------
def fit_branching_model(df, net, initial_marking, final_marking):
    print("\n[MODEL] Initializing BranchingAdvanced")

    config = FitConfig(
        case_id_col="case:concept:name",
        activity_col="concept:name",
        timestamp_col="time:timestamp" if "time:timestamp" in df.columns else None,
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


# -----------------------------
# Inspection helpers
# -----------------------------
def inspect_model_overview(model: BranchingAdvanced, top_k=10):
    print("\n[MODEL] Overview")

    # Model internals are used only for reporting; this script does not modify them.
    learned = getattr(model, "_models", {})  # pylint: disable=protected-access
    empirical = getattr(model, "_empirical", {})  # pylint: disable=protected-access
    vocab = getattr(model, "_feature_index", {})  # pylint: disable=protected-access

    print("[MODEL] Decision points learned:", len(learned))
    print("[MODEL] Feature vocabulary size:", len(vocab))

    if not empirical:
        print("[MODEL] No empirical distributions stored (this is unexpected for a successful fit).")
        return

    totals = [(key, sum(cnts.values())) for key, cnts in empirical.items()]
    totals.sort(key=lambda x: -x[1])

    print(f"[MODEL] Top {min(top_k, len(totals))} decision points by training examples:")
    for key, n in totals[:top_k]:
        enabled_from_key = list(key[1])
        marking_sig = key[0]
        print("  - examples =", n, "| enabled =", enabled_from_key, "| marking_sig =", marking_sig)


def _predict_distribution(model: BranchingAdvanced, key, history):
    """
    Predict a probability distribution for a specific decision key and history.

    Returns dict[label -> prob] or None if the key is unknown.
    """
    model_entry = getattr(model, "_models", {}).get(key)  # pylint: disable=protected-access
    if model_entry is None:
        return None

    classes = model_entry["classes"]
    clf = model_entry["model"]

    x = model._vectorize(history, config=model.config)  # pylint: disable=protected-access
    probs = clf.predict_proba(x)

    # sklearn usually returns shape (1, n_classes); be robust
    if hasattr(probs, "shape") and len(probs.shape) == 2:
        probs = probs[0]

    return {cls: float(p) for cls, p in zip(classes, probs)}


def _iter_traces_for_inspection(model: BranchingAdvanced, df: pd.DataFrame):
    """
    Yield traces in a way compatible with multiple possible implementations.

    - If BranchingAdvanced._build_traces returns list[list[str]], we fabricate case ids.
    - If it returns list[dict] with case_id/events, we pass through.
    """
    work = model._prepare_dataframe(df, model.config)  # pylint: disable=protected-access
    traces = model._build_traces(work, model.config)  # pylint: disable=protected-access

    # Case 1: list of lists
    if traces and isinstance(traces[0], list):
        for i, seq in enumerate(traces, start=1):
            yield {"case_id": f"trace_{i}", "events": seq}

    # Case 2: already dict-like
    else:
        for tr in traces:
            if isinstance(tr, dict) and "events" in tr:
                yield tr
            else:
                # Last resort: treat as sequence
                yield {"case_id": "trace_unknown", "events": list(tr)}


def inspect_replay_decisions(model, df, net, initial_marking, max_cases=3, max_points_per_case=8, min_prob=0.01):
    print("\n[INSPECT] Inspecting decision points via replay (sample cases)")

    traces = list(_iter_traces_for_inspection(model, df))

    for ci, trace in enumerate(traces[:max_cases], start=1):
        case_id = trace["case_id"]
        events = trace["events"]

        print(f"\n[CASE] {ci} / {max_cases} | case_id = {case_id} | events = {len(events)}")

        # IMPORTANT: BranchingAdvanced expects a sequence of activity labels
        examples = model._replay_and_extract_examples(  # pylint: disable=protected-access
            events,
            net=net,
            initial_marking=initial_marking,
            config=model.config,
        )

        printed = 0
        for (key, history, enabled_labels, chosen) in examples:
            dist = _predict_distribution(model, key, history)
            if not dist:
                continue

            # Filter small probs for readability
            filtered = {k: v for k, v in dist.items() if v >= min_prob}
            if len(filtered) < 2:
                continue

            enabled_from_key = list(key[1])
            enabled_from_replay = list(enabled_labels)

            print("\n  [DECISION POINT]")
            print("   marking_sig:", key[0])
            print("   enabled:", enabled_from_key)

            # Consistency check: this is the main issue in the previous version
            if sorted(enabled_from_replay) != sorted(enabled_from_key):
                print("   enabled_from_replay:", enabled_from_replay)

            print("   history_tail:", list(history[-6:]))
            print("   actual_next:", chosen)

            print("   [DISTRIBUTION]")
            for lbl, prob in sorted(filtered.items(), key=lambda x: -x[1]):
                print("    ", lbl, "->", round(prob, 4))

            printed += 1
            if printed >= max_points_per_case:
                break

        if printed == 0:
            print("  [INFO] No decision points found in this case.")


def inspect_global_examples(model, df, net, initial_marking, max_points=15, min_prob=0.01):
    print("\n[INSPECT] Inspecting global decision-point examples (across log)")

    traces = list(_iter_traces_for_inspection(model, df))

    printed = 0
    for trace in traces:
        examples = model._replay_and_extract_examples(  # pylint: disable=protected-access
            trace["events"],
            net=net,
            initial_marking=initial_marking,
            config=model.config,
        )

        for (key, history, enabled_labels, chosen) in examples:
            dist = _predict_distribution(model, key, history)
            if not dist:
                continue

            filtered = {k: v for k, v in dist.items() if v >= min_prob}
            if len(filtered) < 2:
                continue

            enabled_from_key = list(key[1])
            enabled_from_replay = list(enabled_labels)

            print("\n  [EXAMPLE]")
            print("   marking_sig:", key[0])
            print("   enabled:", enabled_from_key)
            if sorted(enabled_from_replay) != sorted(enabled_from_key):
                print("   enabled_from_replay:", enabled_from_replay)
            print("   history_tail:", list(history[-6:]))
            print("   actual_next:", chosen)
            print("   [DISTRIBUTION]")
            for lbl, prob in sorted(filtered.items(), key=lambda x: -x[1]):
                print("    ", lbl, "->", round(prob, 4))

            printed += 1
            if printed >= max_points:
                return


# -----------------------------
# Main
# -----------------------------
def main():
    print("\n[START] BPIC-17 BranchingAdvanced exploratory testing")

    log, df = load_bpic17_log_and_dataframe()
    print_basic_stats(df)

    net, initial_marking, final_marking = discover_petri_net(log)

    model = fit_branching_model(df, net, initial_marking, final_marking)

    inspect_model_overview(model, top_k=10)
    inspect_replay_decisions(model, df, net, initial_marking, max_cases=3, max_points_per_case=8, min_prob=0.01)
    inspect_global_examples(model, df, net, initial_marking, max_points=15, min_prob=0.01)

    print("\n[END] Exploratory testing completed")


if __name__ == "__main__":
    main()
