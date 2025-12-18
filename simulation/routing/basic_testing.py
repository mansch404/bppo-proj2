import pandas as pd
import math

from branching_basic import BranchingBasic


def _build_simple_xor_dataframe():
    print("\n[BUILD] Creating simple XOR dataframe")
    df = pd.DataFrame(
        {
            "case:concept:name": ["1", "1", "1", "2", "2", "2", "3", "3", "3"],
            "concept:name": ["A", "X", "C", "A", "Y", "C", "A", "X", "C"],
            "time:timestamp": pd.date_range("2020-01-01", periods=9, freq="min"),
            "lifecycle:transition": ["complete"] * 9,
        }
    )
    print(df)
    return df


def test_learns_correct_xor_distribution():
    print("\n[TEST] test_learns_correct_xor_distribution")
    df = _build_simple_xor_dataframe()

    model = BranchingBasic(max_context=1, seed=42)
    print("[MODEL] Initialized:", model)

    model = model.fit_from_dataframe(df)
    print("[MODEL] Fitted model")

    history = ["A"]
    enabled = ["X", "Y"]

    print("[QUERY] history =", history)
    print("[QUERY] enabled_next =", enabled)

    dist = model.get_distribution(history=history, enabled_next=enabled)

    print("[RESULT] distribution =", dist)

    assert dist is not None
    assert set(dist.keys()) == {"X", "Y"}

    print("[EXPECT] X =", 2 / 3, "Y =", 1 / 3)
    assert math.isclose(dist["X"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(dist["Y"], 1 / 3, rel_tol=1e-6)


def test_distribution_is_normalized():
    print("\n[TEST] test_distribution_is_normalized")
    df = _build_simple_xor_dataframe()

    model = BranchingBasic(max_context=1).fit_from_dataframe(df)
    dist = model.get_distribution(history=["A"], enabled_next=["X", "Y"])

    total = sum(dist.values())
    print("[CHECK] distribution =", dist)
    print("[CHECK] sum =", total)

    assert abs(total - 1.0) < 1e-9


def test_filters_to_enabled_successors_only():
    print("\n[TEST] test_filters_to_enabled_successors_only")
    df = _build_simple_xor_dataframe()
    model = BranchingBasic(max_context=1).fit_from_dataframe(df)

    history = ["A"]
    enabled = ["X"]

    print("[QUERY] history =", history)
    print("[QUERY] enabled_next =", enabled)

    dist = model.get_distribution(history=history, enabled_next=enabled)
    print("[RESULT] distribution =", dist)

    assert dist is not None
    assert set(dist.keys()) == {"X"}
    assert math.isclose(dist["X"], 1.0, rel_tol=1e-9)


def test_returns_singleton_for_non_xor():
    print("\n[TEST] test_returns_singleton_for_non_xor")
    df = _build_simple_xor_dataframe()
    model = BranchingBasic(max_context=1).fit_from_dataframe(df)

    dist = model.get_distribution(history=["A"], enabled_next=["X"])
    print("[RESULT] distribution =", dist)

    assert dist is not None
    assert len(dist) == 1


def test_unseen_context_fallback():
    print("\n[TEST] test_unseen_context_fallback")
    df = _build_simple_xor_dataframe()
    model = BranchingBasic(max_context=1).fit_from_dataframe(df)

    history = ["Z"]
    enabled = ["X", "Y"]

    print("[QUERY] history =", history)
    print("[QUERY] enabled_next =", enabled)

    dist = model.get_distribution(history=history, enabled_next=enabled)
    print("[RESULT] distribution =", dist)

    assert dist is not None
    assert set(dist.keys()) == {"X", "Y"}

    print("[EXPECT] fallback to global frequencies: X=2/3, Y=1/3")
    assert math.isclose(dist["X"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(dist["Y"], 1 / 3, rel_tol=1e-6)



def test_reproducibility_with_seed():
    print("\n[TEST] test_reproducibility_with_seed")
    df = _build_simple_xor_dataframe()

    model_1 = BranchingBasic(max_context=1, seed=7).fit_from_dataframe(df)
    model_2 = BranchingBasic(max_context=1, seed=7).fit_from_dataframe(df)

    dist_1 = model_1.get_distribution(history=["A"], enabled_next=["X", "Y"])
    dist_2 = model_2.get_distribution(history=["A"], enabled_next=["X", "Y"])

    print("[RESULT] distribution 1 =", dist_1)
    print("[RESULT] distribution 2 =", dist_2)

    assert dist_1 == dist_2


if __name__ == "__main__":
    print("\n[INFO] Running tests manually (without pytest)")
    test_learns_correct_xor_distribution()
    test_distribution_is_normalized()
    test_filters_to_enabled_successors_only()
    test_returns_singleton_for_non_xor()
    test_unseen_context_fallback()
    test_reproducibility_with_seed()
    print("\n[INFO] All manual tests completed")
