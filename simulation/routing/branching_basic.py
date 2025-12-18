
"""
simulation.routing.branching_basic
==================================

Basic XOR branching decision logic for the BPIC-17 based simulator.

Assignment requirement (Task 1.4 - Basic)
- Learn branching probabilities from the event log.
- Use those probabilities during simulation at XOR decision points.
- The main challenge is mapping "relevant traces" to "relevant branches".
  This implementation addresses that by learning conditional probabilities
  over contexts (n-grams of preceding activities) and applying them only
  when a decision point actually exists (i.e., multiple enabled successors).

Design principles
- Log-driven: learns from the observed directly-follows relations.
- Contextual (basic): conditions on the last k activities (default k=3),
  with backoff to shorter contexts.
- Engine-agnostic: can be called from any engine component with just:
    - a process instance history (list[str])
    - the currently enabled next activities (list[str])
- Persistable: can save/load probabilities to a pickle file, matching the
  proposed Structure.md layout (data/interim/routing_probabilities.pkl).

Typical usage
-------------
Training / fitting:
    model = BranchingBasic(max_context=3)
    model.fit_from_dataframe(df, case_id_col="case:concept:name",
                             activity_col="concept:name",
                             timestamp_col="time:timestamp")

Simulation:
    next_act = model.choose_next(history=instance.history,
                                 enabled_next=["A", "B", "C"])
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

try:
    import pm4py
except Exception:  # pragma: no cover
    pm4py = None


Context = Tuple[str, ...]          # e.g., ("A", "B")
Activity = str
Probabilities = Dict[Activity, float]


@dataclass(frozen=True)
class FitConfig:
    """Configuration used for fitting transition statistics."""
    case_id_col: str = "case:concept:name"
    activity_col: str = "concept:name"
    timestamp_col: str = "time:timestamp"
    lifecycle_col: Optional[str] = "lifecycle:transition"
    keep_only_complete: bool = True
    start_token: str = "__START__"
    end_token: str = "__END__"


class BranchingBasic:
    """
    Basic branching model based on conditional next-activity probabilities.

    The model estimates:
        P(next_activity | last_k_activities)

    It stores counts per context length 0..max_context:
        - context length 0 = unconditional distribution over next activities
        - context length 1 = conditioned on last 1 activity
        - ...
        - context length k = conditioned on last k activities

    During prediction, it uses the longest available context and backs off
    until it finds probabilities that intersect the enabled_next set.
    """

    def __init__(
        self,
        max_context: int = 3,
        seed: Optional[int] = None,
        laplace_alpha: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        max_context:
            Maximum number of preceding activities to condition on.
            A value of 1 corresponds to a first-order Markov model.
        seed:
            Random seed used when sampling from categorical distributions.
        laplace_alpha:
            Optional Laplace/additive smoothing. Use 0.0 for no smoothing.
            If enabled, unseen (context,next) pairs get small probability mass.
        """
        if max_context < 0:
            raise ValueError("max_context must be >= 0")
        if laplace_alpha < 0:
            raise ValueError("laplace_alpha must be >= 0")

        self.max_context = int(max_context)
        self.random = random.Random(seed)
        self.laplace_alpha = float(laplace_alpha)

        # counts[k][context][next_activity] = int
        self._counts: List[Dict[Context, Dict[Activity, int]]] = [
            {} for _ in range(self.max_context + 1)
        ]
        # probs[k][context][next_activity] = float
        self._probs: List[Dict[Context, Probabilities]] = [
            {} for _ in range(self.max_context + 1)
        ]

        # A convenience set of observed activities (used for smoothing / validation)
        self._activity_vocab: set[str] = set()

        # Metadata for trace boundary handling
        self._fit_config: Optional[FitConfig] = None

    # ---------------------------------------------------------------------
    # Publicly Usable API
    # ---------------------------------------------------------------------

    def fit_from_event_log(
        self,
        path: str | Path,
        *,
        config: Optional[FitConfig] = None,
    ) -> "BranchingBasic":
        """
        Compatible with both xes and csv

        Notes
        -----
        - .xes: pm4py.read_xes is used
        - .csv: Using pandas.read_csv.
        """
        path = Path(path)
        config = config or FitConfig()
        df = self._read_event_log(path, config=config)
        return self.fit_from_dataframe(df, config=config)

    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        config: Optional[FitConfig] = None,
    ) -> "BranchingBasic":
        """
        Fit the model from a DataFrame.

        Expected columns (default names are in FitConfig):
        - case_id_col
        - activity_col
        - timestamp_col
        - lifecycle_col (optional)

        The DataFrame may contain start/complete pairs. If keep_only_complete=True,
        only 'complete' events are used.
        """
        config = config or FitConfig()
        self._fit_config = config

        df_prepped = self._prepare_dataframe(df, config)
        traces = self._build_traces(df_prepped, config)

        self._reset_counts()
        self._fit_from_traces(traces, config)
        self._compute_probabilities()

        return self

    def choose_next(
        self,
        *,
        history: Sequence[str],
        enabled_next: Sequence[str],
        default_strategy: str = "uniform",
    ) -> str:
        """
        Choose the next activity at an XOR decision point.

        Parameters
        ----------
        history:
            Sequence of executed activities, in order (excluding lifecycle states).
        enabled_next:
            Activities that are currently enabled by the process model.
            If the gateway is XOR, this set should contain >1 candidates.
        default_strategy:
            Behaviour if no probability information is available for the
            current situation. Options:
              - "uniform": uniform random over enabled_next (default)
              - "first": return enabled_next[0]
              - "error": raise RuntimeError

        Returns
        -------
        str
            Chosen activity name.
        """
        if not enabled_next:
            raise ValueError("enabled_next must be non-empty")

        enabled_set = set(enabled_next)

        # Use longest context first, backoff to shorter ones.
        for k in range(min(self.max_context, len(history)), -1, -1):
            ctx = tuple(history[-k:]) if k > 0 else tuple()
            probs_for_ctx = self._probs[k].get(ctx)
            if not probs_for_ctx:
                continue

            filtered = {a: p for a, p in probs_for_ctx.items() if a in enabled_set}
            if not filtered:
                continue

            return self._sample_categorical(filtered)

        # Fallback
        return self._fallback_choice(enabled_next, default_strategy)

    def get_distribution(
        self,
        *,
        history: Sequence[str],
        enabled_next: Optional[Sequence[str]] = None,
    ) -> Probabilities:
        """
        Return the predicted probability distribution for the next activity.

        If enabled_next is provided, the distribution is filtered and re-normalized.

        If no distribution is available, returns {}.
        """
        enabled_set = set(enabled_next) if enabled_next is not None else None

        for k in range(min(self.max_context, len(history)), -1, -1):
            ctx = tuple(history[-k:]) if k > 0 else tuple()
            probs_for_ctx = self._probs[k].get(ctx)
            if not probs_for_ctx:
                continue

            if enabled_set is None:
                return dict(probs_for_ctx)

            filtered = {a: p for a, p in probs_for_ctx.items() if a in enabled_set}
            if not filtered:
                continue
            return self._renormalize(filtered)

        return {}

    def save(self, path: str | Path) -> None:
        """Persist learned probabilities and configuration to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "max_context": self.max_context,
            "laplace_alpha": self.laplace_alpha,
            "fit_config": self._fit_config,
            "activity_vocab": sorted(self._activity_vocab),
            "counts": self._counts,
            "probs": self._probs,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, *, seed: Optional[int] = None) -> "BranchingBasic":
        """Load a model from a pickle file."""
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        model = cls(
            max_context=int(payload["max_context"]),
            seed=seed,
            laplace_alpha=float(payload.get("laplace_alpha", 0.0)),
        )
        model._fit_config = payload.get("fit_config")
        model._activity_vocab = set(payload.get("activity_vocab", []))
        model._counts = payload.get("counts", model._counts)
        model._probs = payload.get("probs", model._probs)
        return model

    # ---------------------------------------------------------------------
    # Fitting Internal Functions
    # ---------------------------------------------------------------------

    def _reset_counts(self) -> None:
        self._counts = [{} for _ in range(self.max_context + 1)]
        self._probs = [{} for _ in range(self.max_context + 1)]
        self._activity_vocab = set()

    @staticmethod
    def _read_event_log(path: Path, *, config: FitConfig) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".xes":
            if pm4py is None:
                raise ImportError("Could not read xes file, missing pm4py")
            log = pm4py.read_xes(str(path))
            return pm4py.convert_to_dataframe(log)
        raise ValueError(f"Unsupported event log type: {path.suffix}")

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame, config: FitConfig) -> pd.DataFrame:
        # Validate columns early for clearer error messages.
        required = {config.case_id_col, config.activity_col, config.timestamp_col}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Event log DataFrame missing required columns: {missing}")

        work = df.copy()

        # Timestamp ordering is crucial.
        work[config.timestamp_col] = pd.to_datetime(work[config.timestamp_col], utc=True, errors="coerce")
        work = work.dropna(subset=[config.timestamp_col])

        # Filter lifecycle (if present and desired)
        if config.keep_only_complete and config.lifecycle_col and config.lifecycle_col in work.columns:
            work = work[work[config.lifecycle_col].astype(str).str.lower() == "complete"]

        # Keep only columns needed for building traces
        cols = [config.case_id_col, config.activity_col, config.timestamp_col]
        work = work[cols].sort_values([config.case_id_col, config.timestamp_col])

        # Drop missing activities / case ids
        work = work.dropna(subset=[config.case_id_col, config.activity_col])
        work[config.case_id_col] = work[config.case_id_col].astype(str)
        work[config.activity_col] = work[config.activity_col].astype(str)

        return work

    @staticmethod
    def _build_traces(df: pd.DataFrame, config: FitConfig) -> List[List[str]]:
        """
        Convert an event log dataframe to an ordered list of traces (activity sequences).

        Adds explicit start/end tokens so the model can learn which activities tend to
        begin or terminate cases.
        """
        traces: List[List[str]] = []
        for _, group in df.groupby(config.case_id_col, sort=False):
            seq = group[config.activity_col].tolist()
            if not seq:
                continue
            traces.append([config.start_token] + seq + [config.end_token])
        return traces

    def _fit_from_traces(self, traces: Sequence[Sequence[str]], config: FitConfig) -> None:
        """
        Populate counts for all context lengths.

        For each position i (predicting next activity at i), we update counts for
        contexts of length k = 0..max_context:
            context = last k activities before next
            next = activity at i
        """
        for trace in traces:
            # update vocabulary excluding boundary tokens
            for a in trace:
                if a not in (config.start_token, config.end_token):
                    self._activity_vocab.add(a)

            for i in range(1, len(trace)):
                nxt = trace[i]
                # contexts draw from trace[0:i]
                prefix = trace[:i]

                for k in range(0, self.max_context + 1):
                    if k == 0:
                        ctx: Context = tuple()
                    else:
                        ctx = tuple(prefix[-k:]) if len(prefix) >= k else tuple(prefix)

                    ctx_map = self._counts[k].setdefault(ctx, {})
                    ctx_map[nxt] = ctx_map.get(nxt, 0) + 1

    def _compute_probabilities(self) -> None:
        """
        Convert counts to probabilities.

        If laplace_alpha > 0, apply additive smoothing using the activity vocab plus
        the end token (so termination is still possible).
        """
        vocab: List[str] = sorted(self._activity_vocab)
        if self._fit_config is None:
            end_token = "__END__"
        else:
            end_token = self._fit_config.end_token
        if end_token not in vocab:
            vocab_plus = vocab + [end_token]
        else:
            vocab_plus = vocab

        for k in range(0, self.max_context + 1):
            for ctx, next_counts in self._counts[k].items():
                total = float(sum(next_counts.values()))
                if total <= 0:
                    continue

                if self.laplace_alpha > 0:
                    # Additive smoothing over vocab_plus
                    smoothed: Dict[str, float] = {}
                    denom = total + self.laplace_alpha * len(vocab_plus)
                    for a in vocab_plus:
                        c = float(next_counts.get(a, 0))
                        smoothed[a] = (c + self.laplace_alpha) / denom
                    self._probs[k][ctx] = smoothed
                else:
                    self._probs[k][ctx] = {a: c / total for a, c in next_counts.items()}

    # ---------------------------------------------------------------------
    # Sampling and fallbacks
    # ---------------------------------------------------------------------

    def _sample_categorical(self, probs: Mapping[str, float]) -> str:
        """Sample an outcome from a categorical distribution."""
        # Normalize defensively (in case of float drift)
        p = self._renormalize(dict(probs))

        r = self.random.random()
        cumulative = 0.0
        last_key = None
        for key, prob in p.items():
            last_key = key
            cumulative += prob
            if r <= cumulative:
                return key

        # shoudl not happen, but return the last key for numerical safety.
        if last_key is None:
            raise RuntimeError("Cannot sample from an empty distribution.")
        return last_key

    @staticmethod
    def _renormalize(probs: Dict[str, float]) -> Dict[str, float]:
        s = float(sum(probs.values()))
        if s <= 0:
            return {}
        return {k: v / s for k, v in probs.items()}

    def _fallback_choice(self, enabled_next: Sequence[str], strategy: str) -> str:
        strategy = strategy.lower().strip()
        if strategy == "uniform":
            return self.random.choice(list(enabled_next))
        if strategy == "first":
            return enabled_next[0]
        if strategy == "error":
            raise RuntimeError(
                "No branching probabilities available for this context and enabled set."
            )
        raise ValueError(f"Unknown default_strategy: {strategy}")

    # ---------------------------------------------------------------------
    # Convenience: compute decision points from learned transitions
    # ---------------------------------------------------------------------

    def learned_successors(self, activity: str) -> List[str]:
        """
        Return the set of successors observed after `activity` in the training log.

        This is derived from the 1-context model (k=1) and ignores longer contexts.
        """
        if self._fit_config is None:
            start_token = "__START__"
        else:
            start_token = self._fit_config.start_token

        if activity == start_token:
            ctx = (start_token,)
        else:
            ctx = (activity,)
        probs = self._probs[1].get(ctx, {})
        # Exclude end token for the "successors" list.
        end_token = self._fit_config.end_token if self._fit_config else "__END__"
        return [a for a in probs.keys() if a != end_token]

    def decision_points(self, min_successors: int = 2) -> Dict[str, List[str]]:
        """
        Identify (basic) XOR decision points from the learned directly-follows graph.

        Returns a mapping:
            preceding_activity -> list of observed successors
        """
        points: Dict[str, List[str]] = {}
        for ctx, probs in self._probs[1].items():
            if len(ctx) != 1:
                continue
            act = ctx[0]
            end_token = self._fit_config.end_token if self._fit_config else "__END__"
            succ = [a for a in probs.keys() if a != end_token]
            if len(succ) >= min_successors:
                points[act] = succ
        return points


# -------------------------------------------------------------------------
# self-test
# -------------------------------------------------------------------------

def _self_test() -> bool:
    df = pd.DataFrame(
        {
            "case:concept:name": ["1", "1", "1", "2", "2", "2", "3", "3", "3"],
            "concept:name": ["A", "X", "C", "A", "Y", "C", "A", "X", "C"],
            "time:timestamp": pd.date_range("2020-01-01", periods=9, freq="min"),
            "lifecycle:transition": ["complete"] * 9,
        }
    )
    model = BranchingBasic(max_context=1, seed=7).fit_from_dataframe(df)
    # After A, X should be twice as likely as Y (2 vs 1).
    dist = model.get_distribution(history=["A"], enabled_next=["X", "Y"])
    if not dist:
        return False
    return abs(dist["X"] - 2/3) < 1e-6 and abs(dist["Y"] - 1/3) < 1e-6
