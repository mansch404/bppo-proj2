
"""
Task 1.4 - Advanced (Next activities / XOR branching)
----------------------------------------------------

Assignment text (summary)
- Identify the data for decision points via token replay.
- Train a predictive model that takes (e.g.) the trace history into account.

This module implements an advanced branching predictor that:
1) Replays traces on a Petri net (token semantics) to identify *actual* decision points,
   meaning situations where multiple visible transitions are enabled at the same marking.
2) Collects supervised learning examples at those points:
      X = features derived from the already executed history
      y = chosen next activity (which enabled branch was actually taken)
3) Trains a light-weight probabilistic classifier per decision point.
   The default classifier is Multinomial Naive Bayes in log-space (no external ML deps).

Integration intent
- Your SimulationEngine.choose_transition(...) currently selects a random enabled transition. fileciteturn2file15
  To use this module, call BranchingAdvanced.choose_transition(...) instead and pass:
      - the current enabled transitions
      - the current marking (for decision-point keying)
      - the case's activity history (list[str])

Notes
- This module is designed to be "engine-friendly" but does not import the engine to avoid cycles.
- It expects pm4py PetriNet + Marking objects (same as your engine). fileciteturn2file16
- The EventLogger writes start/complete, but fitting defaults to complete-only, consistent
  with branching_basic. fileciteturn2file0
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

try:
    import pm4py
    from pm4py.objects.petri_net.obj import PetriNet, Marking
except Exception:  # pragma: no cover
    pm4py = None
    PetriNet = object  # type: ignore
    Marking = object  # type: ignore

# Types

Activity = str
Feature = str
DecisionKey = Tuple[Tuple[Tuple[str, int], ...], Tuple[str, ...]]
# DecisionKey = (marking_signature, enabled_labels_sorted)

# Config

@dataclass(frozen=True)
class FitConfig:
    """Configuration for reading and preparing event logs."""
    case_id_col: str = "case:concept:name"
    activity_col: str = "concept:name"
    timestamp_col: str = "time:timestamp"
    lifecycle_col: Optional[str] = "lifecycle:transition"
    keep_only_complete: bool = True

    # Replay safety controls
    max_tau_steps_per_event: int = 50
    max_total_steps_per_case: int = 5000

    # Feature config
    use_bag_of_activities: bool = True
    use_last_activity: bool = True
    use_last_bigram: bool = True
    last_bigram_sep: str = "|"

    # Naive Bayes smoothing
    alpha: float = 1.0



# Multinomial Naive Bayes (lightweight, no sklearn dependency)


class _MultinomialNB:
    """
    A minimal multinomial Naive Bayes classifier with log-probability outputs.

    Supports:
    - fit(X: list[dict[int,int]], y: list[int], n_features: int, n_classes: int)
    - predict_proba(x) -> list[float] over classes

    Uses Laplace smoothing controlled via alpha.
    """

    def __init__(self, *, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for MultinomialNB")
        self.alpha = float(alpha)
        self.class_log_prior: List[float] = []
        self.feature_log_prob: List[List[float]] = []
        self.n_features = 0
        self.n_classes = 0

    def fit(
        self,
        X: Sequence[Mapping[int, int]],
        y: Sequence[int],
        *,
        n_features: int,
        n_classes: int,
    ) -> "_MultinomialNB":
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)

        # Count docs per class
        class_counts = [0] * self.n_classes
        for cls in y:
            class_counts[int(cls)] += 1

        n_docs = len(y)
        self.class_log_prior = [
            math.log((c + self.alpha) / (n_docs + self.alpha * self.n_classes))
            for c in class_counts
        ]

        # Count features per class
        feat_counts = [[0.0] * self.n_features for _ in range(self.n_classes)]
        total_feat = [0.0] * self.n_classes

        for x, cls in zip(X, y):
            cls = int(cls)
            for j, v in x.items():
                if v <= 0:
                    continue
                feat_counts[cls][int(j)] += float(v)
                total_feat[cls] += float(v)

        # Likelihoods with Laplace smoothing
        self.feature_log_prob = []
        for cls in range(self.n_classes):
            denom = total_feat[cls] + self.alpha * self.n_features
            self.feature_log_prob.append(
                [math.log((feat_counts[cls][j] + self.alpha) / denom) for j in range(self.n_features)]
            )
        return self

    def predict_log_proba(self, x: Mapping[int, int]) -> List[float]:
        # log P(y) + sum_j x_j * log P(j|y)
        out = [lp for lp in self.class_log_prior]
        for cls in range(self.n_classes):
            s = out[cls]
            ll = self.feature_log_prob[cls]
            for j, v in x.items():
                if v <= 0:
                    continue
                s += float(v) * ll[int(j)]
            out[cls] = s
        return out

    @staticmethod
    def _log_softmax(logits: Sequence[float]) -> List[float]:
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        z = sum(exps)
        if z <= 0:
            # Should not happen; uniform fallback
            return [math.log(1.0 / len(logits))] * len(logits)
        return [math.log(e / z) for e in exps]

    def predict_proba(self, x: Mapping[int, int]) -> List[float]:
        logp = self.predict_log_proba(x)
        logp = self._log_softmax(logp)
        return [math.exp(v) for v in logp]


# BranchingAdvanced

class BranchingAdvanced:
    """
    Advanced XOR branching predictor based on token replay + supervised learning.

    High-level behavior
    - During fitting, traces are replayed on the Petri net to capture *true* enabled sets.
    - For each decision point key (marking signature + enabled label set), we train a
      classifier that maps trace history features -> branch choice probabilities.
    - During simulation, given enabled transitions and current marking, the predictor
      selects one enabled transition (or label) using the learned classifier.

    Compared to BranchingBasic, this is more aligned with the assignment's
    advanced requirement because it uses token replay to identify decision points. fileciteturn2file14
    """

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        config: Optional[FitConfig] = None,
    ) -> None:
        self.random = random.Random(seed)
        self.config = config or FitConfig()

        # Global feature vocabulary (string -> index)
        self._feat2idx: Dict[Feature, int] = {}
        self._idx2feat: List[Feature] = []

        # Decision point models:
        # key -> {"classes": [label], "model": _MultinomialNB}
        self._models: Dict[DecisionKey, Dict[str, object]] = {}

        # For convenience/fallbacks: unconditional counts per key
        self._empirical: Dict[DecisionKey, Dict[str, int]] = {}

    
    # Public API: fit
    

    def fit_from_event_log(
        self,
        log_path: str | Path,
        *,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Optional[Marking] = None,
        config: Optional[FitConfig] = None,
    ) -> "BranchingAdvanced":
        """
        Fit from a .xes or .csv event log at log_path.

        Parameters
        - net, initial_marking: Petri net used for token replay
        - final_marking: optional (used only for diagnostics; replay uses enabled transitions)
        """
        config = config or self.config
        df = self._read_event_log(Path(log_path), config)
        return self.fit_from_dataframe(
            df,
            net=net,
            initial_marking=initial_marking,
            final_marking=final_marking,
            config=config,
        )

    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Optional[Marking] = None,
        config: Optional[FitConfig] = None,
    ) -> "BranchingAdvanced":
        """
        Fit from an event log DataFrame using token replay on the provided Petri net.
        """
        config = config or self.config
        work = self._prepare_dataframe(df, config)

        # Build traces
        traces = self._build_traces(work, config)

        # Collect decision-point training examples
        examples: List[Tuple[DecisionKey, List[str], List[str], str]] = []
        for trace in traces:
            examples.extend(
                self._replay_and_extract_examples(
                    trace,
                    net=net,
                    initial_marking=initial_marking,
                    config=config,
                )
            )

        # Build feature vocab globally
        self._build_feature_vocab(examples, config=config)

        # Group by decision key and train a model per key
        grouped: Dict[DecisionKey, List[Tuple[Mapping[int, int], str, List[str]]]] = {}
        for key, history, enabled_labels, chosen in examples:
            x = self._vectorize(history, config=config)
            grouped.setdefault(key, []).append((x, chosen, enabled_labels))

        self._models = {}
        self._empirical = {}

        for key, rows in grouped.items():
            # Determine class order as the enabled label set from key
            enabled_sorted = list(key[1])
            cls2idx = {lbl: i for i, lbl in enumerate(enabled_sorted)}

            X = []
            y = []

            empirical_counts: Dict[str, int] = {lbl: 0 for lbl in enabled_sorted}

            for x, chosen, _enabled_labels in rows:
                if chosen not in cls2idx:
                    # This can occur if replay extracted enabled set differs
                    # from the key's enabled set. Skip defensively.
                    continueڍ
                X.append(x)
                y.append(cls2idx[chosen])
                empirical_counts[chosen] = empirical_counts.get(chosen, 0) + 1

            if len(X) < 5:
                # Too few samples: keep only empirical fallback, no NB model
                self._empirical[key] = empirical_counts
                continue

            nb = _MultinomialNB(alpha=config.alpha).fit(
                X, y, n_features=len(self._idx2feat), n_classes=len(enabled_sorted)
            )
            self._models[key] = {"classes": enabled_sorted, "model": nb}
            self._empirical[key] = empirical_counts

        return self

    
    # Public API: prediction for engine
    

    def choose_transition(
        self,
        *,
        enabled: Sequence[PetriNet.Transition],
        marking: Marking,
        history: Sequence[str],
        default_strategy: str = "uniform",
    ) -> PetriNet.Transition:
        """
        Choose which enabled transition to fire.

        - Filters enabled transitions to visible ones (label not empty).
        - If only one visible transition exists, returns it.
        - If multiple visible transitions exist, attempts to use the learned model
          for this decision point. Otherwise applies fallback.
        """
        enabled_visible = [t for t in enabled if (t.label is not None and str(t.label).strip() != "")]
        if not enabled_visible:
            # Only tau transitions enabled; keep engine behavior deterministic-ish
            return self.random.choice(list(enabled))

        if len(enabled_visible) == 1:
            return enabled_visible[0]

        enabled_labels = [str(t.label) for t in enabled_visible]
        key = (self._marking_signature(marking), tuple(sorted(set(enabled_labels))))

        # Predict label
        label = self.choose_next_label(
            history=history,
            enabled_labels=enabled_labels,
            decision_key=key,
            default_strategy=default_strategy,
        )

        # Map label back to a transition among enabled
        candidates = [t for t in enabled_visible if str(t.label) == label]
        if candidates:
            return self.random.choice(candidates)

        # Safety fallback: return any visible enabled transition
        return self.random.choice(enabled_visible)

    def choose_next_label(
        self,
        *,
        history: Sequence[str],
        enabled_labels: Sequence[str],
        decision_key: Optional[DecisionKey] = None,
        default_strategy: str = "uniform",
    ) -> str:
        """
        Predict the next activity label among enabled_labels.

        If decision_key is not provided, uses only enabled_labels (less precise).
        """
        if not enabled_labels:
            raise ValueError("enabled_labels must be non-empty")

        enabled_sorted = tuple(sorted(set(map(str, enabled_labels))))
        if decision_key is None:
            # Unknown marking: attempt any key with same enabled set, else fallback
            matching = [k for k in self._models.keys() if k[1] == enabled_sorted]
            if matching:
                decision_key = matching[0]
            else:
                return self._fallback_choice(list(enabled_labels), default_strategy)

        # If we have a trained model, use it
        model_entry = self._models.get(decision_key)
        if model_entry is not None:
            classes: List[str] = model_entry["classes"]  # type: ignore[assignment]
            nb: _MultinomialNB = model_entry["model"]  # type: ignore[assignment]

            x = self._vectorize(history, config=self.config)
            probs = nb.predict_proba(x)

            # Filter to actually enabled labels (in case of duplicates or mismatch)
            enabled_set = set(map(str, enabled_labels))
            filtered = [(cls, p) for cls, p in zip(classes, probs) if cls in enabled_set]
            if filtered:
                # Sample categorical
                return self._sample([(c, p) for c, p in filtered])

        # If no NB model, try empirical counts
        counts = self._empirical.get(decision_key)
        if counts:
            enabled_set = set(map(str, enabled_labels))
            filtered_counts = {k: v for k, v in counts.items() if k in enabled_set}
            if filtered_counts:
                total = sum(filtered_counts.values())
                if total > 0:
                    probs = [(k, v / total) for k, v in filtered_counts.items()]
                    return self._sample(probs)

        return self._fallback_choice(list(enabled_labels), default_strategy)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "feat2idx": self._feat2idx,
            "idx2feat": self._idx2feat,
            "models": self._models,
            "empirical": self._empirical,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, *, seed: Optional[int] = None) -> "BranchingAdvanced":
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        obj = cls(seed=seed, config=payload.get("config"))
        obj._feat2idx = payload.get("feat2idx", {})
        obj._idx2feat = payload.get("idx2feat", [])
        obj._models = payload.get("models", {})
        obj._empirical = payload.get("empirical", {})
        return obj

    
    # Replay and extraction
    

    @staticmethod
    def _read_event_log(path: Path, config: FitConfig) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".xes":
            if pm4py is None:
                raise ImportError("pm4py is required to read .xes logs")
            log = pm4py.read_xes(str(path))
            return pm4py.convert_to_dataframe(log)
        raise ValueError(f"Unsupported log type: {path.suffix}")

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame, config: FitConfig) -> pd.DataFrame:
        req = {config.case_id_col, config.activity_col, config.timestamp_col}
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise KeyError(f"Event log DataFrame missing required columns: {missing}")

        work = df.copy()
        work[config.timestamp_col] = pd.to_datetime(work[config.timestamp_col], utc=True, errors="coerce")
        work = work.dropna(subset=[config.timestamp_col])

        if config.keep_only_complete and config.lifecycle_col and config.lifecycle_col in work.columns:
            work = work[work[config.lifecycle_col].astype(str).str.lower() == "complete"]

        work = work.dropna(subset=[config.case_id_col, config.activity_col])
        work[config.case_id_col] = work[config.case_id_col].astype(str)
        work[config.activity_col] = work[config.activity_col].astype(str)

        work = work.sort_values([config.case_id_col, config.timestamp_col])
        return work[[config.case_id_col, config.activity_col, config.timestamp_col]]

    @staticmethod
    def _build_traces(df: pd.DataFrame, config: FitConfig) -> List[List[str]]:
        traces: List[List[str]] = []
        for _, g in df.groupby(config.case_id_col, sort=False):
            seq = g[config.activity_col].tolist()
            if seq:
                traces.append(seq)
        return traces

    def _replay_and_extract_examples(
        self,
        trace: Sequence[str],
        *,
        net: PetriNet,
        initial_marking: Marking,
        config: FitConfig,
    ) -> List[Tuple[DecisionKey, List[str], List[str], str]]:
        """
        Replay one trace on the Petri net.

        We use a practical replay:
        - Repeatedly fire enabled tau transitions (label None / empty) before each event.
        - Then ensure the next labeled event can be fired from enabled transitions.
        - If multiple visible transitions are enabled at that point, record a decision example.

        Returns:
            list of (decision_key, history_before, enabled_labels, chosen_label)
        """
        # Copy marking
        marking = Marking()
        for place, tokens in initial_marking.items():
            marking[place] = tokens

        history: List[str] = []
        examples: List[Tuple[DecisionKey, List[str], List[str], str]] = []

        total_steps = 0

        for evt in trace:
            total_steps += 1
            if total_steps > config.max_total_steps_per_case:
                break

            # Fire tau transitions greedily to reach stable visible frontier
            self._fire_tau_closure(net, marking, max_steps=config.max_tau_steps_per_event)

            enabled = self._get_enabled_transitions(net, marking)
            enabled_visible = [t for t in enabled if (t.label is not None and str(t.label).strip() != "")]

            # If there is a decision (XOR-like): multiple visible enabled
            if len(enabled_visible) >= 2:
                enabled_labels = sorted({str(t.label) for t in enabled_visible})
                chosen_label = str(evt)

                # Only create a training point if the chosen label is among enabled labels.
                if chosen_label in enabled_labels:
                    key: DecisionKey = (self._marking_signature(marking), tuple(enabled_labels))
                    examples.append((key, list(history), list(enabled_labels), chosen_label))

            # Now fire the transition that matches evt (may require tau closure again)
            fired = self._fire_matching_visible(net, marking, activity=str(evt), config=config)
            if not fired:
                # If replay fails, stop extracting for this case (avoid corrupting training)
                break

            history.append(str(evt))

        return examples

    
    # Petri net semantics helpers (mirrors your engine logic)
    

    @staticmethod
    def _marking_signature(marking: Marking) -> Tuple[Tuple[Tuple[str, int], ...], ...] | Tuple[Tuple[str, int], ...]:
        """
        Stable signature of a marking: tuple(sorted((place_name, tokens))).
        """
        items = []
        for place, tokens in marking.items():
            name = getattr(place, "name", str(place))
            items.append((str(name), int(tokens)))
        items.sort()
        return tuple(items)

    @staticmethod
    def _is_enabled(transition: PetriNet.Transition, marking: Marking) -> bool:
        for arc in transition.in_arcs:
            place = arc.source
            required = arc.weight if hasattr(arc, "weight") else 1
            if place not in marking or marking[place] < required:
                return False
        return True

    def _get_enabled_transitions(self, net: PetriNet, marking: Marking) -> List[PetriNet.Transition]:
        enabled = []
        for t in net.transitions:
            if self._is_enabled(t, marking):
                enabled.append(t)
        return enabled

    @staticmethod
    def _update_marking(transition: PetriNet.Transition, marking: Marking) -> None:
        # consume
        for arc in transition.in_arcs:
            place = arc.source
            w = arc.weight if hasattr(arc, "weight") else 1
            marking[place] -= w
            if marking[place] <= 0:
                del marking[place]
        # produce
        for arc in transition.out_arcs:
            place = arc.target
            w = arc.weight if hasattr(arc, "weight") else 1
            if place in marking:
                marking[place] += w
            else:
                marking[place] = w

    def _fire_tau_closure(self, net: PetriNet, marking: Marking, *, max_steps: int) -> None:
        steps = 0
        while steps < max_steps:
            enabled = self._get_enabled_transitions(net, marking)
            tau = [t for t in enabled if (t.label is None or str(t.label).strip() == "")]
            if not tau:
                return
            # deterministic choice to reduce variance in extracted datasets
            tau.sort(key=lambda t: getattr(t, "name", str(t)))
            t = tau[0]
            self._update_marking(t, marking)
            steps += 1

    def _fire_matching_visible(self, net: PetriNet, marking: Marking, *, activity: str, config: FitConfig) -> bool:
        """
        Attempt to fire an enabled visible transition with label == activity.

        Strategy:
        - Apply tau closure
        - If a matching visible transition is enabled, fire it
        - Else: try limited additional tau steps, then retry
        """
        # First closure
        self._fire_tau_closure(net, marking, max_steps=config.max_tau_steps_per_event)

        for _ in range(3):
            enabled = self._get_enabled_transitions(net, marking)
            enabled_visible = [t for t in enabled if (t.label is not None and str(t.label).strip() != "")]
            candidates = [t for t in enabled_visible if str(t.label) == activity]
            if candidates:
                # deterministic-ish: prefer name order, but keep stability
                candidates.sort(key=lambda t: getattr(t, "name", str(t)))
                self._update_marking(candidates[0], marking)
                return True

            # try firing one tau transition and retry
            tau = [t for t in enabled if (t.label is None or str(t.label).strip() == "")]
            if not tau:
                return False
            tau.sort(key=lambda t: getattr(t, "name", str(t)))
            self._update_marking(tau[0], marking)

        return False

    
    # Feature engineering
    

    def _build_feature_vocab(
        self,
        examples: Sequence[Tuple[DecisionKey, List[str], List[str], str]],
        *,
        config: FitConfig,
    ) -> None:
        feats = set()

        for _key, history, _enabled, _chosen in examples:
            for f in self._extract_features(history, config=config).keys():
                feats.add(f)

        self._idx2feat = sorted(feats)
        self._feat2idx = {f: i for i, f in enumerate(self._idx2feat)}

    def _extract_features(self, history: Sequence[str], *, config: FitConfig) -> Dict[Feature, int]:
        """
        Turn a trace history into a feature multiset.

        Features:
        - Bag-of-activities counts: "ACT=<activity>"
        - Last activity: "LAST=<activity>"
        - Last bigram: "BIGRAM=<a>|<b>"
        """
        out: Dict[Feature, int] = {}

        if config.use_bag_of_activities:
            for a in history:
                f = f"ACT={a}"
                out[f] = out.get(f, 0) + 1

        if config.use_last_activity and len(history) >= 1:
            f = f"LAST={history[-1]}"
            out[f] = out.get(f, 0) + 1

        if config.use_last_bigram and len(history) >= 2:
            a, b = history[-2], history[-1]
            f = f"BIGRAM={a}{config.last_bigram_sep}{b}"
            out[f] = out.get(f, 0) + 1

        return out

    def _vectorize(self, history: Sequence[str], *, config: FitConfig) -> Dict[int, int]:
        feats = self._extract_features(history, config=config)
        x: Dict[int, int] = {}
        for f, v in feats.items():
            idx = self._feat2idx.get(f)
            if idx is None:
                continue
            x[int(idx)] = int(v)
        return x

    
    # Sampling / fallbacks
    

    def _sample(self, probs: Sequence[Tuple[str, float]]) -> str:
        # normalize defensively
        s = sum(p for _, p in probs)
        if s <= 0:
            return probs[0][0]
        r = self.random.random()
        cum = 0.0
        last = probs[0][0]
        for k, p in probs:
            last = k
            cum += p / s
            if r <= cum:
                return k
        return last

    def _fallback_choice(self, enabled: List[str], strategy: str) -> str:
        strategy = strategy.lower().strip()
        if strategy == "uniform":
            return self.random.choice(enabled)
        if strategy == "first":
            return enabled[0]
        if strategy == "error":
            raise RuntimeError("No advanced branching model available for this decision point.")
        raise ValueError(f"Unknown default_strategy: {strategy}")



# Quick diagnostic (optional)


def _smoke_test() -> bool:
    # Only checks importability and basic NB behavior
    nb = _MultinomialNB(alpha=1.0).fit(
        X=[{0: 2, 1: 1}, {0: 0, 1: 3}, {0: 1, 1: 1}],
        y=[0, 1, 0],
        n_features=2,
        n_classes=2,
    )
    p = nb.predict_proba({0: 1, 1: 0})
    return len(p) == 2 and abs(sum(p) - 1.0) < 1e-6
