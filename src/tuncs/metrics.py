"""Token-, sentence- and switch-level evaluation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .config import LABELS, SEED
from .data import is_code_switched


def f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def token_counts(gold: list[str], pred: list[str]) -> Counter:
    c = Counter()
    for g, p in zip(gold, pred):
        for lab in LABELS:
            c[lab, "tp"] += g == lab and p == lab
            c[lab, "fp"] += g != lab and p == lab
            c[lab, "fn"] += g == lab and p != lab
    return c


def macro_f1(c: Counter) -> float:
    return float(np.mean([f1(c[l, "tp"], c[l, "fp"], c[l, "fn"]) for l in LABELS]))


def switch_points(labels: list[str]) -> set[int]:
    return {i for i in range(1, len(labels)) if labels[i] != labels[i - 1]}


def evaluate(data: list[dict], preds: list[list[str]], train_counts: dict) -> dict:
    """All metrics. `train_counts` (word -> Counter of labels) defines unseen and ambiguous words."""
    gold = [l for s in data for l in s["labels"]]
    pred = [l for p in preds for l in p]
    words = [w for s in data for w in s["norm"]]
    assert len(gold) == len(pred), "one prediction per word is required"
    c = token_counts(gold, pred)
    res = {"accuracy": float(np.mean([g == p for g, p in zip(gold, pred)])),
           "FOR_F1": f1(c["FOR", "tp"], c["FOR", "fp"], c["FOR", "fn"]),
           "TUN_F1": f1(c["TUN", "tp"], c["TUN", "fp"], c["TUN", "fn"]),
           "macro_F1": macro_f1(c)}
    subsets = {"unseen": lambda w: w not in train_counts, "ambiguous": lambda w: len(train_counts.get(w, ())) == 2}
    for name, keep in subsets.items():
        idx = [i for i, w in enumerate(words) if keep(w)]
        res[f"{name}_macro_F1"] = macro_f1(token_counts([gold[i] for i in idx], [pred[i] for i in idx]))
    gs = [is_code_switched(s["labels"]) for s in data]
    ps = [is_code_switched(p) for p in preds]
    res["cs_sentence_F1"] = f1(sum(g and p for g, p in zip(gs, ps)), sum(p and not g for g, p in zip(gs, ps)),
                               sum(g and not p for g, p in zip(gs, ps)))
    tp = fp = fn = 0
    for s, p in zip(data, preds):
        g_sw, p_sw = switch_points(s["labels"]), switch_points(p)
        tp, fp, fn = tp + len(g_sw & p_sw), fp + len(p_sw - g_sw), fn + len(g_sw - p_sw)
    res["switch_point_F1"] = f1(tp, fp, fn)
    return res


def bootstrap_for_f1(data: list[dict], preds: list[list[str]], n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """95 % interval for foreign-word F1, resampling test sentences."""
    per = [token_counts(s["labels"], p) for s, p in zip(data, preds)]
    arr = np.array([[c["FOR", "tp"], c["FOR", "fp"], c["FOR", "fn"]] for c in per])
    rng = np.random.default_rng(seed)
    scores = [f1(*arr[rng.integers(0, len(arr), len(arr))].sum(0)) for _ in range(n_boot)]
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)
