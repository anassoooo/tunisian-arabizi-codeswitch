"""Linear-chain CRF with hand-crafted word features and a +/-2 word context window."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import sklearn_crfsuite

from ..data import normalize, word_label_counts
from ..metrics import evaluate

ACCENT = re.compile(r"[àâçéèêëîïôûùüÿœæ]")
ARITHMOGRAPH = re.compile(r"[235789]")  # digits used as letters: 3=ع 5=خ 7=ح 9=ق 8=ه 2=ء


def word_features(token: str) -> dict:
    w = normalize(token)
    return {"w": w, "p2": w[:2], "p3": w[:3], "s1": w[-1:], "s2": w[-2:], "s3": w[-3:], "s4": w[-4:],
            "arith": bool(ARITHMOGRAPH.search(w)), "accent": bool(ACCENT.search(w)), "apos": "'" in w,
            "upper": token.isupper(), "title": token.istitle(), "len": str(min(len(w), 8))}


def sentence_features(tokens: list[str]) -> list[dict]:
    base = [word_features(t) for t in tokens]
    feats = []
    for i, f in enumerate(base):
        d = {"bias": 1.0, **f}
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < len(base):
                for k in ("w", "s3", "arith", "accent"):
                    d[f"{off}:{k}"] = base[j][k]
            else:
                d[f"{off}:pad"] = True
        feats.append(d)
    return feats


class CRFTagger:
    name = "CRF (features + context)"

    def __init__(self, c1: float = 0.01, c2: float = 0.01):
        self.c1, self.c2, self.crf = c1, c2, None
        self.search_log: list[tuple[float, float, float]] = []

    def _fit_one(self, X, y, c1, c2):
        crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=200, all_possible_transitions=True)
        crf.fit(X, y)
        return crf

    def fit(self, train, dev=None, grid=((0.01, 0.01), (0.01, 0.1), (0.01, 1.0), (0.1, 0.01), (0.1, 0.1), (0.1, 1.0))):
        """Fit; if `dev` is given, choose (c1, c2) on dev macro F1."""
        X = [sentence_features(s["tokens"]) for s in train]
        y = [s["labels"] for s in train]
        if dev is None:
            self.crf = self._fit_one(X, y, self.c1, self.c2)
            return self
        counts, best = word_label_counts(train), None
        for c1, c2 in grid:
            self.crf = self._fit_one(X, y, c1, c2)
            score = evaluate(dev, self.predict(dev), counts)["macro_F1"]
            self.search_log.append((c1, c2, score))
            if best is None or score > best[0]:
                best = (score, c1, c2, self.crf)
        _, self.c1, self.c2, self.crf = best
        return self

    def predict(self, sents):
        return self.crf.predict([sentence_features(s["tokens"]) for s in sents])

    def transitions(self) -> dict[str, float]:
        return {f"{a}->{b}": round(w, 2) for (a, b), w in self.crf.transition_features_.items()}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "crf.pkl").open("wb") as f:
            pickle.dump({"crf": self.crf, "c1": self.c1, "c2": self.c2}, f)

    @classmethod
    def load(cls, directory: Path):
        with (directory / "crf.pkl").open("rb") as f:
            d = pickle.load(f)
        m = cls(d["c1"], d["c2"])
        m.crf = d["crf"]
        return m
