"""Character n-gram TF-IDF + logistic regression (context-free word classifier)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..data import word_label_counts
from ..metrics import evaluate


def _flat(sents):
    return [w for s in sents for w in s["norm"]]


def _unflatten(flat, sents):
    out, i = [], 0
    for s in sents:
        out.append(list(flat[i:i + len(s["norm"])]))
        i += len(s["norm"])
    return out


class CharNgramTagger:
    name = "Char n-gram LogReg"

    def __init__(self, C: float = 10.0, ngram_range=(1, 5)):
        self.C, self.ngram_range = C, ngram_range
        self.vectorizer = self.clf = None
        self.search_log: list[tuple[float, float]] = []

    def _fit_one(self, X, y, C):
        return LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)

    def fit(self, train, dev=None, grid=(0.3, 1, 3, 10, 30)):
        """Fit; if `dev` is given, choose C on dev macro F1."""
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range, sublinear_tf=True)
        X = self.vectorizer.fit_transform(_flat(train))
        y = [l for s in train for l in s["labels"]]
        if dev is None:
            self.clf = self._fit_one(X, y, self.C)
            return self
        counts = word_label_counts(train)
        best = None
        for C in grid:
            clf = self._fit_one(X, y, C)
            self.clf = clf
            score = evaluate(dev, self.predict(dev), counts)["macro_F1"]
            self.search_log.append((C, score))
            if best is None or score > best[0]:
                best = (score, C, clf)
        _, self.C, self.clf = best
        return self

    def predict(self, sents):
        return _unflatten(self.clf.predict(self.vectorizer.transform(_flat(sents))), sents)

    def top_ngrams(self, k: int = 15) -> dict[str, list[str]]:
        """Strongest n-grams per class; '_' marks a word boundary."""
        feats = np.array(self.vectorizer.get_feature_names_out())
        coef = self.clf.coef_[0] if self.clf.classes_[1] == "TUN" else -self.clf.coef_[0]  # positive = TUN
        order = np.argsort(coef)
        show = lambda arr: [f.replace(" ", "_") for f in arr]
        return {"FOR": show(feats[order[:k]]), "TUN": show(feats[order[::-1][:k]])}

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "char_ngram_logreg.pkl").open("wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf, "C": self.C}, f)

    @classmethod
    def load(cls, directory: Path):
        with (directory / "char_ngram_logreg.pkl").open("rb") as f:
            d = pickle.load(f)
        m = cls(C=d["C"])
        m.vectorizer, m.clf = d["vectorizer"], d["clf"]
        return m

