"""Reference baselines: majority class and training-word lookup."""

from __future__ import annotations

import json
from pathlib import Path

from ..data import word_label_counts


class MajorityTagger:
    name = "Majority (all TUN)"

    def fit(self, train, dev=None):
        return self

    def predict(self, sents):
        return [["TUN"] * len(s["norm"]) for s in sents]


class LexiconTagger:
    """Each word gets its most frequent training label; unseen words get TUN (the majority class)."""

    name = "Lexicon lookup"

    def __init__(self):
        self.lexicon: dict[str, str] = {}

    def fit(self, train, dev=None):
        self.lexicon = {w: c.most_common(1)[0][0] for w, c in word_label_counts(train).items()}
        return self

    def predict(self, sents):
        return [[self.lexicon.get(w, "TUN") for w in s["norm"]] for s in sents]

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "lexicon.json").write_text(json.dumps(self.lexicon, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path):
        m = cls()
        m.lexicon = json.loads((directory / "lexicon.json").read_text(encoding="utf-8"))
        return m
