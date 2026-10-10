"""Helpers shared by the notebook and the command-line scripts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import COLORS, FIGURES, MODELS, PREDICTIONS, ROOT, TABLES
from .data import word_label_counts
from .metrics import bootstrap_for_f1, evaluate

MODEL_DIRS = {"LexiconTagger": "lexicon", "CharNgramTagger": "char_ngram_logreg", "CRFTagger": "crf",
              "BiLSTMCRFTagger": "bilstm_crf", "XLMRTagger": "xlmr"}
TABLE_COLUMNS = ["FOR_F1", "FOR_F1_95CI", "macro_F1", "unseen_macro_F1", "ambiguous_macro_F1",
                 "cs_sentence_F1", "switch_point_F1", "accuracy", "dev_macro_F1"]


def model_dir(cls) -> Path:
    return MODELS / MODEL_DIRS[cls.__name__]


def train_or_load(cls, train, dev, retrain: bool = True, **kwargs):
    """Train and save a model, or load the saved one when `retrain` is False and it exists."""
    directory = model_dir(cls)
    marker = directory / "DONE"
    if not retrain and marker.exists():
        print(f"loading saved {cls.__name__} from {directory.relative_to(ROOT).as_posix()}/")
        return cls.load(directory)
    model = cls(**kwargs).fit(train, dev)
    model.save(directory)
    marker.write_text("trained\n")
    return model


def score(model, train, dev, test) -> tuple[dict, list[list[str]]]:
    counts = word_label_counts(train)
    test_pred = model.predict(test)
    row = evaluate(test, test_pred, counts)
    lo, hi = bootstrap_for_f1(test, test_pred)
    row["FOR_F1_95CI"] = f"[{lo:.3f}, {hi:.3f}]"
    row["dev_macro_F1"] = evaluate(dev, model.predict(dev), counts)["macro_F1"]
    return row, test_pred


def results_table(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).T[TABLE_COLUMNS]


def save_results(table: pd.DataFrame, preds: dict[str, list[list[str]]], test: list[dict]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "results_test.csv")
    for name, pred in preds.items():
        slug = name.split(" (")[0].lower().replace(" ", "_").replace("-", "_")
        with (PREDICTIONS / f"test_{slug}.jsonl").open("w", encoding="utf-8") as f:
            for s, p in zip(test, pred):
                f.write(json.dumps({"id": s["id"], "tokens": s["tokens"], "gold": s["labels"], "pred": p},
                                   ensure_ascii=False) + "\n")


def plot_results(table: pd.DataFrame, path: Path | None = FIGURES / "results_test.png"):
    metrics = [("FOR_F1", "Foreign-word F1"), ("unseen_macro_F1", "Unseen words (macro F1)"),
               ("ambiguous_macro_F1", "Ambiguous words (macro F1)"), ("switch_point_F1", "Switch-point F1")]
    names = list(table.index)[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        vals = table.loc[names, m].astype(float)
        ax.barh(names, vals, color=COLORS["TUN"], height=0.6)
        for y, v in enumerate(vals):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def tag_string(tokens: list[str], labels: list[str]) -> str:
    """Render a tagged sentence with foreign words in [brackets]."""
    return " ".join(f"[{t}]" if l == "FOR" else t for t, l in zip(tokens, labels))
