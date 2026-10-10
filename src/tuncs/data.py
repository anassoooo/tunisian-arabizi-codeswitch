"""Loading, cleaning, normalising and splitting the TArC corpus.

A sentence is a dict: {"id", "genre", "tokens", "labels", "norm"}.
"""

from __future__ import annotations

import csv
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .config import PROCESSED, RAW_TSV, SEED, TARC_URL

LETTER = re.compile(r"[^\W\d_]")
PLACEHOLDER = re.compile(r"^m5abbi\d+$", re.I)  # anonymised user names inserted by the corpus authors


def download_tarc(path: Path = RAW_TSV) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TARC_URL, path)
    return path


def read_raw(path: Path = RAW_TSV) -> tuple[list[str], list[list[str]], list[dict]]:
    """Return the header, the raw token rows, and raw sentences (split on `<eos>`)."""
    with Path(path).open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        rows = list(reader)
    sents, cur = [], []
    for r in rows + [["", "", "<eos>"]]:
        if r[2] == "<eos>":
            if cur:
                sents.append({"genre": cur[0][0], "tokens": [x[2] for x in cur], "classes": [x[3] for x in cur]})
            cur = []
        else:
            cur.append(r)
    return header, rows, sents


def normalize(token: str) -> str:
    """Lower-case and squeeze letters repeated 3+ times (loooool -> lool). Digits and accents are kept."""
    return re.sub(r"(.)\1{2,}", r"\1\1", token.lower())


def clean_sentence(tokens: list[str], classes: list[str]) -> tuple[list[str], list[str], Counter]:
    """Keep lexical tokens only and map TArC classes to TUN / FOR."""
    out_t, out_l, log = [], [], Counter()
    for tok, cls in zip(tokens, classes):
        tok = tok.replace("\\'", "'")
        if cls == "nan":  # a single unlabelled token ("yarhmek"), clearly Tunisian
            cls = "arabizi"
            log["nan relabelled TUN"] += 1
        if cls == "emotag":
            log["emotag"] += 1
            continue
        if PLACEHOLDER.match(tok):
            log["user placeholder"] += 1
            continue
        if not LETTER.search(tok):
            log["no letter (punct/number)"] += 1
            continue
        out_t.append(tok)
        out_l.append("TUN" if cls == "arabizi" else "FOR")
    return out_t, out_l, log


def make_sentence(tokens: list[str], labels: list[str] | None = None, **meta) -> dict:
    labels = labels if labels is not None else ["TUN"] * len(tokens)
    return {**meta, "tokens": tokens, "labels": labels, "norm": [normalize(t) for t in tokens]}


def tokenize_text(text: str) -> list[str]:
    """Whitespace tokenisation that drops tokens without letters (same policy as the corpus cleaning)."""
    return [t for t in text.split() if LETTER.search(t)]


def load_corpus(path: Path = RAW_TSV) -> tuple[list[dict], Counter]:
    raw = read_raw(download_tarc(path))[2]
    sents, log = [], Counter()
    for i, s in enumerate(raw):
        toks, labs, d = clean_sentence(s["tokens"], s["classes"])
        log.update(d)
        if toks:
            sents.append(make_sentence(toks, labs, id=f"{s['genre']}_{i:04d}", genre=s["genre"]))
    return sents, log


def deduplicate(sents: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in sents:
        key = tuple(s["norm"])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def split(sents: list[dict], seed: int = SEED, ratios=(0.8, 0.1)) -> tuple[list[dict], list[dict], list[dict]]:
    order = list(range(len(sents)))
    random.Random(seed).shuffle(order)
    n_tr, n_dev = int(ratios[0] * len(order)), int(ratios[1] * len(order))
    pick = lambda idx: [sents[i] for i in idx]
    return pick(order[:n_tr]), pick(order[n_tr:n_tr + n_dev]), pick(order[n_tr + n_dev:])


def word_label_counts(sents: list[dict]) -> dict[str, Counter]:
    counts = defaultdict(Counter)
    for s in sents:
        for w, l in zip(s["norm"], s["labels"]):
            counts[w][l] += 1
    return counts


def is_code_switched(labels: list[str]) -> bool:
    return len(set(labels)) == 2


def save_jsonl(sents: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:  # same bytes on every OS
        for s in sents:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prepare_splits(out_dir: Path = PROCESSED, seed: int = SEED) -> dict:
    """Clean, de-duplicate and split the corpus; write train/dev/test JSONL files. Returns a summary."""
    sents, log = load_corpus()
    unique = deduplicate(sents)
    train, dev, test = split(unique, seed)
    for name, data in (("train", train), ("dev", dev), ("test", test)):
        save_jsonl(data, out_dir / f"{name}.jsonl")
    summary = {"cleaning_log": dict(log), "sentences_after_cleaning": len(sents), "duplicates_removed": len(sents) - len(unique),
               "train": len(train), "dev": len(dev), "test": len(test), "seed": seed}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return summary


def load_splits(out_dir: Path = PROCESSED) -> tuple[list[dict], list[dict], list[dict]]:
    if not (out_dir / "train.jsonl").exists():
        prepare_splits(out_dir)
    return tuple(load_jsonl(out_dir / f"{n}.jsonl") for n in ("train", "dev", "test"))
