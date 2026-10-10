"""Fine-tuned XLM-RoBERTa token classifier (first sub-word of each word is trained and scored)."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from ..config import LABELS, SEED
from ..data import word_label_counts
from ..metrics import evaluate

L2I = {l: i for i, l in enumerate(LABELS)}


class XLMRTagger:
    name = "XLM-R base (fine-tuned)"

    def __init__(self, model_name="xlm-roberta-base", epochs=5, batch_size=16, lr=3e-5, chunk=64, seed=SEED, device=None):
        self.model_name, self.epochs, self.batch_size, self.lr, self.chunk, self.seed = model_name, epochs, batch_size, lr, chunk, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = self.device.type == "cuda"
        self.tokenizer = self.model = None
        self.history: list[dict] = []

    # Long sentences are cut into chunks of `chunk` words and reassembled, so every word gets a prediction.
    def _chunks(self, data):
        return [(si, s["tokens"][b:b + self.chunk], s["labels"][b:b + self.chunk])
                for si, s in enumerate(data) for b in range(0, len(s["tokens"]), self.chunk)]

    def _encode(self, items):
        enc = self.tokenizer([w for _, w, _ in items], is_split_into_words=True, truncation=True, max_length=256,
                             padding=True, return_tensors="pt")
        labels = torch.full(enc["input_ids"].shape, -100)
        for b, (_, _, labs) in enumerate(items):
            prev = None
            for j, wid in enumerate(enc.word_ids(b)):
                if wid is not None and wid != prev:
                    labels[b, j] = L2I[labs[wid]]
                prev = wid
        return enc, labels

    def fit(self, train, dev, verbose=True):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I).to(self.device)
        items = self._chunks(train)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * math.ceil(len(items) / self.batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        counts, best_f, best_state, t0 = word_label_counts(train), -1.0, None, time.time()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            random.shuffle(items)
            total = 0.0
            for b in range(0, len(items), self.batch_size):
                enc, labels = self._encode(items[b:b + self.batch_size])
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                    loss = self.model(**{k: v.to(self.device) for k, v in enc.items()}, labels=labels.to(self.device)).loss
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item()
            f = evaluate(dev, self.predict(dev), counts)["macro_F1"]
            improved = f > best_f
            if improved:
                best_f = f
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.history.append({"epoch": epoch, "loss": total, "dev_macro_F1": f})
            if verbose:
                print(f"epoch {epoch}  loss {total:7.2f}  dev macro F1 {f:.4f}{'  *' if improved else ''}  ({time.time() - t0:.0f}s)")
        self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, sents, batch_size=32):
        self.model.eval()
        items = self._chunks(sents)
        preds = [[] for _ in sents]
        for b in range(0, len(items), batch_size):
            part = items[b:b + batch_size]
            enc, _ = self._encode(part)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                ids = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.argmax(-1).cpu()
            for i, (si, words, _) in enumerate(part):
                first = {}
                for j, wid in enumerate(enc.word_ids(i)):
                    if wid is not None and wid not in first:
                        first[wid] = j
                preds[si] += [LABELS[int(ids[i, first[k]])] if k in first else "TUN" for k in range(len(words))]
        return preds

    @torch.no_grad()
    def predict_proba(self, sents) -> list[list[float]]:
        """Probability that each word is foreign (used by the demo app)."""
        self.model.eval()
        out = []
        for s in sents:
            enc, _ = self._encode([(0, s["tokens"], s["labels"])])
            probs = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.float().softmax(-1)[0].cpu()
            first = {}
            for j, wid in enumerate(enc.word_ids(0)):
                if wid is not None and wid not in first:
                    first[wid] = j
            out.append([float(probs[first[k], L2I["FOR"]]) if k in first else 0.0 for k in range(len(s["tokens"]))])
        return out

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

    @classmethod
    def load(cls, directory: Path, device=None):
        m = cls(model_name=str(directory), device=device)
        # transformers 4.57 wrongly warns about a "Mistral regex" for saved XLM-R tokenizers; tokenization is
        # identical to xlm-roberta-base, so silence it while loading.
        level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            m.tokenizer = AutoTokenizer.from_pretrained(directory)
        finally:
            hf_logging.set_verbosity(level)
        m.model = AutoModelForTokenClassification.from_pretrained(directory).to(m.device)
        return m
