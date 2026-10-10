"""BiLSTM-CRF tagger: word embedding + character CNN -> BiLSTM -> CRF (Lample et al., 2016)."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..config import LABELS, SEED
from ..data import word_label_counts
from ..metrics import evaluate

L2I = {l: i for i, l in enumerate(LABELS)}


class CRF(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.start, self.end = nn.Parameter(torch.zeros(k)), nn.Parameter(torch.zeros(k))
        self.trans = nn.Parameter(torch.zeros(k, k))

    def nll(self, em, tags, mask):
        B, T, _ = em.shape
        ar = torch.arange(B, device=em.device)
        gold = self.start[tags[:, 0]] + em[ar, 0, tags[:, 0]]
        alpha = self.start + em[:, 0]
        for t in range(1, T):
            gold = gold + torch.where(mask[:, t], self.trans[tags[:, t - 1], tags[:, t]] + em[ar, t, tags[:, t]], 0.0)
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        last = tags[ar, mask.sum(1) - 1]
        return (torch.logsumexp(alpha + self.end, 1) - gold - self.end[last]).mean()

    @torch.no_grad()
    def decode(self, em, mask):
        score, back = self.start + em[:, 0], []
        for t in range(1, em.shape[1]):
            best, arg = (score.unsqueeze(2) + self.trans).max(1)
            score = torch.where(mask[:, t].unsqueeze(1), best + em[:, t], score)
            back.append(arg)
        score = score + self.end
        out = []
        for i, n in enumerate(mask.sum(1).tolist()):
            path = [int(score[i].argmax())]
            for t in range(n - 2, -1, -1):
                path.append(int(back[t][i, path[-1]]))
            out.append(path[::-1])
        return out


class BiLSTMCRF(nn.Module):
    def __init__(self, n_words, n_chars, word_dim=100, char_dim=30, char_filters=64, hidden=128, dropout=0.4):
        super().__init__()
        self.wemb = nn.Embedding(n_words, word_dim, padding_idx=0)
        self.cemb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_dim, char_filters, 3, padding=1)
        self.lstm = nn.LSTM(word_dim + char_filters, hidden, batch_first=True, bidirectional=True)
        self.drop, self.out, self.crf = nn.Dropout(dropout), nn.Linear(2 * hidden, len(LABELS)), CRF(len(LABELS))

    def emissions(self, W, CH, M):
        B, T, C = CH.shape
        c = torch.relu(self.conv(self.cemb(CH.view(B * T, C)).transpose(1, 2))).amax(2).view(B, T, -1)
        x = self.drop(torch.cat([self.wemb(W), c], 2))
        packed = pack_padded_sequence(x, M.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=T)
        return self.out(self.drop(h))


class BiLSTMCRFTagger:
    name = "BiLSTM-CRF (word + char CNN)"

    def __init__(self, epochs=30, patience=5, batch_size=32, lr=2e-3, word_dropout=0.1, seed=SEED, device=None):
        self.epochs, self.patience, self.batch_size, self.lr = epochs, patience, batch_size, lr
        self.word_dropout, self.seed = word_dropout, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.word_vocab = self.char_vocab = self.model = None
        self.history: list[dict] = []

    def _batches(self, data, shuffle=False, word_dropout=0.0, size=None):
        idx = list(range(len(data)))
        if shuffle:
            random.shuffle(idx)
        size = size or self.batch_size
        for b in range(0, len(idx), size):
            chunk = [data[i] for i in idx[b:b + size]]
            T = max(len(s["norm"]) for s in chunk)
            C = min(24, max(len(w) for s in chunk for w in s["norm"]))
            W = torch.zeros(len(chunk), T, dtype=torch.long)
            CH = torch.zeros(len(chunk), T, C, dtype=torch.long)
            Y = torch.zeros(len(chunk), T, dtype=torch.long)
            M = torch.zeros(len(chunk), T, dtype=torch.bool)
            for i, s in enumerate(chunk):
                for j, (w, l) in enumerate(zip(s["norm"], s["labels"])):
                    W[i, j] = 1 if word_dropout and random.random() < word_dropout else self.word_vocab.get(w, 1)
                    Y[i, j], M[i, j] = L2I[l], True
                    for k, ch in enumerate(w[:C]):
                        CH[i, j, k] = self.char_vocab.get(ch, 1)
            yield W.to(self.device), CH.to(self.device), Y.to(self.device), M.to(self.device)

    def fit(self, train, dev, verbose=True):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.word_vocab = {"<pad>": 0, "<unk>": 1}
        for w in sorted({w for s in train for w in s["norm"]}):
            self.word_vocab[w] = len(self.word_vocab)
        self.char_vocab = {"<pad>": 0, "<unk>": 1}
        for ch in sorted({ch for s in train for w in s["norm"] for ch in w}):
            self.char_vocab[ch] = len(self.char_vocab)
        self.model = BiLSTMCRF(len(self.word_vocab), len(self.char_vocab)).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        counts = word_label_counts(train)
        best_f, best_state, wait, t0 = -1.0, None, 0, time.time()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total = 0.0
            for W, CH, Y, M in self._batches(train, shuffle=True, word_dropout=self.word_dropout):
                loss = self.model.crf.nll(self.model.emissions(W, CH, M), Y, M)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                total += loss.item()
            f = evaluate(dev, self.predict(dev), counts)["macro_F1"]
            improved = f > best_f
            if improved:
                best_f, wait = f, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
            self.history.append({"epoch": epoch, "loss": total, "dev_macro_F1": f})
            if verbose:
                print(f"epoch {epoch:2d}  loss {total:8.1f}  dev macro F1 {f:.4f}{'  *' if improved else ''}")
            if wait == self.patience:
                break
        self.model.load_state_dict(best_state)
        if verbose:
            print(f"best dev macro F1 {best_f:.4f}  ({time.time() - t0:.0f}s)")
        return self

    @torch.no_grad()
    def predict(self, sents):
        self.model.eval()
        out = []
        for W, CH, Y, M in self._batches(sents, size=64):
            out += [[LABELS[i] for i in p] for p in self.model.crf.decode(self.model.emissions(W, CH, M), M)]
        return out

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "bilstm_crf.pt")
        (directory / "vocab.json").write_text(json.dumps({"words": self.word_vocab, "chars": self.char_vocab},
                                                         ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, device=None):
        m = cls(device=device)
        v = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        m.word_vocab, m.char_vocab = v["words"], v["chars"]
        m.model = BiLSTMCRF(len(m.word_vocab), len(m.char_vocab)).to(m.device)
        m.model.load_state_dict(torch.load(directory / "bilstm_crf.pt", map_location=m.device))
        return m
