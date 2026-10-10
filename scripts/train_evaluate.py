"""Train every model, evaluate it on the test split and write results/ and models/.

    python scripts/train_evaluate.py              # all models (XLM-R needs a GPU to be practical)
    python scripts/train_evaluate.py --skip-xlmr  # classical + BiLSTM-CRF only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tuncs.data import load_splits  # noqa: E402
from tuncs.experiment import plot_results, results_table, save_results, score, train_or_load  # noqa: E402
from tuncs.models import CharNgramTagger, CRFTagger, LexiconTagger, MajorityTagger  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-xlmr", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="load already-saved models instead of retraining")
    args = ap.parse_args()

    train, dev, test = load_splits()
    print(f"train {len(train)} | dev {len(dev)} | test {len(test)} sentences")
    models = [MajorityTagger().fit(train)]
    for cls in (LexiconTagger, CharNgramTagger, CRFTagger):
        models.append(train_or_load(cls, train, dev, retrain=not args.reuse))
    from tuncs.models.bilstm_crf import BiLSTMCRFTagger
    models.append(train_or_load(BiLSTMCRFTagger, train, dev, retrain=not args.reuse))
    if not args.skip_xlmr:
        from tuncs.models.xlmr import XLMRTagger
        models.append(train_or_load(XLMRTagger, train, dev, retrain=not args.reuse))

    rows, preds = {}, {}
    for m in models:
        rows[m.name], preds[m.name] = score(m, train, dev, test)
    table = results_table(rows)
    save_results(table, preds, test)
    plot_results(table)
    print(table.round(3).to_string())
