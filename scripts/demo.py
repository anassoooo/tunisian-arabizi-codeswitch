"""Tag your own sentences with the saved models (run train_evaluate.py or the notebook first).

    python scripts/demo.py "la la ma n7ebch nemchi la fac lyoum"
    python scripts/demo.py            # interactive mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tuncs.data import make_sentence, tokenize_text  # noqa: E402
from tuncs.experiment import model_dir, tag_string  # noqa: E402
from tuncs.models import CharNgramTagger, CRFTagger, LexiconTagger  # noqa: E402


def load_models():
    classes = [LexiconTagger, CharNgramTagger, CRFTagger]
    try:
        from tuncs.models.bilstm_crf import BiLSTMCRFTagger
        from tuncs.models.xlmr import XLMRTagger
        classes += [BiLSTMCRFTagger, XLMRTagger]
    except ImportError:
        pass
    return {cls.name: cls.load(model_dir(cls)) for cls in classes if (model_dir(cls) / "DONE").exists()}


def show(models, text):
    s = make_sentence(tokenize_text(text))
    print(f"\n{text}")
    for name, m in models.items():
        print(f"   {name:30} {tag_string(s['tokens'], m.predict([s])[0])}")


if __name__ == "__main__":
    models = load_models()
    if not models:
        sys.exit("No saved models found in models/. Run scripts/train_evaluate.py first.")
    if len(sys.argv) > 1:
        for text in sys.argv[1:]:
            show(models, text)
    else:
        print("Type a sentence (empty line to quit). Foreign words are shown in [brackets].")
        while (text := input("> ").strip()):
            show(models, text)
