"""Quick checks of cleaning, normalisation and metrics:  python -m unittest discover -s tests"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tuncs.data import clean_sentence, make_sentence, normalize, tokenize_text  # noqa: E402
from tuncs.metrics import evaluate, switch_points  # noqa: E402


class TestData(unittest.TestCase):
    def test_normalize_keeps_digits_and_accents(self):
        self.assertEqual(normalize("Loooool"), "lool")
        self.assertEqual(normalize("3AYCHEK"), "3aychek")
        self.assertEqual(normalize("Été"), "été")

    def test_clean_sentence_policy(self):
        toks, labs, log = clean_sentence(["ya", "m5abbi12", "😂", "?", "mais", "j\\'ai", "yarhmek"],
                                         ["arabizi", "arabizi", "emotag", "arabizi", "foreign", "foreign", "nan"])
        self.assertEqual(toks, ["ya", "mais", "j'ai", "yarhmek"])
        self.assertEqual(labs, ["TUN", "FOR", "FOR", "TUN"])
        self.assertEqual(log["emotag"], 1)
        self.assertEqual(log["user placeholder"], 1)

    def test_tokenize_text_drops_punctuation(self):
        self.assertEqual(tokenize_text("chbik ? c'est 2024 normal"), ["chbik", "c'est", "normal"])


class TestMetrics(unittest.TestCase):
    def test_switch_points(self):
        self.assertEqual(switch_points(["TUN", "TUN", "FOR", "FOR", "TUN"]), {2, 4})

    def test_perfect_prediction(self):
        s = make_sentence(["ena", "mech", "demain"], ["TUN", "TUN", "FOR"])
        r = evaluate([s], [s["labels"]], {"ena": {"TUN": 1}})
        self.assertEqual(r["FOR_F1"], 1.0)
        self.assertEqual(r["switch_point_F1"], 1.0)
        self.assertEqual(r["cs_sentence_F1"], 1.0)


if __name__ == "__main__":
    unittest.main()
