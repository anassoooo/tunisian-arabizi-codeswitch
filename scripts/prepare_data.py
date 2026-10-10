"""Clean, de-duplicate and split TArC into data/processed/{train,dev,test}.jsonl.

    python scripts/prepare_data.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tuncs.data import prepare_splits  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(prepare_splits(), indent=2))
