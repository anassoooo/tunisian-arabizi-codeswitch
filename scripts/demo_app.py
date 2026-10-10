"""Live demo web app: type a sentence and see how every saved model labels each word.

    python scripts/demo_app.py            # opens http://127.0.0.1:8000 in your browser
    python scripts/demo_app.py --port 8080 --no-browser

Uses only the Python standard library for the server, runs fully offline, and loads the models from models/.
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tuncs.data import LETTER, make_sentence  # noqa: E402

from demo import load_models  # noqa: E402

PAGE = (ROOT / "app" / "index.html").read_bytes()
SHORT = {"Lexicon lookup": "Lexicon", "Char n-gram LogReg": "Char n-grams", "CRF (features + context)": "CRF",
         "BiLSTM-CRF (word + char CNN)": "BiLSTM-CRF", "XLM-R base (fine-tuned)": "XLM-R"}
MODELS = {}
LOCK = threading.Lock()


def tag(text: str) -> dict:
    raw = text.split()[:120]
    scored = [i for i, t in enumerate(raw) if LETTER.search(t)]
    s = make_sentence([raw[i] for i in scored])
    rows = []
    if s["tokens"]:
        with LOCK:  # one GPU, one request at a time
            for name, m in MODELS.items():
                t0 = time.perf_counter()
                labels = [str(l) for l in m.predict([s])[0]]
                proba = m.predict_proba([s])[0] if hasattr(m, "predict_proba") else None
                rows.append({"name": name, "short": SHORT.get(name, name), "labels": labels, "p_for": proba,
                             "ms": round(1000 * (time.perf_counter() - t0), 1)})
    return {"words": [{"text": t, "scored": i in scored} for i, t in enumerate(raw)], "models": rows}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/tag":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            result = tag(str(body.get("text", "")))
            self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        except Exception as e:  # keep the demo alive whatever happens
            self._send(500, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    print("loading models ...")
    MODELS.update(load_models())
    if not MODELS:
        sys.exit("No saved models found in models/. Run scripts/train_evaluate.py first.")
    tag("ahla bik")  # warm-up so the first live request is fast
    print("loaded:", ", ".join(MODELS))

    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"demo running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
