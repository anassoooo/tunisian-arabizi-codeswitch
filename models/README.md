# Saved models

These folders are created by `scripts/train_evaluate.py`, or by the notebook when `RETRAIN = True`. They are not tracked by git, because XLM-R alone takes about 1 GB.

| Folder | Model |
|---|---|
| `lexicon/` | word → label lookup table |
| `char_ngram_logreg/` | TF-IDF character n-grams + logistic regression |
| `crf/` | CRF with hand-crafted features |
| `bilstm_crf/` | BiLSTM-CRF weights + vocabularies |
| `xlmr/` | fine-tuned XLM-R (Hugging Face format) |

A `DONE` file inside a folder means that model finished training and can be reloaded.
