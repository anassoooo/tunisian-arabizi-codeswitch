# Word-level detection of French/English code-switching in Latin-script Tunisian Arabic

## 1. Introduction

Tunisians often write their dialect online in Latin letters and digits, a style called *Arabizi*: `3andi` (I have), `n7ebek` (I love you), `barcha` (a lot). They also switch freely into French, and sometimes English, within the same sentence. **Code-switching detection** means labelling each word with its language. Downstream tools such as sentiment analysis, machine translation and transliteration need this step.

In Arabic-script corpora the task is almost trivial, because Tunisian words are in Arabic letters and foreign words in Latin letters. In Arabizi both languages share one alphabet. The cues are subtle:

- digits used as letters (`3` = ع, `7` = ح, `9` = ق, `5` = خ);
- French spelling and accents;
- **context**, for words that exist in both languages (`la`, `ma`, `en`, `ou`).

**Research question.** How well can classical and neural models separate Tunisian from foreign words in Arabizi text? How well do they handle unseen spellings and ambiguous words?

## 2. Data

**Corpus.** We use TArC, the Tunisian Arabish Corpus (Gugliotta & Dinarelli, 2022): 4,798 sentences from social media, forums, blogs and rap lyrics. Each token is labelled `arabizi`, `foreign` or `emotag`, and the authors checked every label by hand.

**Cleaning** (`src/tuncs/data.py`):
- Kept: lexical tokens, mapped `arabizi` → TUN and `foreign` → FOR.
- Removed:
  - 617 emoticon tokens;
  - 4,148 tokens without letters (punctuation, numbers);
  - 2,986 anonymised user-name placeholders (`m5abbiNNNN`).
- Fixed:
  - an escaped apostrophe (`j\'ai` → `j'ai`);
  - one unlabelled token, `yarhmek`, set to TUN.

That leaves 3,960 sentences and 35,577 words.

**Duplicates and split.** We removed 502 exact duplicate sentences so that no sentence can appear in both train and test. The remaining 3,458 sentences were split 80/10/10 with seed 42:

| Split | Sentences | Words | Foreign words | Code-switched sentences | Words unseen in train |
|---|---|---|---|---|---|
| Train | 2,766 | 27,474 | 28.9 % | 37.3 % | – |
| Dev | 345 | 3,285 | 29.4 % | 38.3 % | 34.4 % |
| Test | 347 | 3,378 | 30.3 % | 37.5 % | 31.4 % |

**Observations from the data analysis:**
- 27 % of Tunisian words contain a digit used as a letter; only 0.6 % of foreign words do.
- 10 % of foreign words carry an accent; only 2.5 % of Tunisian words do.
- Most words carry neither cue.
- 123 word forms appear with **both** labels (`el`, `la`, `le`, `ya`, `ou`, `ma`, `je`, `men`, `en`, ...), and they make up 11.2 % of all words.
- Genres differ a lot: forum posts are 53 % French, while blog and rap texts are about 12 %.

**Normalisation.** Words are lower-cased and letters repeated 3+ times are squeezed (`loooool` → `lool`). Digits and accents are kept on purpose, because they are the main cues.

## 3. Methods

All models share one interface (`fit`, `predict`, `save`, `load`) in `src/tuncs/models/`. Hyper-parameters were chosen on dev macro F1, and the test set was used once per model.

1. **Majority baseline**: every word is TUN.
2. **Lexicon lookup**: each word gets its most frequent training label; unseen words get TUN.
3. **Character n-gram logistic regression**: TF-IDF features over character 1–5-grams at word boundaries, a class-balanced logistic regression, and C chosen from {0.3, 1, 3, 10, 30} (chosen: 30). This model looks at each word on its own, with no context.
4. **CRF** (linear-chain, `sklearn-crfsuite`).
   - Features for each word: its form, prefixes and suffixes, digit-as-letter, accent, apostrophe, capitalisation and length.
   - The same form, suffix and cue features for the two previous and two next words.
   - L1/L2 penalties chosen on a grid (chosen: c1 = c2 = 0.01).
5. **BiLSTM-CRF** (Lample et al., 2016).
   - Architecture: a 100-dimensional word embedding plus a character-CNN vector (64 filters), fed into a BiLSTM (2×128) with a CRF output layer.
   - During training, 10 % of words are replaced by an unknown-word token, so the model learns to rely on characters for new spellings.
   - Training: Adam (lr 2e-3), early stopping on dev.
6. **XLM-RoBERTa base** (Conneau et al., 2020).
   - Fine-tuning: all weights trained for 5 epochs with AdamW (lr 3e-5, 10 % warm-up, linear decay), batch size 16, mixed precision.
   - Each word is labelled from its first sub-word; long sentences are cut into 64-word chunks.
   - The best dev epoch is kept.

**Metrics** (`src/tuncs/metrics.py`):
- foreign-word F1 (the main metric), with a 95 % confidence interval from 1,000 bootstrap resamples of test sentences;
- macro F1 over TUN and FOR;
- macro F1 on *unseen* words (never seen in train) and on *ambiguous* words (seen in train with both labels);
- sentence-level F1 for spotting code-switched sentences;
- switch-point F1, which checks the exact positions where the language changes.

## 4. Results

| Model | FOR F1 [95 % CI] | Macro F1 | Unseen | Ambiguous | Code-switched sentence | Switch point | Accuracy |
|---|---|---|---|---|---|---|---|
| Majority | 0.000 | 0.411 | 0.436 | 0.378 | 0.000 | 0.000 | 0.697 |
| Lexicon | 0.834 [0.808, 0.854] | 0.887 | 0.436 | 0.867 | 0.803 | 0.510 | 0.911 |
| Char n-gram LogReg | 0.946 [0.930, 0.959] | 0.961 | 0.938 | 0.852 | 0.877 | 0.776 | 0.966 |
| CRF | 0.959 [0.947, 0.969] | 0.971 | 0.939 | 0.957 | 0.881 | 0.816 | 0.975 |
| BiLSTM-CRF | 0.966 [0.954, 0.975] | 0.975 | 0.948 | 0.975 | 0.915 | 0.838 | 0.979 |
| **XLM-R** | **0.978 [0.968, 0.985]** | **0.984** | **0.967** | **0.977** | **0.961** | **0.904** | **0.987** |

**Cross-genre robustness.** We trained on social media, forum and blog text and tested on 509 rap sentences (8,277 words, 1,008 foreign):

| Model | FOR F1 on rap | FOR F1, random split |
|---|---|---|
| Lexicon | 0.462 | 0.834 |
| Char n-gram LogReg | 0.786 | 0.946 |
| CRF | 0.771 | 0.959 |

## 5. Discussion

- **Accuracy hides the problem.** The majority baseline already reaches 70 % accuracy.
- **Memorisation is not enough.** About 31 % of test words are unseen, because Arabizi spelling is not standardised. The lexicon gets them wrong 22.6 % of the time; XLM-R gets them wrong 2.3 % of the time.
- **Spelling solves unseen words; context solves ambiguous ones.** Character n-grams raise unseen-word F1 from 0.44 to 0.94. But that model can't use context, so its score on ambiguous words (0.85) is *lower* than the lexicon's. Every context model fixes this: CRF 0.96, BiLSTM-CRF 0.97, XLM-R 0.98. For `la` and `le`, which are mostly French in the test set, XLM-R is 100 % and 96 % correct; the lexicon is 91 % and 81 % correct.
- **Pre-training helps the most.** XLM-R is best on every metric. The largest gains are at sentence level (code-switched sentence F1 0.96) and on exact switch points (0.90). Its confidence interval does not overlap the lexicon's or the char n-gram model's. It slightly overlaps the CRF's and BiLSTM-CRF's, so those gaps are smaller and less certain.
- **Remaining errors.** XLM-R makes 45 errors out of 3,378 words: 29 missed foreign words and 16 false alarms. They fall into a few groups:
  - short function words such as `ou` (Tunisian "and" vs French "or");
  - English words and brand names (`damn`, `max`);
  - French-origin words written the Tunisian way (`flousse`, `ettourisme`, `clubiste`). Here even the gold label is debatable.
- **Domain shift.** Testing on an unseen genre (rap) lowers foreign-word F1 from about 0.95 to about 0.78. So random-split scores overestimate how the models would do on new kinds of text.

## 6. Limitations

- We used one random split and one training seed per model. The bootstrap interval reflects test-set sampling, not training randomness.
- The "foreign" class mixes French and English, and the authors put Modern Standard Arabic inside the `arabizi` class.
- TArC has no author IDs, so posts by the same author may appear in both train and test.
- We ran the cross-genre experiment only for the fast classical models.

## 7. Conclusion and future work

In Latin-script Tunisian, foreign-word detection can't rely on the alphabet, so it is a genuine sequence-labelling problem. Character-level features handle the open, unstandardised vocabulary. Context models handle the words shared by both languages. A fine-tuned multilingual transformer gives the best results: 0.978 foreign-word F1 and 0.90 switch-point F1.

Next steps:
- separate French from English;
- continue pre-training on unlabelled Arabizi (for example, the TUNIZI corpus);
- add training data by transliterating Arabic-script Tunisian corpora into Arabizi;
- evaluate across genres and authors.

## References

- E. Gugliotta, M. Dinarelli. *TArC: Tunisian Arabish Corpus, First complete release.* LREC 2022.
- E. Gugliotta, M. Dinarelli, O. Kraif. *Multi-Task Sequence Prediction for Tunisian Arabizi Multi-Level Annotation.* WANLP 2020.
- J. Lafferty, A. McCallum, F. Pereira. *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.* ICML 2001.
- G. Lample, M. Ballesteros, S. Subramanian, K. Kawakami, C. Dyer. *Neural Architectures for Named Entity Recognition.* NAACL 2016.
- A. Conneau et al. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.
