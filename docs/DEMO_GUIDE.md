# Live demo guide (about 5 minutes)

## Before presenting

1. Activate the project's Python environment, open a terminal in the project folder and start the app:
   ```bash
   python scripts/demo_app.py
   ```
2. Wait for `demo running at http://127.0.0.1:8000`. The browser opens by itself. Loading the models takes about 20 seconds; keep the window open.
3. Zoom the browser to 125–150 % so the back of the room can read it.
4. It runs **offline**: no internet is needed. Close other programs that use the GPU.
5. If port 8000 is busy, add `--port 8080`.
6. The page follows your system's light/dark setting. To force one, open `http://127.0.0.1:8000/?theme=light` (best on projectors) or `?theme=dark`.

**Backup, if the browser fails:** the command-line demo shows the same output:
```bash
python scripts/demo.py "3andi rendez-vous m3a el docteur demain"
```
The executed notebook (section 14) also already contains demo outputs.

## What the screen shows

- **Big sentence at the top:** what the best model (XLM-R) says. Orange means foreign, blue means Tunisian.
- **Counters:** how many words, foreign words and language switches the sentence has.
- **"All models, word by word" table:** the five models compared. A yellow **◆** marks words where they disagree; that is where the interesting part is.
- **XLM-R confidence:** a thin bar under each XLM-R word shows how confident it is. Hover a word to see the exact probability.

## Suggested script

Click the example buttons in this order. The outputs below were checked with the saved models.

### 1. "There is no alphabet shortcut"
Type `c'est la vie ya sa7bi`.
> *"Both languages are written in Latin letters, so the model can't just look at the script. Here every model handles a classic phrase easily. Now let's make it harder."*

### 2. Words never seen in training → characters matter
Click **`3andi rendez-vous m3a el docteur demain`**.
> *"The Lexicon model only memorises training words. It has never seen `rendez-vous`, `docteur` or `demain`, so it calls everything Tunisian. The character-based models recognise French spelling patterns and catch them. About a third of test words are unseen, because Arabizi has no standard spelling."*

Then click **`ena fi retard 5ater el metro ma jech`**: only the character n-gram model and XLM-R catch `retard` and `metro`.

### 3. Same spelling, two languages → context matters
Click **`ou ba3d chnowa sar ?`**.
> *"`ou` is French for 'or', but in Tunisian it means 'and'. The character n-gram model looks at one word at a time, so it says French. The context models (CRF, BiLSTM-CRF, XLM-R) read the whole sentence and get it right: Tunisian."*

### 4. An honest hard case
Click **`la la ma n7ebch nemchi la fac lyoum`** ("no, no, I don't want to go to uni today").
> *"The first two `la` mean 'no' in Tunisian, and the third is the French article in `la fac`. The BiLSTM-CRF gets all three right. XLM-R, our best model overall, misses `la fac` here. One sentence isn't the whole story: on the full test set XLM-R is the best model on ambiguous words, 0.98 vs 0.87 for the Lexicon."*

### 5. English too
Click **`nchallah el weekend jey nemchiw lel plage`**.
> *"Only XLM-R spots the English word `weekend`. Its multilingual pre-training helps with words that are rare in our training data."*

### 6. Let the audience try
Ask someone in the audience for a sentence and type it in. Good inputs mix French nouns into Tunisian grammar: `taw nab3athlek el message sur facebook`, `el prof 9al l'examen sera facile`.

## Likely questions

| Question | Short answer |
|---|---|
| Why is accuracy not the main metric? | 70 % of words are Tunisian, so labelling everything Tunisian already gets 70 % accuracy. We use foreign-word F1. |
| Why does XLM-R sometimes disagree with the others? | It uses context and multilingual knowledge. It's the best overall (0.978 foreign-word F1), but no model is perfect on single sentences. |
| Is `merci` Tunisian or French? | Both! Borrowed words are labelled differently in different contexts in the corpus, so even the gold labels are debatable. |
| Would it work on other text? | Performance drops on unseen genres: trained without rap, tested on rap, F1 falls from 0.95 to 0.78. |
| Where does the data come from? | TArC (Gugliotta & Dinarelli, LREC 2022): about 4,800 manually checked sentences from social media, forums, blogs and rap. |
