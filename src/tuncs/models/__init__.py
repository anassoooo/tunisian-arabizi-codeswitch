"""Word-level language taggers sharing one interface:

    model.fit(train, dev)       -> model
    model.predict(sentences)    -> list[list[str]]   (one label per word)
    model.save(directory) / Model.load(directory)
"""

from .baselines import LexiconTagger, MajorityTagger
from .crf import CRFTagger
from .logreg import CharNgramTagger

__all__ = ["MajorityTagger", "LexiconTagger", "CharNgramTagger", "CRFTagger"]
# BiLSTMCRFTagger (models.bilstm_crf) and XLMRTagger (models.xlmr) need torch / transformers and are imported explicitly.
