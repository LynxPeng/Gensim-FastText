"""Compatibility names for archived pickles; instrumentation was removed."""
from .evaluation_support import save_csv, score_event_count, stable_hs_score, score_sentences
from .models import Representation

LossRepresentation = Representation

class EpochLoss:
    """State-only compatibility for previously serialized callbacks; no training hooks."""
    pass
