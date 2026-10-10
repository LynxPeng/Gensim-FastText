"""Compatibility aliases for archived models; no counters or worker wrappers."""
from gensim.models import Word2Vec, FastText
ParallelLossWord2Vec = Word2Vec
ParallelCorpusFastText = FastText
