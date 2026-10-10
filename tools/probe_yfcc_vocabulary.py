"""Build the original min-frequency dictionary without performing training."""
from pathlib import Path
import sys
import fasttext
from gensim_lab.common import write_json
from gensim_lab.runtime import training_threads

sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parents[1]
raw = root / "data/paper_reproduction/raw/yfcc100m"
out = root / "artifacts/paper_reproduction"
model = fasttext.train_supervised(input=str(raw / "train"), dim=1, epoch=0, lr=0.1,
                                  minCount=100, minCountLabel=100, wordNgrams=1,
                                  minn=0, maxn=0, bucket=0, loss="hs", thread=training_threads(), verbose=2)
words, word_frequencies = model.get_words(include_freq=True)
labels, label_frequencies = model.get_labels(include_freq=True)
record = {"word_count": len(words), "label_count": len(labels), "epochs": 0,
          "min_count": 100, "min_count_label": 100,
          "words": dict(zip(words, map(int, word_frequencies), strict=True)),
          "labels": {label.removeprefix('__label__'): int(frequency)
                     for label, frequency in zip(labels, label_frequencies, strict=True)}}
write_json(out / "yfcc_vocabulary.json", record)
print(f"Original YFCC vocabulary: {len(words):,} words, {len(labels):,} labels", flush=True)
