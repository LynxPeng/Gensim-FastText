"""Compare transparent evaluation masks with the paper's reported denominator."""
from pathlib import Path
import sys
from gensim_lab.common import read_json, write_json
from gensim_lab.paper_data import rows

sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parents[1]
vocab = read_json(root / "artifacts/paper_reproduction/yfcc_vocabulary.json")
known_words, known_labels = set(vocab["words"]) - {"</s>"}, set(vocab["labels"])
result = {}
for part in ["test", "valid"]:
    counts = {"raw": 0, "has_raw_label_and_text": 0, "has_retained_label": 0,
              "has_retained_label_and_raw_text": 0, "has_retained_label_and_retained_word": 0}
    for _, labels, words in rows(root / "data/paper_reproduction/raw/yfcc100m" / part):
        counts["raw"] += 1
        retained = any(label in known_labels for label in labels)
        counts["has_raw_label_and_text"] += bool(labels and words)
        counts["has_retained_label"] += retained
        counts["has_retained_label_and_raw_text"] += bool(retained and words)
        counts["has_retained_label_and_retained_word"] += bool(retained and any(word in known_words for word in words))
    result[part] = counts
    print(part, counts, flush=True)
write_json(root / "artifacts/paper_reproduction/yfcc_denominator_check.json", result)
