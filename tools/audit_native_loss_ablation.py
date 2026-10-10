"""Quantify the one native-protocol deviation found in the audit: loss=hs vs the author script's softmax.

The archived v0.9.2 classification-results.sh passes no -loss flag, so Table 1 used softmax;
configs/paper_reproduction.yaml sets native_loss: hs for the eight classification datasets.

Runs on the already-prepared ag_news partition; native only, no Gensim.
Run: conda run -n Gensim python tools/audit_native_loss_ablation.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/paper_reproduction/prepared/full/ag_news"
OUT = ROOT / "artifacts/reviews/native_loss_ablation.json"
CSV_OUT = ROOT / "reports/data/native_loss_ablation.csv"
SEEDS = [42, 2026, 7]
LOSSES = ["softmax", "hs"]
NGRAMS = [1, 2]
PAPER = {1: 91.5, 2: 92.5}
LR = 0.25  # author-published value for ag_news


def main():
    train, test = DATA / "train.txt", DATA / "test.txt"
    rows = []
    for loss in LOSSES:
        for ngram in NGRAMS:
            for seed in SEEDS:
                started = time.perf_counter()
                model = fasttext.train_supervised(
                    input=str(train), dim=10, epoch=5, lr=LR, loss=loss, minCount=1, minCountLabel=1,
                    wordNgrams=ngram, bucket=10000000, minn=0, maxn=0, thread=20, seed=seed, verbose=0)
                n, precision, recall = model.test(str(test))
                rows.append({"loss": loss, "word_ngrams": ngram, "seed": seed, "n": n,
                             "accuracy": precision, "seconds": time.perf_counter() - started})
                print(f"{loss:<8} ngram={ngram} seed={seed:<5} acc={precision * 100:.3f} "
                      f"({rows[-1]['seconds']:.1f}s)", flush=True)

    summary = []
    for loss in LOSSES:
        for ngram in NGRAMS:
            values = [row["accuracy"] for row in rows if row["loss"] == loss and row["word_ngrams"] == ngram]
            summary.append({"loss": loss, "word_ngrams": ngram, "n_seeds": len(values),
                            "accuracy_mean_percent": statistics.mean(values) * 100,
                            "accuracy_stdev_percent": statistics.stdev(values) * 100 if len(values) > 1 else 0.0,
                            "paper_percent": PAPER[ngram]})
    for entry in summary:
        print(f"{entry['loss']:<8} ngram={entry['word_ngrams']} "
              f"mean={entry['accuracy_mean_percent']:.2f}±{entry['accuracy_stdev_percent']:.2f} "
              f"paper={entry['paper_percent']}", flush=True)

    # Same configuration repeated: quantifies 20-thread run-to-run variability, which the seed
    # standard deviation does not separate from genuine seed effects.
    repeats = []
    for repeat in range(4):
        model = fasttext.train_supervised(
            input=str(train), dim=10, epoch=5, lr=LR, loss="softmax", minCount=1, minCountLabel=1,
            wordNgrams=1, bucket=10000000, minn=0, maxn=0, thread=20, seed=42, verbose=0)
        _, precision, _ = model.test(str(test))
        repeats.append(precision)
        print(f"repeat {repeat}: softmax ngram=1 seed=42 acc={precision * 100:.3f}", flush=True)
    repeatability = {"configuration": "softmax/ngram=1/seed=42", "repeats": repeats,
                     "accuracy_percent": [value * 100 for value in repeats],
                     "range_percent": (max(repeats) - min(repeats)) * 100,
                     "stdev_percent": statistics.stdev(repeats) * 100}
    print(f"repeat range={repeatability['range_percent']:.3f} stdev={repeatability['stdev_percent']:.3f}",
          flush=True)

    result = {"dataset": "ag_news", "dimension": 10, "epochs": 5, "learning_rate": LR,
              "threads": 20, "seeds": SEEDS,
              "note": "author script (v0.9.2 classification-results.sh) passes no -loss flag, so Table 1 used softmax",
              "rows": rows, "summary": summary, "repeatability": repeatability}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["loss", "word_ngrams", "seed", "n", "accuracy", "seconds"])
        writer.writerows([[row["loss"], row["word_ngrams"], row["seed"], row["n"],
                           f"{row['accuracy']:.6f}", f"{row['seconds']:.3f}"] for row in rows])
    print(f"\nWrote {OUT.relative_to(ROOT)} and {CSV_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
