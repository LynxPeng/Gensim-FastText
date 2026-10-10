"""One isolated full-scale paper experiment; no writes to the old experiment index."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import fasttext
import joblib
import numpy as np
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.svm import LinearSVC

from .common import ROOT, PeakMemory, file_hash, read_json, stamp, write_json
from .paper_data import PAPER_ARTIFACTS, TextCorpus, partition_path, rows, settings, unlabeled_corpus


class EpochLog(CallbackAny2Vec):
    def __init__(self, directory):
        self.directory = Path(directory)
        history = self.directory / "embedding_epochs.json"
        self.records = read_json(history) if history.exists() else []

    def on_epoch_begin(self, model):
        self.begin = time.monotonic()

    def on_epoch_end(self, model):
        self.records.append({"epoch": len(self.records) + 1, "seconds": time.monotonic() - self.begin,
                             "checkpoint_ready": True})
        # Epoch boundary checkpoint: no asynchronous jobs are left in this epoch.
        model.paper_epoch_records = self.records.copy()
        temporary = self.directory / "embedding_checkpoint.gensim.tmp"
        model.save(str(temporary), separately=[])
        temporary.replace(self.directory / "embedding_checkpoint.gensim")
        write_json(self.directory / "embedding_epochs.json", self.records)
        print(f"Embedding epoch {len(self.records)} complete in {self.records[-1]['seconds']:.1f}s", flush=True)


def document_vector(model, words, dimension, cache=None):
    counts = collections.Counter(words)
    vector, accepted = np.zeros(dimension, np.float32), 0
    missing = 0
    for token, frequency in counts.items():
        index = model.wv.key_to_index.get(token)
        if index is not None:
            value = model.wv.vectors[index]
        else:
            missing += frequency
            try:
                value = cache[token] if cache is not None and token in cache else model.wv.get_vector(token)
                if cache is not None and len(cache) < 100000:
                    cache[token] = value
            except KeyError:
                continue
        vector += value * frequency
        accepted += frequency
    if accepted:
        vector /= accepted
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite document vector")
    return vector, missing, not bool(accepted)


def embed_documents(model, path, n, dimension, output):
    """Use a disk-backed feature matrix; aggregate repeated tokens with counts."""
    receipt = output.with_suffix(".json")
    expected = {"n": n, "dimension": dimension, "input_sha256": file_hash(path),
                "representation_sha256": file_hash(output.parent / "representation.gensim")}
    if receipt.exists() and read_json(receipt).get("spec") == expected and output.exists() and output.stat().st_size == n * dimension * 4:
        return np.memmap(output, np.float32, mode="r", shape=(n, dimension)), read_json(receipt)["coverage"]
    result = np.memmap(output, np.float32, mode="w+", shape=(n, dimension))
    oov = total = zero = 0
    count = 0
    vector_cache = {}
    for index, _, words in rows(path):
        vector, missing, empty = document_vector(model, words, dimension, vector_cache)
        total += len(words)
        oov += missing
        zero += empty
        result[index] = vector
        count = index + 1
        if count % 1000000 == 0:
            result.flush()
            print(f"Embed {path.name}: {count:,}/{n:,}", flush=True)
    if count != n:
        raise ValueError(f"Feature row count changed: {count} != {n}")
    result.flush()
    coverage = {"tokens": total, "oov_tokens": oov, "oov_rate": oov / max(1, total), "zero_documents": zero}
    write_json(receipt, {"spec": expected, "coverage": coverage, "bytes": output.stat().st_size})
    return result, coverage


def encode_targets(manifest, directory):
    label_map = {label: index for index, label in enumerate(manifest["labels"])}
    n = manifest["partitions"]["train"]["rows"]
    maximum = sum(manifest["partitions"]["train"]["label_counts"].values())
    offsets_path, target_path = directory / "label_offsets.i64", directory / "label_indices.i32"
    receipt = directory / "label_indices.json"
    if receipt.exists():
        previous = read_json(receipt)
        if previous["data_fingerprint"] == manifest["fingerprint"] and offsets_path.stat().st_size == (n + 1) * 8:
            return np.memmap(offsets_path, np.int64, mode="r", shape=(n + 1,)), np.memmap(target_path, np.int32, mode="r", shape=(previous["targets"],))
    offsets = np.memmap(offsets_path, np.int64, mode="w+", shape=(n + 1,))
    indices = np.memmap(target_path, np.int32, mode="w+", shape=(maximum,))
    position = 0
    for index, labels, _ in rows(partition_path(manifest, "train")):
        offsets[index] = position
        values = [label_map[label] for label in labels if label in label_map]
        indices[position:position + len(values)] = values
        position += len(values)
    offsets[n] = position
    offsets.flush()
    indices.flush()
    write_json(receipt, {"data_fingerprint": manifest["fingerprint"], "targets": position,
                         "selection": "uniform among a document's training labels each epoch"})
    return offsets, indices


def evaluate(manifest, part, predict_batch, directory):
    labels = manifest["labels"]
    label_map = {label: i for i, label in enumerate(labels)}
    truth_count, prediction_count, true_positive = [np.zeros(len(labels), np.int64) for _ in range(3)]
    confusion = np.zeros((len(labels), len(labels)), np.int64) if len(labels) <= 20 else None
    n = hits = truth_total = unseen = 0
    start = time.monotonic()
    path = directory / f"predictions_{part}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "true_labels", "prediction", "score", "log_score", "score_type", "hit_at_1"])
        batch = []

        def consume(records):
            nonlocal n, hits, truth_total, unseen
            predicted, scores, score_type = predict_batch(records)
            if len(predicted) != len(records) or not np.isfinite(scores).all():
                raise ValueError("Prediction shape or scores are invalid")
            for (row, true_labels, _), prediction, score in zip(records, predicted, scores, strict=True):
                hit = prediction in true_labels
                n += 1
                hits += hit
                truth_total += len(true_labels)
                for label in true_labels:
                    if label in label_map:
                        truth_count[label_map[label]] += 1
                    else:
                        unseen += 1
                if prediction not in label_map:
                    raise ValueError("Classifier produced an unknown label")
                prediction_count[label_map[prediction]] += 1
                true_positive[label_map[prediction]] += hit
                if confusion is not None and len(true_labels) == 1:
                    confusion[label_map[true_labels[0]], label_map[prediction]] += 1
                log_score = float(np.log(np.clip(score, np.finfo(float).tiny, 1))) if score_type == "probability" else ""
                writer.writerow([row, json.dumps(true_labels, ensure_ascii=False), prediction, float(score), log_score, score_type, int(hit)])

        for record in rows(partition_path(manifest, part)):
            batch.append(record)
            if len(batch) == 4096:
                consume(batch)
                batch = []
        if batch:
            consume(batch)
    if n != manifest["partitions"][part]["rows"]:
        raise ValueError("Prediction count does not cover the complete split")
    precision = np.divide(true_positive, prediction_count, out=np.zeros(len(labels), float), where=prediction_count != 0)
    recall = np.divide(true_positive, truth_count, out=np.zeros(len(labels), float), where=truth_count != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(labels), float), where=precision + recall != 0)
    per_class = directory / f"per_class_{part}.csv"
    with per_class.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "support", "predicted", "true_positive", "precision", "recall", "f1"])
        writer.writerows(zip(labels, truth_count, prediction_count, true_positive, precision, recall, f1, strict=True))
    result = {"n": n, "precision_at_1": hits / n, "recall_at_1": hits / max(1, truth_total),
              "macro_f1_over_training_labels": float(f1.mean()), "hits": int(hits),
              "truth_labels": truth_total, "unseen_truth_labels": unseen,
              "seconds": time.monotonic() - start, "prediction_sha256": file_hash(path)}
    if len(labels) <= 20:
        result["accuracy"] = result["precision_at_1"]
    write_json(directory / f"metrics_{part}.json", result)
    if confusion is not None:
        with (directory / f"confusion_{part}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true/predicted", *labels])
            writer.writerows([label, *values] for label, values in zip(labels, confusion, strict=True))
    return result


def verify_reload(manifest, part, predictor, directory):
    """Compare every saved prediction, rather than certifying a small prefix."""
    count, maximum = 0, 0.0
    with (directory / f"predictions_{part}.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved = iter(csv.DictReader(stream))
        batch = []

        def compare(records):
            nonlocal count, maximum
            labels, values, kind = predictor(records)
            for record, label, value in zip(records, labels, values, strict=True):
                previous = next(saved)
                difference = abs(float(previous["score"]) - float(value))
                if (int(previous["row"]) != record[0] or json.loads(previous["true_labels"]) != record[1]
                        or previous["prediction"] != label or previous["score_type"] != kind or difference > 1e-6):
                    raise ValueError(f"Reload changed {part} prediction at row {record[0]}")
                count += 1
                maximum = max(maximum, difference)

        for record in rows(partition_path(manifest, part)):
            batch.append(record)
            if len(batch) == 4096:
                compare(batch)
                batch = []
        if batch:
            compare(batch)
        if next(saved, None) is not None or count != manifest["partitions"][part]["rows"]:
            raise ValueError("Saved prediction count differs after reload")
    return {"samples": count, "same_labels_and_scores": True, "max_score_difference": maximum}


def run(directory):
    directory = Path(directory)
    spec = read_json(directory / "spec.json")
    for name, digest in spec["protocol_sha256"].items():
        if file_hash(ROOT / "gensim_lab" / name) != digest:
            raise ValueError(f"Source changed since scheduling: {name}")
    configuration = spec["configuration"]
    manifest = read_json(ROOT / spec["manifest"])
    if manifest["fingerprint"] != spec["data_fingerprint"]:
        raise ValueError("Prepared data differs from the scheduled experiment")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n = manifest["partitions"]["train"]["rows"]
    summary = {"spec": spec, "started": stamp(), "metrics": {}}
    with PeakMemory() as monitor:
        if spec["method"] == "native_fasttext":
            native_params = configuration[spec["task"]]
            native_path = directory / "native.bin"
            if native_path.exists():
                model = fasttext.load_model(str(native_path))
            else:
                minimum = 100 if spec["task"] == "tags" and not spec["smoke"] else 1
                model = fasttext.train_supervised(input=str(partition_path(manifest, "train")),
                    dim=spec["dimension"], epoch=native_params["native_epochs"], lr=spec["learning_rate"], loss=native_params["native_loss"], minCount=minimum, minCountLabel=minimum,
                    wordNgrams=spec["word_ngrams"], bucket=native_params["native_bucket"] if "native_bucket" in native_params else 10000000, minn=0, maxn=0,
                    thread=native_params["native_threads"], seed=spec["seed"], verbose=2)
                model.save_model(str(native_path))
            def predictor(batch):
                text = [" ".join(words) for _, _, words in batch]
                predictions, probabilities = model.predict(text, k=1)
                return [values[0].removeprefix("__label__") for values in predictions], [float(values[0]) for values in probabilities], "probability"
        else:
            params = dict(configuration["embedding"])
            params.update(vector_size=spec["dimension"], seed=spec["seed"])
            if spec["task"] == "tags" and not spec["smoke"]:
                params["min_count"] = configuration["tags"]["gensim_min_count"]
            corpus = unlabeled_corpus(manifest)
            saved = directory / "representation.gensim"
            if saved.exists():
                model = FastText.load(str(saved))
            else:
                callback = EpochLog(directory)
                checkpoint = directory / "embedding_checkpoint.gensim"
                if checkpoint.exists():
                    model = FastText.load(str(checkpoint))
                    callback.records = model.paper_epoch_records
                    write_json(directory / "embedding_epochs.json", callback.records)
                    remaining = params["epochs"] - len(callback.records)
                    if remaining > 0:
                        start_alpha = params["alpha"] + (params["min_alpha"] - params["alpha"]) * len(callback.records) / params["epochs"]
                        model.train(corpus_file=str(corpus), total_words=model.corpus_total_words,
                                    epochs=remaining, start_alpha=start_alpha, end_alpha=params["min_alpha"], callbacks=[callback])
                else:
                    # Build with the same ASCII tokenizer as the native dictionary.
                    # Gensim's default LineSentence would also split Unicode spaces.
                    model = FastText(**params)
                    model.build_vocab(corpus_iterable=TextCorpus(partition_path(manifest, "train")))
                    model.train(corpus_file=str(corpus), total_words=model.corpus_total_words,
                                epochs=params["epochs"], callbacks=[callback])
                model.save(str(saved), separately=[])
            features, coverage = embed_documents(model, partition_path(manifest, "train"), n,
                                                 spec["dimension"], directory / "features_train.f32")
            summary["training_coverage"] = coverage
            if spec["task"] == "classification":
                classifier_path = directory / "classifier.joblib"
                if classifier_path.exists():
                    classifier = joblib.load(classifier_path)
                else:
                    truth = np.fromiter((int(labels[0]) for _, labels, _ in rows(partition_path(manifest, "train"))), np.int32, count=n)
                    classifier = LinearSVC(C=configuration["classification"]["classifier_c"], dual=False, random_state=spec["seed"], max_iter=20000).fit(features, truth)
                    joblib.dump(classifier, classifier_path)
            else:
                from .paper_hs import HuffmanClassifier
                if (directory / "classifier_hs.npz").exists():
                    classifier = HuffmanClassifier.load(directory)
                else:
                    offsets, indices = encode_targets(manifest, directory)
                    classifier = HuffmanClassifier(manifest["labels"],
                        [manifest["training_label_counts"][label] for label in manifest["labels"]],
                        spec["dimension"], spec["seed"])
                    tag_parameters = configuration["tags"]
                    classifier.fit(features, offsets, indices, directory,
                        epochs=tag_parameters["classifier_epochs"], learning_rate=tag_parameters["classifier_learning_rate"],
                        batch_size=tag_parameters["classifier_batch_size"], device=tag_parameters["classifier_device"])
                    classifier.save(directory)
            del features

            def predictor(batch):
                vectors, vector_cache = [], {}
                for _, _, words in batch:
                    vector, _, _ = document_vector(model, words, spec["dimension"], vector_cache)
                    vectors.append(vector)
                matrix = np.asarray(vectors, dtype=np.float32)
                if spec["task"] == "classification":
                    predictions = [str(value) for value in classifier.predict(matrix)]
                    margins = classifier.decision_function(matrix)
                    scores = np.abs(margins) if margins.ndim == 1 else margins.max(axis=1)
                    return predictions, scores, "svm_margin"
                values, log_probability, _ = classifier.predict(matrix)
                return [manifest["labels"][value] for value in values], np.exp(log_probability), "probability"

        summary["fit_seconds"] = monitor.elapsed if hasattr(monitor, "elapsed") else time.perf_counter() - monitor.started
        # Validation remains separate for YFCC; no test-driven parameter selection.
        for part in ["validation", "test"]:
            if part in manifest["partitions"]:
                summary["metrics"][part] = evaluate(manifest, part, predictor, directory)
        if spec["method"] == "native_fasttext":
            model = fasttext.load_model(str(directory / "native.bin"))
        else:
            model = FastText.load(str(directory / "representation.gensim"))
            classifier = (joblib.load(directory / "classifier.joblib") if spec["task"] == "classification"
                          else HuffmanClassifier.load(directory))
        checks = {part: verify_reload(manifest, part, predictor, directory) for part in summary["metrics"]}
        write_json(directory / "save_load_check.json", {"samples": sum(item["samples"] for item in checks.values()),
                    "same_labels_and_scores": True, "partitions": checks,
                    "max_score_difference": max(item["max_score_difference"] for item in checks.values())})
    summary.update(completed=stamp(), peak_memory_mib=monitor.peak / 2**20, total_seconds=monitor.elapsed)
    # Resource usage excludes disposable document matrices.
    summary["model_bytes"] = sum(path.stat().st_size for path in directory.iterdir()
                                  if path.suffix in {".gensim", ".bin", ".joblib", ".npz"}
                                  and "checkpoint" not in path.name)
    summary["seal"] = {path.name: file_hash(path) for path in directory.iterdir()
                        if path.name in {"spec.json", "native.bin", "representation.gensim", "classifier.joblib",
                                         "classifier_hs.npz", "classifier_hs.json", "metrics_test.json", "predictions_test.csv",
                                         "metrics_validation.json", "predictions_validation.csv", "save_load_check.json"}}
    write_json(directory / "summary.json", summary)
    print(f"Completed {spec['dataset']}/{spec['method']}/seed {spec['seed']}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run(args.run)
