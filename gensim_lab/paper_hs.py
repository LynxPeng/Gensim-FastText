"""External frozen-vector Huffman classifier, not Gensim supervised FastText."""
from __future__ import annotations

import heapq
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from numba import njit

from .common import read_json, write_json


def huffman(counts):
    if len(counts) < 2 or any(value <= 0 for value in counts):
        raise ValueError("At least two positive-frequency training labels are required")
    size = len(counts)
    # Internal nodes: 0..size-2; leaves: size-1..2*size-2.
    parent = np.full(2 * size - 1, -1, dtype=np.int32)
    branch = np.zeros(2 * size - 1, dtype=np.int8)
    left, right = np.empty(size - 1, np.int32), np.empty(size - 1, np.int32)
    heap = [(int(count), index, size - 1 + index) for index, count in enumerate(counts)]
    heapq.heapify(heap)
    for internal in range(size - 1):
        weight_l, _, node_l = heapq.heappop(heap)
        weight_r, _, node_r = heapq.heappop(heap)
        left[internal], right[internal] = node_l, node_r
        parent[node_l] = parent[node_r] = internal
        branch[node_r] = 1
        heapq.heappush(heap, (weight_l + weight_r, size + internal, internal))
    paths, codes = [], []
    for label in range(size):
        nodes, bits, node = [], [], size - 1 + label
        while parent[node] != -1:
            nodes.append(parent[node])
            bits.append(branch[node])
            node = parent[node]
        paths.append(nodes[::-1])
        codes.append(bits[::-1])
    width = max(map(len, paths))
    padded = np.zeros((size, width), dtype=np.int32)
    targets = np.zeros((size, width), dtype=np.float32)
    mask = np.zeros((size, width), dtype=np.float32)
    for label, (nodes, bits) in enumerate(zip(paths, codes, strict=True)):
        padded[label, :len(nodes)] = nodes
        targets[label, :len(bits)] = bits
        mask[label, :len(nodes)] = 1
    return {"left": left, "right": right, "root": size - 2,
            "paths": padded, "targets": targets, "mask": mask}


@njit(cache=True)
def exact_top1(features, weights, left, right, root):
    """Best-first search: internal path probability bounds every descendant."""
    n_labels = len(left) + 1
    predictions = np.empty(len(features), np.int32)
    log_probabilities = np.empty(len(features), np.float64)
    visited = np.empty(len(features), np.int32)
    # Reuse a heap across rows. The full label count is a safe upper capacity.
    heap_nodes = np.empty(n_labels, np.int32)
    heap_prob = np.empty(n_labels, np.float64)
    for row in range(len(features)):
        heap_nodes[0], heap_prob[0], length, examined = root, 0.0, 1, 0
        while length:
            node, path_probability = heap_nodes[0], heap_prob[0]
            length -= 1
            if length:
                replacement_node, replacement_probability = heap_nodes[length], heap_prob[length]
                position = 0
                while position * 2 + 1 < length:
                    child = position * 2 + 1
                    if child + 1 < length and heap_prob[child + 1] > heap_prob[child]:
                        child += 1
                    if heap_prob[child] <= replacement_probability:
                        break
                    heap_nodes[position], heap_prob[position] = heap_nodes[child], heap_prob[child]
                    position = child
                heap_nodes[position], heap_prob[position] = replacement_node, replacement_probability
            examined += 1
            if node >= n_labels - 1:
                predictions[row] = node - (n_labels - 1)
                log_probabilities[row], visited[row] = path_probability, examined
                break
            score = 0.0
            for dimension in range(features.shape[1]):
                score += float(features[row, dimension]) * float(weights[node, dimension])
            if score >= 0:
                log_right = -math.log1p(math.exp(-score))
                log_left = -score + log_right
            else:
                log_left = -math.log1p(math.exp(score))
                log_right = score + log_left
            for child_node, child_probability in ((left[node], path_probability + log_left),
                                                   (right[node], path_probability + log_right)):
                position = length
                length += 1
                while position:
                    ancestor = (position - 1) // 2
                    if heap_prob[ancestor] >= child_probability:
                        break
                    heap_nodes[position], heap_prob[position] = heap_nodes[ancestor], heap_prob[ancestor]
                    position = ancestor
                heap_nodes[position], heap_prob[position] = child_node, child_probability
    return predictions, log_probabilities, visited


class HuffmanClassifier:
    def __init__(self, labels, counts, dimension, seed):
        self.labels = labels
        self.tree = huffman(counts)
        self.dimension, self.seed = dimension, seed
        self.weights = np.zeros((len(labels) - 1, dimension), np.float32)

    def fit(self, features, offsets, targets, directory, *, epochs=5, learning_rate=0.1,
            batch_size=8192, device="auto"):
        import torch
        import torch.nn.functional as functional
        from .runtime import numerical_threads
        torch.set_num_threads(numerical_threads())
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        directory = Path(directory)
        checkpoint = directory / "hs_checkpoint.pt"
        rng = np.random.default_rng(self.seed)
        weights = torch.nn.Parameter(torch.from_numpy(self.weights.copy()).to(device))
        optimizer = torch.optim.SGD([weights], lr=learning_rate)
        paths = torch.from_numpy(self.tree["paths"].astype(np.int64)).to(device)
        codes = torch.from_numpy(self.tree["targets"]).to(device)
        masks = torch.from_numpy(self.tree["mask"]).to(device)
        completed_rows, history, partial_count = 0, [], 0
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            with torch.no_grad():
                weights.copy_(saved["weights"].to(device))
            completed_rows, history = saved["completed_rows"], saved["history"]
            partial_count = saved["partial_count"]
            rng.bit_generator.state = saved["rng_state"]
        n = len(features)
        begin, last_checkpoint = time.monotonic(), time.monotonic()
        for epoch in range(completed_rows // n, epochs):
            first = completed_rows % n if epoch == completed_rows // n else 0
            for start in range(first, n, batch_size):
                stop = min(start + batch_size, n)
                lo = np.asarray(offsets[start:stop], dtype=np.int64)
                lengths = np.asarray(offsets[start + 1:stop + 1] - lo, dtype=np.int64)
                usable = lengths > 0
                positions = np.flatnonzero(usable) + start
                if len(positions):
                    lo, lengths = lo[usable], lengths[usable]
                    choice = lo + (rng.random(len(positions)) * lengths).astype(np.int64)
                    target = torch.from_numpy(np.array(targets[choice], dtype=np.int64)).to(device)
                    x = torch.from_numpy(np.array(features[positions], dtype=np.float32)).to(device)
                    optimizer.param_groups[0]["lr"] = learning_rate * max(0.0001, 1 - completed_rows / (epochs * n))
                    logits = (weights[paths[target]] * x[:, None, :]).sum(dim=2)
                    losses = (functional.binary_cross_entropy_with_logits(logits, codes[target], reduction="none") * masks[target]).sum(dim=1)
                    loss = losses.mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite hierarchical classifier loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    partial_count += len(positions)
                completed_rows = epoch * n + stop
                if time.monotonic() - last_checkpoint > 120 or stop == n:
                    if stop == n:
                        if not partial_count:
                            raise ValueError("No labeled examples in the complete training epoch")
                        history.append({"epoch": epoch + 1, "documents": partial_count})
                        partial_count = 0
                    temporary = checkpoint.with_suffix(".pt.tmp")
                    torch.save({"weights": weights.detach().cpu(), "completed_rows": completed_rows,
                                "history": history,
                                "partial_count": partial_count, "rng_state": rng.bit_generator.state}, temporary)
                    temporary.replace(checkpoint)
                    print(f"External HS epoch {epoch + 1}: {stop:,}/{n:,}, device {device}", flush=True)
                    last_checkpoint = time.monotonic()
        self.weights = weights.detach().cpu().numpy().copy()
        self.history = history
        self.runtime = {"device": device, "torch": torch.__version__, "seconds": time.monotonic() - begin,
                        "batch_size": batch_size, "exact_top1_search": True,
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None}
        return self

    def predict(self, features):
        return exact_top1(np.asarray(features), self.weights, self.tree["left"], self.tree["right"], self.tree["root"])

    def save(self, directory):
        directory = Path(directory)
        np.savez(directory / "classifier_hs.npz", weights=self.weights, **self.tree)
        write_json(directory / "classifier_hs.json", {"labels": self.labels, "dimension": self.dimension,
                  "seed": self.seed, "history": self.history, "runtime": self.runtime})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = read_json(directory / "classifier_hs.json")
        instance = cls.__new__(cls)
        instance.__dict__.update(metadata)
        arrays = np.load(directory / "classifier_hs.npz")
        instance.weights = arrays["weights"]
        instance.tree = {key: arrays[key] for key in arrays.files if key != "weights"}
        instance.tree["root"] = int(instance.tree["root"])
        return instance
