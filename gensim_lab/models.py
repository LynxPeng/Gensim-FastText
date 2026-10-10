from __future__ import annotations

import collections
import hashlib
import time

import fasttext
import joblib
import numpy as np
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from gensim.models import Doc2Vec, FastText, LsiModel, TfidfModel, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


class SentenceCorpus:
    """Re-iterable corpus; epochs must not exhaust a single generator."""
    def __init__(self, docs):
        self.docs = docs

    def __iter__(self):
        for d in self.docs:
            yield from d["sentences"]


class Representation:
    def __init__(self, method, parameters, dictionary_parameters, seed):
        self.method = method
        self.parameters = parameters
        self.dictionary_parameters = dictionary_parameters
        self.seed = seed
        self.model = None
        self.dictionary = None
        self.tfidf = None

    def fit(self, docs):
        words = [d["tokens"] for d in docs]
        self.dictionary = Dictionary(words)
        self.dictionary.filter_extremes(**self.dictionary_parameters)
        if not len(self.dictionary):
            raise ValueError("Empty training dictionary after filtering")
        bows = [self.dictionary.doc2bow(x) for x in words]
        self.tfidf = TfidfModel(bows, dictionary=self.dictionary, normalize=True)
        p = self.parameters
        base = {k: p[k] for k in ["vector_size", "window", "min_count", "epochs", "negative", "hs", "sample", "alpha", "min_alpha", "workers", "batch_words"]}
        base["seed"] = self.seed
        if self.method == "lsi":
            topics = min(p["vector_size"], len(self.dictionary), max(1, len(docs) - 1))
            self.model = LsiModel(self.tfidf[bows], id2word=self.dictionary, num_topics=topics, random_seed=self.seed)
        elif self.method == "word2vec":
            self.model = Word2Vec(sentences=SentenceCorpus(docs), sg=p["sg"], **base)
        elif self.method == "fasttext":
            self.model = FastText(sentences=SentenceCorpus(docs), sg=p["sg"], min_n=p["min_n"], max_n=p["max_n"], bucket=p["bucket"], **base)
        elif self.method == "doc2vec":
            tagged = [TaggedDocument(d["tokens"], [d["id"]]) for d in docs]
            self.model = Doc2Vec(documents=tagged, dm=0, **base)
        elif self.method != "tfidf":
            raise ValueError(self.method)
        self.max_idf = max(self.tfidf.idfs.values(), default=1.0)
        self.fit_ids = [d["id"] for d in docs]
        return self

    def transform(self, docs):
        vocabulary = self.model.wv.key_to_index if self.method in ["word2vec", "fasttext", "doc2vec"] else self.dictionary.token2id
        total, oov, valid_oov, zero_docs = 0, 0, 0, 0
        vectors = []
        if self.method in ["tfidf", "lsi"]:
            corpus = []
            for d in docs:
                total += len(d["tokens"])
                oov += sum(t not in vocabulary for t in d["tokens"])
                v = self.tfidf[self.dictionary.doc2bow(d["tokens"])]
                if self.method == "lsi":
                    v = self.model[v]
                zero_docs += not bool(v)
                corpus.append(v)
            width = len(self.dictionary) if self.method == "tfidf" else self.model.num_topics
            matrix = corpus2csc(corpus, num_terms=width, num_docs=len(docs), dtype=np.float32).T.tocsr()
        elif self.method == "doc2vec":
            for d in docs:
                total += len(d["tokens"])
                oov += sum(t not in vocabulary for t in d["tokens"])
                # Reset RNG per document so inference is stable after save/load and independent of order.
                stable = int.from_bytes(hashlib.sha256((str(self.seed) + d["id"]).encode()).digest()[:4], "little")
                self.model.random.seed(stable)
                v = self.model.infer_vector(d["tokens"], alpha=0.025, min_alpha=0.0001, epochs=30)
                vectors.append(v)
                zero_docs += not bool(np.any(v))
            matrix = np.asarray(vectors, dtype=np.float32)
        else:
            for d in docs:
                total += len(d["tokens"])
                result = np.zeros(self.parameters["vector_size"], dtype=np.float32)
                weight_sum = 0.0
                counts = collections.Counter(d["tokens"])
                for word, count in counts.items():
                    unseen = word not in vocabulary
                    oov += count if unseen else 0
                    try:
                        v = self.model.wv.get_vector(word)
                        if not np.all(np.isfinite(v)) or not np.any(v):
                            continue
                    except KeyError:
                        continue
                    valid_oov += count if unseen else 0
                    weight = float(count)
                    if self.parameters["pooling"] == "tfidf":
                        token_id = self.dictionary.token2id.get(word)
                        weight *= self.tfidf.idfs.get(token_id, self.max_idf)
                    result += weight * v
                    weight_sum += weight
                if weight_sum:
                    result /= weight_sum
                else:
                    zero_docs += 1
                vectors.append(result)
            matrix = np.asarray(vectors, dtype=np.float32)
        matrix = normalize(matrix, norm="l2", copy=False)
        coverage = {"tokens": total, "oov_tokens": oov, "oov_rate": oov / total if total else 0.0, "oov_vector_coverage": valid_oov / oov if oov else 1.0, "zero_documents": int(zero_docs)}
        if not np.all(np.isfinite(matrix.data if sparse.issparse(matrix) else matrix)):
            raise ValueError("Non-finite feature matrix")
        return matrix, coverage


def metrics(truth, predictions, labels):
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(truth, predictions, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(),
        "n": len(truth),
    }


class Bundle:
    def __init__(self, spec):
        self.spec = spec
        self.method = spec["method"]
        self.labels = [f"{i:02d}" for i in range(1, spec["classes"] + 1)]
        self.representation = None
        self.classifier = None
        self.native = None

    def fit(self, train, validation, run_dir):
        y = [d["page_label"] for d in train]
        fit_start = time.perf_counter()
        tuning = []
        if self.method == "native_fasttext":
            path = run_dir / "native_train.txt"
            path.write_text("".join(f"__label__p{d['page_label']} {' '.join(d['tokens'])}\n" for d in train), encoding="utf-8")
            self.native = fasttext.train_supervised(input=str(path), seed=self.spec["seed"], verbose=0, **self.spec["native"])
        elif self.method == "majority":
            self.classifier = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), y)
        else:
            self.representation = Representation(self.method, self.spec["embedding"], self.spec["dictionary"], self.spec["seed"]).fit(train)
            x, _ = self.representation.transform(train)
            xv, _ = self.representation.transform(validation)
            best = None
            for c in self.spec["svm_c"]:
                clf = LinearSVC(C=c, class_weight="balanced", random_state=self.spec["seed"], dual="auto", max_iter=20000)
                clf.fit(x, y)
                score = metrics([d["page_label"] for d in validation], clf.predict(xv), self.labels)["macro_f1"]
                tuning.append({"C": c, "validation_macro_f1": score})
                if best is None or score > best[0]:
                    best = score, clf
            self.classifier = best[1]
        self.fit_seconds = time.perf_counter() - fit_start
        self.tuning = tuning
        return self

    def predict(self, docs):
        if self.method == "native_fasttext":
            predictions, scores = [], []
            words = set(self.native.get_words())
            total = sum(len(d["tokens"]) for d in docs)
            oov = sum(t not in words for d in docs for t in d["tokens"])
            for d in docs:
                labels, values = self.native.predict(" ".join(d["tokens"]), k=1)
                predictions.append(labels[0].removeprefix("__label__p"))
                scores.append(float(values[0]))
            coverage = {"tokens": total, "oov_tokens": oov, "oov_rate": oov / total if total else 0.0, "oov_vector_coverage": 0.0 if oov else 1.0, "zero_documents": sum(not any(t in words for t in d["tokens"]) for d in docs)}
        elif self.method == "majority":
            predictions = self.classifier.predict(np.zeros((len(docs), 1))).tolist()
            scores = [1.0] * len(docs)
            coverage = {}
        else:
            x, coverage = self.representation.transform(docs)
            predictions = self.classifier.predict(x).tolist()
            margins = self.classifier.decision_function(x)
            scores = np.abs(margins).tolist() if margins.ndim == 1 else margins.max(axis=1).tolist()
        return predictions, scores, coverage

    def save(self, directory):
        native = self.native
        if native is not None:
            native.save_model(str(directory / "native.bin"))
            self.native = None
        try:
            joblib.dump(self, directory / "bundle.joblib", compress=0)
        finally:
            self.native = native

    @classmethod
    def load(cls, directory):
        bundle = joblib.load(directory / "bundle.joblib")
        if bundle.method == "native_fasttext":
            bundle.native = fasttext.load_model(str(directory / "native.bin"))
        return bundle

