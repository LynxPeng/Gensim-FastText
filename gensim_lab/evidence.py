from __future__ import annotations

import collections
import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import fasttext
import numpy as np
from gensim.models import FastText
from gensim.models.fasttext_inner import compute_ngrams_bytes, ft_hash_bytes

from .common import ARTIFACTS, REPORTS, ROOT, code_hash, environment, file_hash, markdown_table, read_json, stamp, write_json
from .runner import index, worker_environment
from .runtime import training_threads


class CountingCorpus:
    def __init__(self, rows):
        self.rows = rows
        self.iterations = 0
        self.yielded = 0

    def __iter__(self):
        self.iterations += 1
        for row in self.rows:
            self.yielded += 1
            yield row


class ObservedFastText(FastText):
    def __init__(self, *args, **kwargs):
        self.observed_jobs = []
        super().__init__(*args, **kwargs)

    def _do_train_job(self, sentences, alpha, inits):
        self.observed_jobs.append({"sentences": len(sentences), "raw_words": sum(map(len, sentences)), "alpha": float(alpha)})
        return super()._do_train_job(sentences, alpha, inits)


def compatibility_hash(value):
    h = 2166136261
    for b in value:
        signed = b if b < 128 else b - 256
        h = ((h ^ (signed & 0xffffffff)) * 16777619) & 0xffffffff
    return h


def probes(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows = [["共同", "新闻", "经济", "市场", "共同"] for _ in range(200)] + [["共同", "新闻", "体育", "比赛", "共同"] for _ in range(200)] + [["稀有词"]]
    result = {"created": stamp(), "environment": environment(), "subword_examples": {}}
    for word in ["测试", "where", "共同新词"]:
        grams = compute_ngrams_bytes(word, 1, 3)
        hashes = [int(ft_hash_bytes(g)) for g in grams]
        expected = [compatibility_hash(g) for g in grams]
        assert hashes == expected
        result["subword_examples"][word] = {"ngrams": [g.decode("utf-8") for g in grams], "hashes": hashes, "signed_byte_hash_match": True}
    for sample in [0.0, 0.001]:
        corpus = CountingCorpus(rows)
        model = ObservedFastText(vector_size=20, min_count=3, min_n=1, max_n=3, bucket=1000, workers=training_threads(), seed=42, epochs=3, sg=1, sample=sample, batch_words=100)
        model.build_vocab(corpus)
        outcome = model.train(corpus, total_examples=model.corpus_count, epochs=3)
        assert "稀有词" not in model.wv.key_to_index
        probabilities = {w: int(model.wv.get_vecattr(w, "sample_int")) / (2**32 - 1) for w in model.wv.index_to_key}
        result[f"sampling_{sample}"] = {"iterations": corpus.iterations, "yielded_sentences": corpus.yielded,
            "vocabulary": model.wv.index_to_key, "keep_probabilities": probabilities,
            "effective_words": int(outcome[0]), "raw_words": int(outcome[1]),
            "job_count": len(model.observed_jobs), "jobs": model.observed_jobs,
            "alpha_start": model.observed_jobs[0]["alpha"], "alpha_last_job": model.observed_jobs[-1]["alpha"]}
        if sample == 0.001:
            wv = model.wv
            known = wv.index_to_key[0]
            i = wv.key_to_index[known]
            ids = list(wv.buckets_word[i])
            manual = (wv.vectors_vocab[i] + wv.vectors_ngrams[ids].sum(axis=0)) / (len(ids) + 1)
            oov = "共同新词"
            oids = [int(ft_hash_bytes(g)) % wv.bucket for g in compute_ngrams_bytes(oov, 1, 3)]
            manual_oov = wv.vectors_ngrams[oids].mean(axis=0)
            result["aggregation"] = {"iv_word": known, "iv_max_error": float(np.max(np.abs(wv.get_vector(known) - manual))),
                                     "oov_word": oov, "oov_max_error": float(np.max(np.abs(wv.get_vector(oov) - manual_oov))),
                                     "oov_norm": float(np.linalg.norm(wv.get_vector(oov)))}
            assert np.allclose(wv.get_vector(known), manual, atol=1e-6)
            assert np.allclose(wv.get_vector(oov), manual_oov, atol=1e-6)
            result["memory"] = {n: {"shape": list(getattr(wv, n).shape), "dtype": str(getattr(wv, n).dtype),
                                    "bytes": getattr(wv, n).nbytes, "c_contiguous": bool(getattr(wv, n).flags.c_contiguous)} for n in ["vectors", "vectors_vocab", "vectors_ngrams"]}
            # Use Gensim's own model serialization in addition to the experiment bundle checks.
            plain = FastText(vector_size=20, min_count=1, min_n=1, max_n=3, bucket=1000, workers=training_threads(), seed=42, epochs=2, sentences=rows)
            plain.save(str(directory / "gensim.model"))
            restored = FastText.load(str(directory / "gensim.model"))
            diff = float(np.max(np.abs(plain.wv.get_vector(oov) - restored.wv.get_vector(oov))))
            assert diff == 0
            result["gensim_native_save_load"] = {"oov_max_error": diff}
    native_input = directory / "native_probe.txt"
    native_input.write_text("".join(f"__label__p{'01' if i < 200 else '02'} {' '.join(row)}\n" for i, row in enumerate(rows[:400])), encoding="utf-8")
    native = fasttext.train_supervised(input=str(native_input), dim=20, epoch=25, lr=0.1, minCount=3, wordNgrams=2, minn=0, maxn=0, bucket=1000, thread=training_threads(), seed=42, verbose=0)
    sentences = ["经济 市场", "体育 比赛", "共同 新闻 经济 市场", "共同 新闻 体育 比赛"]
    before, probabilities = native.predict(sentences)
    native.save_model(str(directory / "native_probe.bin"))
    restored = fasttext.load_model(str(directory / "native_probe.bin"))
    after, other = restored.predict(sentences)
    assert before == after and np.array_equal(probabilities, other)
    matrix = native.get_input_matrix()
    args = native.f.getArgs()
    result["native"] = {"predictions": before, "probabilities": np.asarray(probabilities).tolist(), "training_test": native.test(str(native_input)),
                        "input_matrix_shape": list(matrix.shape), "output_matrix_shape": list(native.get_output_matrix().shape), "dtype": str(matrix.dtype),
                        "parameters": {n: getattr(args, n) for n in ["dim", "epoch", "wordNgrams", "minn", "maxn", "bucket", "thread", "seed", "minCount"]}, "save_load_identical": True}
    assert native.test(str(native_input))[1] >= 0.99
    write_json(directory / "probes.json", result)
    return result


def reproduce():
    """Retrain all seven January K=12 seed=42 methods in independent processes."""
    results = []
    for key, rec in index()["jobs"].items():
        if rec["classes"] != 12 or rec["seed"] != 42 or not any(u["profile"] == "main" for u in rec["uses"]):
            continue
        original = ARTIFACTS / "runs" / key
        spec = read_json(original / "spec.json")
        from .integrity import spec_is_current, cache_problem, seal_run
        if not spec_is_current(spec):
            continue
        directory = ARTIFACTS / "reproducibility" / key
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "spec.json", spec)
        if cache_problem(directory, spec) is not None:
            print(f"REPRODUCE {rec['method']} {key}", flush=True)
            module = "gensim_lab.supplemental_worker" if "doc2vec_inference_seed" in spec else "gensim_lab.worker"
            run = subprocess.run([sys.executable, "-m", module, str(directory)], env=worker_environment(), cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            (directory / "console.log").write_text(run.stdout + run.stderr, encoding="utf-8")
            if run.returncode:
                raise RuntimeError(run.stderr)
            seal_run(directory, spec)
        first = file_hash(original / "predictions_validation.csv")
        second = file_hash(directory / "predictions_validation.csv")
        deterministic = spec['embedding']['workers']==1 and spec['native']['thread']==1
        values = {"method": rec["method"], "run": key, "prediction_csv_sha256": first, "retrained_sha256": second,
                  "identical": first == second, "bitwise_equality_required": deterministic}
        results.append(values)
    assert results and all(r['identical'] or not r['bitwise_equality_required'] for r in results), results
    write_json(ARTIFACTS / "reproducibility" / "checks.json", results)
    return results


def source_index():
    manifest = read_json(ROOT / "references" / "source_manifest.json")
    needles = [
        ("字符子词", "models/fasttext_inner.pyx", "cpdef compute_ngrams_bytes"),
        ("兼容 FNV 哈希", "models/fasttext_inner.pyx", "cpdef ft_hash_bytes"),
        ("OOV 向量", "models/fasttext.py", "def get_vector(self, word"),
        ("词内向量聚合", "models/fasttext.py", "def adjust_vectors"),
        ("子词桶缓存", "models/fasttext.py", "def recalc_char_ngram_buckets"),
        ("训练 Cython 内核", "models/fasttext_inner.pyx", "def train_batch_any"),
        ("负采样", "models/fasttext_inner.pyx", "cdef void fasttext_fast_sentence_sg_neg"),
        ("层次 softmax", "models/fasttext_inner.pyx", "cdef void fasttext_fast_sentence_sg_hs"),
        ("词表过滤与下采样", "models/word2vec.py", "def prepare_vocab"),
        ("负采样分布", "models/word2vec.py", "def make_cum_table"),
        ("工作线程", "models/word2vec.py", "def _worker_loop(self"),
        ("任务生产与批次", "models/word2vec.py", "def _job_producer"),
        ("学习率", "models/word2vec.py", "def _get_next_alpha"),
        ("保存模型", "models/fasttext.py", "def save(self"),
        ("监督分类特征", "src/dictionary.cc", "void Dictionary::addWordNgrams"),
        ("原生子词", "src/dictionary.cc", "void Dictionary::computeSubwords"),
        ("原生哈希兼容", "src/dictionary.cc", "uint32_t Dictionary::hash"),
        ("监督模式下采样", "src/dictionary.cc", "bool Dictionary::discard"),
        ("特征平均", "src/model.cc", "void Model::computeHidden"),
        ("监督目标", "src/fasttext.cc", "void FastText::supervised"),
        ("原生线程及学习率", "src/fasttext.cc", "void FastText::trainThread"),
        ("原生目标分派", "src/fasttext.cc", "std::shared_ptr<Loss> FastText::createLoss"),
        ("Python 入口", "python/fasttext_module/fasttext/FastText.py", "def train_supervised"),
        ("pybind 与 GIL", "python/fasttext_module/fasttext/pybind/fasttext_pybind.cc", "gil_scoped_release"),
        ("原生保存", "src/fasttext.cc", "void FastText::saveModel"),
        ("连续内存", "src/densematrix.cc", "DenseMatrix::DenseMatrix"),
    ]
    rows, output = [], []
    for concept, suffix, needle in needles:
        source = next(s for s in manifest["sources"] if s["path"].endswith(suffix))
        lines = (ROOT / source["path"]).read_text(encoding="utf-8").splitlines()
        line = next(i + 1 for i, text in enumerate(lines) if needle in text)
        url = source["url"]
        if "raw.githubusercontent.com/facebookresearch/fastText/" in url:
            url = url.replace("raw.githubusercontent.com", "github.com").replace("/fastText/v0.9.2/", "/fastText/blob/v0.9.2/")
        url += f"#L{line}"
        output.append({"concept": concept, "path": source["path"], "line": line, "symbol": needle, "url": url})
        rows.append([concept, f"[{suffix}:{line}]({url})", f"[本地副本](../{source['path']})"])
    write_json(ROOT / "references" / "source_positions.json", output)
    (REPORTS / "源码位置索引.md").write_text("# 固定版本源码位置索引\n\n位置从保存的源码实际扫描得到，Gensim 4.4.0 / FastText v0.9.2。文件 SHA-256 和下载来源见[源码清单](../references/source_manifest.json)。\n\n" + markdown_table(["概念", "版本固定的源码定位", "归档"], rows) + "\n", encoding="utf-8")


def run_evidence():
    REPORTS.mkdir(exist_ok=True)
    source_index()
    p = probes(ARTIFACTS / "evidence")
    repeated = reproduce()
    md = "# 运行核验证据\n\n本文件由真实探针和独立进程重训结果生成。完整数值见[探针 JSON](../artifacts/evidence/probes.json)与[重训校验](../artifacts/reproducibility/checks.json)。\n\n"
    md += "## 子词与哈希\n\n" + markdown_table(["词", "1–3 子词", "兼容哈希匹配"], [[w, "、".join(v["ngrams"]), v["signed_byte_hash_match"]] for w, v in p["subword_examples"].items()])
    md += f"\n\n词内聚合最大误差 {p['aggregation']['iv_max_error']:.3g}；OOV 聚合最大误差 {p['aggregation']['oov_max_error']:.3g}；OOV 范数 {p['aggregation']['oov_norm']:.6f}。有向量不代表该 OOV 语义已正确学习；从未训练过的桶也含初始化向量。\n\n"
    md += "## 过滤、下采样、迭代与批次\n\n" + markdown_table(["sample", "词表大小", "遍历次数", "原始训练词", "有效训练词", "任务数", "首/末任务学习率"], [[s, len(p[f"sampling_{s}"]["vocabulary"]), p[f"sampling_{s}"]["iterations"], p[f"sampling_{s}"]["raw_words"], p[f"sampling_{s}"]["effective_words"], p[f"sampling_{s}"]["job_count"], f"{p[f'sampling_{s}']['alpha_start']:.6f} / {p[f'sampling_{s}']['alpha_last_job']:.6f}"] for s in [0.0, 0.001]])
    md += "\n\n最低词频 3 已过滤仅出现一次的“稀有词”。小语料使用可重复迭代对象，建词表与每个 epoch 能重新遍历；每个实际任务的词数和学习率保存在探针 JSON。`batch_words` 控制调度任务大小，学习率按任务更新，末任务学习率不一定恰好等于 min_alpha。\n\n"
    md += "## 内存与原生分类\n\n" + markdown_table(["数组", "形状", "类型", "字节", "连续"], [[n, v["shape"], v["dtype"], v["bytes"], v["c_contiguous"]] for n, v in p["memory"].items()])
    md += f"\n\n原生输入矩阵 {p['native']['input_matrix_shape']}、输出矩阵 {p['native']['output_matrix_shape']}；小样本训练集 P@1={p['native']['training_test'][1]:.4f}，保存加载预测完全一致。正式主矩阵分数低不能自动归因于接口失效，原生训练链路在可分的小语料上通过核验。\n\n"
    md += "## 独立进程重训\n\n一月 12 类、seed 42 的七种方法全部重新训练，未复用原训练模型。比较完整验证预测 CSV，包含标签及预测分数。\n\n" + markdown_table(["方法", "原运行", "CSV 字节一致"], [[r["method"], r["run"], r["identical"]] for r in repeated])
    md += "\n\n历史单线程参照要求重训预测字节一致；多线程训练记录差异，不要求独立重训逐位一致。保存加载仍核验预测与分数。当前 Doc2Vec 使用 tokens_sha256_v1 按词序列重置推断随机状态，编号和输入顺序不影响同一文本向量。修正前的编号依赖证据及重训结果见[补充报告](补充实验与改进报告.md)。\n"
    (REPORTS / "运行核验证据.md").write_text(md, encoding="utf-8")
    print("Evidence probes and independent retraining passed", flush=True)
