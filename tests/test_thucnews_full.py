"""Full-corpus completeness, fixed holdout, recovery and native checkpoint continuation."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import jieba
import numpy as np
from gensim.models import Word2Vec

from gensim_lab.common import ROOT, file_hash, write_json
from gensim_lab.thucnews import save_training_model
from gensim_lab.thucnews_data import clean, tokenize
from gensim_lab.thucnews_full_data import assign_partition


def test_reference_holdout_has_priority_over_earlier_duplicate():
    reference = {"id": "kept", "text_sha256": "same", "sentences": [["新闻"]]}
    assert assign_partition({**reference, "id": "copy"}, {"kept": reference}, {"same"}, set()) is None
    assert assign_partition(reference, {"kept": reference}, {"same"}, set()) == "holdout"
    assert assign_partition({"id": "other", "text_sha256": "different"}, {"kept": reference}, {"same"}, {"different"}) is None


def test_full_archive_is_exhausted_and_committed_preparation_can_resume(tmp_path, monkeypatch):
    import gensim_lab.thucnews_full_data as module
    raw = tmp_path / "raw"
    raw.mkdir()
    parameters = {"mode": "full", "documents": None, "seed": 42, "holdout_documents": 1,
                  "holdout_reference": "reference/manifest.json", "preprocessing_workers": 1,
                  "minimum_tokens": 2, "maximum_sentence_tokens": 256, "tokenizer_hmm": False}
    held_text = "中国 经济 新闻。"
    archive = raw / "THUCNews.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name, text in [("1.txt", held_text), ("2.txt", "全球 科技 市场。"),
                           ("3.txt", held_text), ("4.txt", "！"), ("5.txt", b"\xff")]:
            stream.writestr("THUCNews/体育/" + name, text)
    tokenizer = jieba.Tokenizer()
    tokenizer.tmp_dir = str(tmp_path)
    sentences = tokenize(clean(held_text), tokenizer, parameters)
    member = "THUCNews/体育/3.txt"
    identifier = hashlib.sha256(member.encode()).hexdigest()[:24]
    document = {"id": identifier, "text_sha256": hashlib.sha256(clean(held_text).encode()).hexdigest(),
                "part": "holdout", "sentences": sentences}
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "documents.jsonl").write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
    (reference / "holdout.txt").write_text("".join(" ".join(s) + "\n" for s in sentences), encoding="utf-8", newline="\n")
    with (reference / "holdout_sentences.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["document_id", "sentence_index", "tokens"])
        writer.writerows([identifier, index, len(s)] for index, s in enumerate(sentences))
    source = {"sha256": file_hash(archive), "article_files": 5, "categories": {"体育": 5}}
    write_json(reference / "manifest.json", {"directory": "reference", "source": source,
               "spec": {"jieba_dictionary_sha256": file_hash(Path(jieba.__file__).parent / "dict.txt")},
               "files": {name: file_hash(reference / name) for name in ["documents.jsonl", "holdout.txt", "holdout_sentences.csv"]}})
    code_dir = tmp_path / "gensim_lab"
    code_dir.mkdir()
    (code_dir / "thucnews_data.py").write_bytes((ROOT / "gensim_lab/thucnews_data.py").read_bytes())
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RAW", raw)
    monkeypatch.setattr(module, "PREPARED", tmp_path / "prepared")
    first = module.prepare_full(source, parameters)
    assert first["processed_articles"] == 5 and first["eligible_articles"] == 2
    assert first["excluded"] == {"duplicate_clean_text": 1, "short_document": 1, "invalid_utf8": 1}
    assert first["partitions"]["train"]["documents"] == first["partitions"]["holdout"]["documents"] == 1
    directory = tmp_path / first["directory"]
    (directory / "manifest.json").unlink()
    with (directory / "documents.unsorted.jsonl").open("ab") as stream:
        stream.write(b"uncommitted invalid tail")
    with (directory / "exclusions.csv").open("ab") as stream:
        stream.write(b"uncommitted invalid tail")
    recovered = module.prepare_full(source, parameters)
    assert recovered["fingerprint"] == first["fingerprint"]




def test_native_checkpoint_continuation_matches_uninterrupted_epochs(tmp_path):
    sentences = [["新闻", "经济", "市场"], ["中国", "科技", "新闻"]] * 100
    parameters = dict(vector_size=8, min_count=1, window=2, sg=0, hs=1, negative=0,
                      workers=1, batch_words=20, seed=42, sample=0)
    continuous, resumed = Word2Vec(**parameters), Word2Vec(**parameters)
    for model in [continuous, resumed]:
        model.build_vocab(sentences)
        model.train(sentences, total_examples=len(sentences), epochs=1,
                    start_alpha=.025, end_alpha=.0125)
    checkpoint = tmp_path / "checkpoint.gensim"
    save_training_model(resumed, checkpoint)
    resumed = Word2Vec.load(str(checkpoint))
    resumed.__class__ = Word2Vec
    for model in [continuous, resumed]:
        model.train(sentences, total_examples=len(sentences), epochs=1,
                    start_alpha=.0125, end_alpha=.0001)
    assert np.array_equal(continuous.wv.vectors, resumed.wv.vectors)
    assert np.array_equal(continuous.syn1, resumed.syn1)
