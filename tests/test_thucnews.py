"""Data-source and partition invariants for the independent news experiment."""
import zipfile
import pytest
from gensim_lab.thucnews_data import article_category, clean, decoded_name, quotas, tokenize


def test_unflagged_utf8_zip_category_is_recovered():
    info = zipfile.ZipInfo("THUCNews/" + "体育".encode().decode("cp437") + "/1.txt")
    assert decoded_name(info) == "THUCNews/体育/1.txt"
    assert article_category(info) == "体育"


def test_appledouble_is_not_an_article():
    assert article_category(zipfile.ZipInfo("__MACOSX/THUCNews/体育/._1.txt")) is None


def test_archive_traversal_is_rejected():
    with pytest.raises(ValueError, match="Unsafe"):
        article_category(zipfile.ZipInfo("../THUCNews/体育/1.txt"))


def test_sampling_and_holdout_totals_are_exact():
    categories = {"a": 101, "b": 53, "c": 3}
    sampled = quotas(categories, 50)
    heldout = quotas(sampled, 5)
    assert sum(sampled.values()) == 50 and sum(heldout.values()) == 5
    assert all(0 <= heldout[k] <= sampled[k] <= categories[k] for k in categories)


def test_control_characters_and_fullwidth_text_have_one_hashable_form():
    assert clean("ＡＢＣ\x00\n  新闻　报道  ") == "ABC\n新闻 报道"


def test_sentence_context_does_not_cross_boundaries_or_chunk_limit():
    class SplitTokenizer:
        def cut(self, text, HMM):
            return text.replace("。", "").split()
    sentences = tokenize("中国 经济 新闻。美国 科技 新闻", SplitTokenizer(),
                         {"tokenizer_hmm": False, "maximum_sentence_tokens": 2})
    assert sentences == [["中国", "经济"], ["新闻"], ["美国", "科技"], ["新闻"]]
