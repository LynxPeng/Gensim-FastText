import copy

import numpy as np

from gensim_lab.common import config
from gensim_lab.content_seed import ContentSeedRepresentation, inference_identity


def test_identity_is_token_based_without_concatenation_ambiguity():
    assert inference_identity(["中文", "新闻"]) == inference_identity(["中文", "新闻"])
    assert inference_identity(["ab", "c"]) != inference_identity(["a", "bc"])


def test_doc2vec_identical_tokens_ignore_id_and_order():
    cfg = config()
    parameters = copy.deepcopy(cfg["embedding"])
    parameters.update(vector_size=10, min_count=1, epochs=2)
    train = [{"id": str(i), "tokens": ["共同", "新闻", "经济" if i < 6 else "体育"] * 8} for i in range(12)]
    model = ContentSeedRepresentation("doc2vec", parameters, {"no_below": 1, "no_above": 1.0, "keep_n": 100}, 42).fit(train)
    docs = [{"id": "a", "tokens": ["经济", "新闻"]}, {"id": "b", "tokens": ["体育", "共同"]}]
    original = copy.deepcopy(docs)
    changed = [{**d, "id": "different" + d["id"]} for d in docs]
    x, _ = model.transform(docs)
    y, _ = model.transform(changed)
    z, _ = model.transform(list(reversed(docs)))
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(x, z[::-1])
    assert docs == original and model.fit_ids == [str(i) for i in range(12)]
    unseen, coverage = model.transform([{"id": "unknown", "tokens": ["验证专属未知词"]}])
    np.testing.assert_array_equal(unseen, np.zeros_like(unseen))
    assert coverage["zero_documents"] == 1 and coverage["oov_vector_coverage"] == 0.0
