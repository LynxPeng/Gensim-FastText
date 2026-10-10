import numpy as np
import pytest
from gensim.models import Word2Vec
from gensim_lab.word_evaluation import correlation_status, evaluate_pairs, evaluate_analogies
from gensim_lab.evaluation_support import score_event_count, stable_hs_score


def toy(hs=1, negative=0):
    return Word2Vec([['中国','北京','经济','上海'], ['日本','东京','经济','中国']]*20,
                    vector_size=8, min_count=1, workers=1, seed=42, sg=1,
                    hs=hs, negative=negative, epochs=2)




def test_pair_oov_and_constant_status(tmp_path):
    m=toy(); p=tmp_path/'pairs.tsv'
    p.write_text('中国\t北京\t9\n日本\t东京\t8\n中国\t经济\t4\n不存在\t北京\t1\n',encoding='utf-8')
    r=evaluate_pairs(m.wv,p,tmp_path,'test',{'中国','北京'})
    assert r['total']==4 and r['covered']==3 and r['oov_ratio_percent']==25
    assert r['common_n']==1 and r['common_spearman_rho'] is None
    assert correlation_status([1,1,1],[1,2,3])=='constant_values'
    assert correlation_status([1,2],[1,2])=='insufficient_pairs'


def test_analogy_full_denominator(tmp_path):
    m=toy(); p=tmp_path/'analogy.txt'
    p.write_text(': cities\n中国 北京 日本 东京\n中国 北京 不存在 东京\n: unknown\n不存在 北京 日本 东京\n',encoding='utf-8')
    r=evaluate_analogies(m.wv,p,tmp_path,'test')
    assert r['total']==3 and r['covered']==1
    assert r['all_accuracy']==r['correct']/3
    assert r['sections'][1]['conditional_accuracy'] is None


def test_stable_hs_no_context_and_guard():
    m=toy()
    assert stable_hs_score(m,['不存在','中国'])==(0.0,0,0)
    assert score_event_count(4,2,1)==10
    assert score_event_count(4,2,0)==4
    value,sat,nodes=stable_hs_score(m,['中国','北京','经济'])
    assert value<0 and np.isfinite(value) and nodes>0
    approx=float(m.score([['中国','北京','经济']],total_sentences=1)[0])
    assert abs(approx-value)<0.3
    with pytest.raises(ValueError,match='HS-only'):
        stable_hs_score(toy(0,5),['中国','北京'])




def test_saved_score_decimal_round_trip(tmp_path):
    from gensim_lab.word_checks import numeric_csv
    score=float(np.float32(-31.336872100830078))
    path=tmp_path/'score.csv'
    path.write_text(f'builtin_log_score\n{score!r}\n',encoding='utf-8')
    assert numeric_csv(path).iloc[0].builtin_log_score==score
