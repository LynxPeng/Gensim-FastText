"""Check the completed fresh baseline against the historical parameter settings."""
from gensim.models import Word2Vec

from gensim_lab.common import ROOT, file_hash, read_json, stamp, write_json


if __name__ == '__main__':
    base = ROOT / 'artifacts/thucnews_ablation'
    experiment = ROOT / read_json(base / 'latest.json')['experiment']
    plan = read_json(experiment / 'index.json')
    row = next(r for r in plan['runs'] if r['method'] == 'word2vec' and r['combination'] == 'cbow_hs' and r['model_seed'] == 42)
    directory = ROOT / row['run']
    previous = ROOT / read_json(ROOT / 'artifacts/thucnews_validation/latest.json')['models']['word2vec']
    old, new = Word2Vec.load(str(previous / 'model.gensim')), Word2Vec.load(str(directory / row['checkpoint']))
    fields = ['vector_size', 'window', 'min_count', 'sg', 'hs', 'negative', 'sample', 'workers',
              'batch_words', 'ns_exponent', 'cbow_mean', 'shrink_windows', 'sorted_vocab', 'seed']
    settings = {k: {'previous': getattr(old, k), 'new': getattr(new, k)} for k in fields}
    assert all(v['previous'] == v['new'] for v in settings.values())
    assert old.corpus_count == new.corpus_count == 18576718
    assert old.corpus_total_words == new.corpus_total_words == 378641761
    assert old.wv.index_to_key == new.wv.index_to_key
    assert all(old.wv.get_vecattr(w, 'count') == new.wv.get_vecattr(w, 'count') for w in old.wv.index_to_key)
    assert not (directory / 'test').exists() and read_json(directory / 'summary.json')['test_evaluated'] is False
    write_json(base / 'baseline_parameter_check.json', {'status': 'passed', 'settings': settings,
        'same_training_corpus_counts_and_vocabulary_frequencies': True, 'search_has_no_test_results': True,
        'baseline_run': row['run'], 'baseline_seal_sha256': file_hash(directory / 'seal.json'),
        'interpretation': 'Multithreaded fresh retraining can change validation scores despite identical settings.',
        'checked': stamp()})
    print('Fresh baseline parameters, full corpus counts and vocabulary frequencies match the historical baseline.')
