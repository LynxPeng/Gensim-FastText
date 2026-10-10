import numpy as np
from gensim.models import KeyedVectors
from gensim_lab import pfr_topic_native as task
from gensim_lab.common import write_json, read_json
from gensim_lab.pfr_topics import TOPICS


def test_identical_predictions_have_zero_paired_difference():
    truth=[t for t in TOPICS for _ in range(3)]
    result=task.paired_bootstrap(truth,{m:truth for m in task.VARIANTS},TOPICS,20,42)
    assert result['difference']['estimate']==result['difference']['ci_low']==result['difference']['ci_high']==0


def test_saturated_native_score_is_a_valid_probability_distribution():
    class Model:
        def predict(self,text,k):
            return [f'__label__t{i}' for i in range(8)],np.asarray([1.00001]+[.00001]*7)
    themes,probability,raw=task.native_distribution(Model(),['比赛'],TOPICS)
    assert themes[0]==TOPICS[0] and raw[0]>1
    assert probability[0]==1 and probability.sum()==1
    assert np.all((probability>=0)&(probability<=1))


def test_actual_native_worker_imports_train_vectors_and_keeps_test_unopened(tmp_path,monkeypatch):
    monkeypatch.setattr(task,'ROOT',tmp_path)
    source=tmp_path/'source';source.mkdir()
    words=['共同']+[f'主题{i}' for i in range(8)]
    vectors=KeyedVectors(vector_size=100)
    vectors.add_vectors(words,np.random.default_rng(42).normal(size=(len(words),100)).astype(np.float32))
    vectors.save(str(source/'vectors.kv'))
    docs=[]
    for part in ['train','validation','test']:
        for j,topic in enumerate(TOPICS):
            key=f'{part}_{j}'
            docs.append({'id':key,'source_id':key,'topic':topic,'tokens':[f'主题{j}','共同']*20})
    splits={part:[d['id'] for d in docs if d['id'].startswith(part)] for part in ['train','validation','test']}
    cfg={'topics':TOPICS,'seeds':[42,7],'native':{'dim':100,'epochs':[1,2],'lr':.1,
         'minCount':1,'wordNgrams':2,'minn':0,'maxn':0,'loss':'softmax','bucket':50}}
    out=tmp_path/'experiment';out.mkdir()
    write_json(out/'plan.json',{'configuration':cfg,'labels':TOPICS,'splits':splits,
         'runtime':{'training_threads':4},'sources':{'fasttext':{'directory':'source'}}})
    write_json(out/'index.json',{'runs':{},'selection':{}})
    task.train(out,docs)
    index=read_json(out/'index.json')
    assert len(index['runs'])==6 and len(index['selection'])==4
    assert read_json(out/'native_input/initialization_probe.json')['lr_zero_import_exact']
    assert not list(out.glob('runs/*/metrics_test.json'))
    assert not (out/'selection_frozen.json').exists()
    text=(out/'native_input/train.txt').read_text(encoding='utf-8')
    assert '__label__t0' in text and '__label__p' not in text
