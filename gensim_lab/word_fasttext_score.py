"""Audit the inherited FastText HS score API without interpreting it as a language model."""
from pathlib import Path
import numpy as np
from gensim.models import Word2Vec, FastText
from .common import ROOT, ARTIFACTS, file_hash, read_json, stamp, write_json
from .models import Bundle
from .data import load_docs
from .reporting import records
from .word_evaluation import BASE, cache_valid, seal
from .evaluation_support import score_sentences
from .integrity import artifact_lock

def run():
    # Check the inherited scoring interface independently of model training.
    sentences=[['中国','北京','经济','人民'],['日本','东京','科学','中国']]*40
    toy=FastText(sentences,vector_size=8,min_count=1,bucket=1000,min_n=1,max_n=3,
                 workers=__import__('gensim_lab.runtime',fromlist=['training_threads']).training_threads(),hs=1,negative=0,seed=42,epochs=2)
    before=toy.wv.vectors.copy()
    toy.train(sentences,total_examples=toy.corpus_count,epochs=2)
    assert not np.array_equal(before,toy.wv.vectors)
    write_json(BASE/'fasttext_inherited_api_probe.json',{
        'score_inherited_from_word2vec':FastText.score is Word2Vec.score,
        'hs_score':float(toy.score([sentences[0]],total_sentences=1)[0]),
        'training_changed_vectors':not np.array_equal(before,toy.wv.vectors),
        'note':'HS inherited score is callable and remains a compatibility diagnostic.'})
    record=next(r for r in records() if r['method']=='fasttext' and any(
        u.get('axis')=='objective' and u.get('value')=='hs' for u in r['uses']))
    source=record['directory']; out=BASE/'fasttext_hs_score'/record['run']
    spec={'model_sha256':file_hash(source/'bundle.joblib'), 'source_run':record['run'],
          'protocol_sha256':file_hash(Path(__file__)), 'helper_sha256':file_hash(ROOT/'gensim_lab/evaluation_support.py')}
    out.mkdir(parents=True,exist_ok=True)
    if not cache_valid(out,spec):
        bundle=Bundle.load(source); model=bundle.representation.model
        assert model.hs==1 and model.negative==0
        # score is inherited verbatim, operating on the final averaged word vectors and HS head.
        proxy=Word2Vec.__new__(Word2Vec)
        proxy.__dict__.update(model.__dict__)
        scoring={}
        for part,ids in record['spec']['splits'].items():
            docs=load_docs(ids)
            sample=[s for d in docs for s in d['sentences']][:20]
            assert np.array_equal(model.score(sample,total_sentences=len(sample)),
                                  proxy.score(sample,total_sentences=len(sample)))
            scoring[part]=score_sentences(proxy,docs,part,out)
        write_json(out/'summary.json',{'completed':stamp(),'spec':spec,'scoring':scoring,
                   'scope':'Inherited Word2Vec HS scoring kernel; skips OOV despite subword generation; not sentence-generation probability.'})
        seal(out,spec)
    summary=read_json(out/'summary.json')
    write_json(BASE/'fasttext_hs_score.json',{'status':'complete','directory':str(out.relative_to(ROOT)),
                                          'source_run':record['run'],'summary':summary})
    print('FastText inherited HS score saved for all train/validation/test sentences',flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    from .runtime import numerical_threads
    with artifact_lock(),threadpool_limits(limits=numerical_threads()):
        run()
