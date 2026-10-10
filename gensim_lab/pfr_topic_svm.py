"""Three immutable representations and one LinearSVC on sealed PFR topics."""
from __future__ import annotations

import argparse
import collections
import copy
import gc
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import yaml

from .common import ROOT,PeakMemory,environment,file_hash,fingerprint,read_json,stamp,write_json
from .evaluation_support import save_csv
from .integrity import artifact_lock
from .pfr_frozen_classification import FrozenRepresentation,metrics,validate_partitions,cached_run,seal_run,paired_bootstrap
from .pfr_topics import load_jsonl
from .runtime import settings

CONFIG=ROOT/'configs/pfr_topic_svm.yaml'
BASE=ROOT/'artifacts/pfr_topic_svm'
METHODS=['tfidf','word2vec','fasttext']


def prepare():
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    topic=ROOT/cfg['topic_experiment'];original=read_json(topic/'plan.json')
    assert read_json(topic/'index.json')['status']=='complete'
    assert read_json(topic/'delivery_verification.json')['status']=='passed'
    corpus=ROOT/original['dataset'];seal=original['annotation_seal']
    for name,key in [('articles.jsonl','articles_sha256'),('annotations.jsonl','annotations_sha256'),('requests.jsonl','requests_sha256')]:
        assert file_hash(corpus/name)==seal[key]
    docs=load_jsonl(topic/'labelled_articles.jsonl')
    assert [d['id'] for d in docs]==original['document_order']
    annotations={a['id']:a for a in load_jsonl(corpus/'annotations.jsonl')}
    assert all(d['topic']==annotations[d['id']]['topic'] and annotations[d['id']]['status']!='out_of_scope' for d in docs)
    assert 'page_label' not in set().union(*(d.keys() for d in docs))
    validate_partitions(original['splits'],{d['id']:d for d in docs})
    sources={}
    for method,relative in cfg['sources'].items():
        folder=ROOT/relative;source_seal=read_json(folder/'seal.json');summary=read_json(folder/'summary.json')
        assert summary['spec']['method']==method and not summary['spec']['smoke']
        names=['model.gensim','summary.json']+(['dictionary.gensim'] if method=='tfidf' else ['vectors.kv'])
        hashes={n:file_hash(folder/n) for n in names}
        assert all(source_seal['files'][n]==h for n,h in hashes.items())
        sources[method]={'directory':relative,'files':hashes,'spec':summary['spec']}
    protocol=['pfr_topic_svm.py','pfr_frozen_classification.py','common.py','runtime.py']
    plan={'configuration':cfg,'configuration_sha256':file_hash(CONFIG),'runtime':settings(),
          'topic_experiment':cfg['topic_experiment'],'topic_plan_sha256':file_hash(topic/'plan.json'),
          'article_file':str((topic/'labelled_articles.jsonl').relative_to(ROOT).as_posix()),
          'article_sha256':file_hash(topic/'labelled_articles.jsonl'),
          'annotation_seal':seal,'sources':sources,'labels':original['labels'],
          'splits':copy.deepcopy(original['splits']),'document_order':original['document_order'],
          'protocol':{n:file_hash(ROOT/'gensim_lab'/n) for n in protocol},
          'environment':environment(),'package_versions':environment()['packages'],
          'scope':'TF-IDF/Word2Vec/Gensim FastText, frozen; same LinearSVC only'}
    identity={k:v for k,v in plan.items() if k!='environment'}
    directory=BASE/'experiments'/fingerprint(identity)[:24];directory.mkdir(parents=True,exist_ok=True)
    if (directory/'plan.json').exists():
        assert {k:v for k,v in read_json(directory/'plan.json').items() if k!='environment'}==identity
    else:
        write_json(directory/'plan.json',plan)
    if not (directory/'index.json').exists():
        write_json(directory/'index.json',{'status':'prepared','started':stamp(),'runs':{},'selection':{}})
    write_json(BASE/'latest.json',{'experiment':directory.relative_to(ROOT).as_posix()})
    print('Reuse sealed topic partitions:',{p:len(v) for p,v in plan['splits'].items()},flush=True)
    return directory,docs


def features(directory,docs):
    plan=read_json(directory/'plan.json');folder=directory/'features';folder.mkdir(exist_ok=True)
    if (folder/'seal.json').exists():
        for n,h in read_json(folder/'seal.json').items():
            assert file_hash(folder/n)==h
        return
    rows={};names=[]
    for method in METHODS:
        print('Frozen topic features:',method,flush=True)
        start=time.perf_counter()
        with PeakMemory() as memory:
            representation=FrozenRepresentation(method,plan['sources'][method])
            x,coverage=representation.transform(docs,plan['runtime']['preprocessing_workers'])
            x=normalize(x,norm='l2',copy=False)
            assert np.isfinite(x.data if sparse.issparse(x) else x).all()
            norms=np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel()) if sparse.issparse(x) else np.linalg.norm(x,axis=1)
            assert np.allclose(norms[norms>0],1,atol=1e-5)
            suffix='.npz' if sparse.issparse(x) else '.npy';name=method+suffix
            if sparse.issparse(x):sparse.save_npz(folder/name,x)
            else:np.save(folder/name,x)
            save_csv(folder/f'{method}_coverage.csv',coverage)
            rows[method]={'width':x.shape[1],'articles':len(docs),'seconds':time.perf_counter()-start,
                 'peak_rss_mib':memory.peak/2**20,'feature_bytes':(folder/name).stat().st_size,
                 'representation_vector_bytes':(ROOT/plan['sources'][method]['directory']/
                      ('dictionary.gensim' if method=='tfidf' else 'vectors.kv')).stat().st_size,
                 'frozen':True}
            names.extend([name,f'{method}_coverage.csv'])
            del representation,x;gc.collect()
    write_json(folder/'summary.json',rows);names.append('summary.json')
    write_json(folder/'seal.json',{n:file_hash(folder/n) for n in names})


def load_features(directory,method):
    folder=directory/'features'
    return sparse.load_npz(folder/'tfidf.npz') if method=='tfidf' else np.load(folder/f'{method}.npy',mmap_mode='r')


def evaluate(out,part,ids,truth,predicted,decision,labels):
    assert len(ids)==len(truth)==len(predicted)==len(decision) and np.isfinite(decision).all()
    result=metrics(truth,predicted,labels)
    write_json(out/f'metrics_{part}.json',result)
    classes=read_json(out/'class_order.json')['classes']
    assert set(classes)==set(labels)
    rows=[]
    for i,y,p,score in zip(ids,truth,predicted,decision):
        rows.append({'id':i,'true_topic':y,'predicted_topic':p,'score':float(score[classes.index(p)]),
                     'score_kind':'decision_function',**{f'decision_{label}':float(s) for label,s in zip(classes,score)}})
    save_csv(out/f'predictions_{part}.csv',rows)
    save_csv(out/f'per_class_{part}.csv',[{'topic':label,**result['per_class'][label]} for label in labels])
    save_csv(out/f'confusion_{part}.csv',[{'true_topic':label,**dict(zip(labels,row))} for label,row in zip(labels,result['confusion_matrix'])])
    return result


def train(directory,docs):
    plan=read_json(directory/'plan.json');cfg=plan['configuration'];options=cfg['svm']
    positions={i:n for n,i in enumerate(plan['document_order'])};lookup={d['id']:d for d in docs}
    train_ids=plan['splits']['train'];val_ids=plan['splits']['validation']
    y=[lookup[i]['topic'] for i in train_ids];yv=[lookup[i]['topic'] for i in val_ids]
    for method in METHODS:
        all_x=load_features(directory,method)
        x=all_x[[positions[i] for i in train_ids]];xv=all_x[[positions[i] for i in val_ids]]
        candidates=[]
        for c in options['C']:
            params={k:v for k,v in options.items() if k!='C'};params['C']=c
            spec={'method':method,'head':'LinearSVC','parameters':params,'experiment':directory.name,
                  'labels':plan['labels'],'features_sha256':read_json(directory/'features/seal.json')}
            key=fingerprint(spec)[:24];out=directory/'runs'/key;out.mkdir(parents=True,exist_ok=True)
            if not cached_run(out,spec):
                print('Fit',method,'LinearSVC C=',c,flush=True)
                model=LinearSVC(**params)
                with PeakMemory() as memory,warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning);model.fit(x,y)
                    assert not any(issubclass(w.category,ConvergenceWarning) for w in caught),'LinearSVC failed to converge'
                write_json(out/'class_order.json',{'classes':model.classes_.tolist()})
                decision=model.decision_function(xv);predicted=model.predict(xv).tolist()
                result=evaluate(out,'validation',val_ids,yv,predicted,decision,plan['labels'])
                joblib.dump(model,out/'classifier.joblib');loaded=joblib.load(out/'classifier.joblib')
                assert np.array_equal(loaded.predict(xv),predicted) and np.array_equal(loaded.decision_function(xv),decision)
                write_json(out/'spec.json',spec)
                write_json(out/'summary.json',{'spec':spec,'validation':result,'fit_seconds':memory.elapsed,
                     'peak_rss_mib':memory.peak/2**20,'n_iter':int(model.n_iter_),
                     'model_bytes':(out/'classifier.joblib').stat().st_size,'reload_validation_exact':True})
                seal_run(out,spec)
                # Class ordering is needed to interpret the complete decision vector.
                seal=read_json(out/'seal.json');seal['files']['class_order.json']=file_hash(out/'class_order.json')
                write_json(out/'seal.json',seal)
            result=read_json(out/'metrics_validation.json')
            candidates.append({'run':key,'C':c,'validation_macro_f1':result['macro_f1']})
            index=read_json(directory/'index.json');index['runs'][key]={'method':method,'C':c}
            write_json(directory/'index.json',index)
        best=sorted(candidates,key=lambda r:(-r['validation_macro_f1'],r['C']))[0]
        index=read_json(directory/'index.json');index['selection'][method]=best
        write_json(directory/'index.json',index)
        print('Selected',method,'C=',best['C'],'validation Macro-F1=',round(best['validation_macro_f1'],4),flush=True)
        del all_x,x,xv;gc.collect()


def finalize(directory,docs):
    plan=read_json(directory/'plan.json');index=read_json(directory/'index.json')
    assert set(index['selection'])==set(METHODS) and len(index['runs'])==9
    frozen=directory/'selection_frozen.json'
    if frozen.exists():assert read_json(frozen)['selection']==index['selection']
    else:write_json(frozen,{'frozen':stamp(),'selection':index['selection'],'metric':'validation_macro_f1'})
    index['status']='selection_frozen';write_json(directory/'index.json',index)
    positions={i:n for n,i in enumerate(plan['document_order'])};lookup={d['id']:d for d in docs}
    ids=plan['splits']['test'];truth=[lookup[i]['topic'] for i in ids]
    predictions={};rows=[]
    for method in METHODS:
        rec=index['selection'][method];out=directory/'runs'/rec['run'];spec=read_json(out/'spec.json')
        assert cached_run(out,spec)
        all_x=load_features(directory,method);x=all_x[[positions[i] for i in ids]]
        start=time.perf_counter();model=joblib.load(out/'classifier.joblib')
        prediction=model.predict(x).tolist();decision=model.decision_function(x);elapsed=time.perf_counter()-start
        loaded=joblib.load(out/'classifier.joblib')
        assert np.array_equal(loaded.predict(x),prediction) and np.array_equal(loaded.decision_function(x),decision)
        result=evaluate(out,'test',ids,truth,prediction,decision,plan['labels'])
        summary=read_json(out/'summary.json')
        rows.append({'method':method,'C':rec['C'],'run':rec['run'],'validation_macro_f1':rec['validation_macro_f1'],
             **{k:result[k] for k in ['n','accuracy','macro_f1','micro_f1','weighted_f1']},
             'fit_seconds':summary['fit_seconds'],'predict_and_load_seconds':elapsed,
             'peak_rss_mib':summary['peak_rss_mib'],'model_bytes':summary['model_bytes'],'n_iter':summary['n_iter']})
        predictions[method]=prediction
        write_json(out/'reload_test.json',{'n':len(ids),'labels_exact':True,'all_decision_scores_exact':True})
        del all_x,x,model,loaded;gc.collect()
    assert len({lookup[i]['group'] for i in ids})==len(ids),'A grouped bootstrap is required'
    options=plan['configuration']['bootstrap']
    write_json(directory/'bootstrap.json',paired_bootstrap(truth,predictions,plan['labels'],options['repetitions'],options['seed']))
    save_csv(directory/'results.csv',rows)
    write_json(directory/'test_seal.json',{str(p.relative_to(directory)):file_hash(p) for p in directory.glob('runs/*/*') if p.is_file() and 'test' in p.name})
    index['status']='complete';index['completed']=stamp();write_json(directory/'index.json',index)
    audit(directory,docs)
    print('Three frozen representations + same LinearSVC complete.',flush=True)


def audit(directory,docs):
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
    plan=read_json(directory/'plan.json');index=read_json(directory/'index.json')
    assert index['status']=='complete' and len(index['runs'])==9 and len(index['selection'])==3
    assert read_json(directory/'selection_frozen.json')['selection']==index['selection']
    assert file_hash(ROOT/plan['article_file'])==plan['article_sha256']
    original=read_json(ROOT/plan['topic_experiment']/'plan.json')
    assert file_hash(ROOT/plan['topic_experiment']/'plan.json')==plan['topic_plan_sha256']
    assert plan['splits']==original['splits'] and plan['annotation_seal']==original['annotation_seal']
    assert file_hash(CONFIG)==plan['configuration_sha256']
    for n,h in plan['protocol'].items():assert file_hash(ROOT/'gensim_lab'/n)==h
    for method,source in plan['sources'].items():
        assert all(file_hash(ROOT/source['directory']/n)==h for n,h in source['files'].items())
    validate_partitions(plan['splits'],{d['id']:d for d in docs})
    for n,h in read_json(directory/'features/seal.json').items():assert file_hash(directory/'features'/n)==h
    for n,h in read_json(directory/'test_seal.json').items():assert file_hash(directory/n)==h
    lookup={d['id']:d for d in docs};rows=0
    for run,record in index['runs'].items():
        out=directory/'runs'/run;spec=read_json(out/'spec.json');assert cached_run(out,spec)
        params=joblib.load(out/'classifier.joblib').get_params()
        assert params==LinearSVC(**spec['parameters']).get_params()
        for part in ['validation']+(['test'] if run in {v['run'] for v in index['selection'].values()} else []):
            frame=pd.read_csv(out/f'predictions_{part}.csv',dtype={'id':str})
            assert frame.id.tolist()==plan['splits'][part]
            assert frame.true_topic.tolist()==[lookup[i]['topic'] for i in frame.id]
            truth,predicted=frame.true_topic,frame.predicted_topic;saved=read_json(out/f'metrics_{part}.json')
            assert np.isclose(saved['accuracy'],accuracy_score(truth,predicted),atol=1e-12)
            for avg in ['macro','micro','weighted']:
                assert np.isclose(saved[f'{avg}_f1'],f1_score(truth,predicted,labels=plan['labels'],average=avg,zero_division=0),atol=1e-12)
            assert saved['per_class']==classification_report(truth,predicted,labels=plan['labels'],output_dict=True,zero_division=0)
            assert saved['confusion_matrix']==confusion_matrix(truth,predicted,labels=plan['labels']).tolist()
            assert np.isfinite(frame[[f'decision_{label}' for label in plan['labels']]]).all().all()
            rows+=len(frame)
    for method in METHODS:
        candidates=[(read_json(directory/'runs'/run/'metrics_validation.json')['macro_f1'],r['C'],run) for run,r in index['runs'].items() if r['method']==method]
        assert index['selection'][method]['run']==sorted(candidates,key=lambda r:(-r[0],r[1]))[0][2]
    write_json(directory/'verification.json',{'status':'passed','verified':stamp(),'candidate_runs':9,'selected_runs':3,
        'recomputed_prediction_rows':rows,'same_topic_partition_as_native':True,'same_linear_svc_settings_except_selected_C':True,
        'new_annotation_requests':0,'representation_sources_unchanged':True,'only_train_fit_only_validation_selection':True,
        'all_saved_predictions_and_decisions_reload_exact':True,'labels':'single-pass DeepSeek weak topics',
        'bootstrap_unit':'test article; all near-duplicate groups unique','classifier_solver':'primal liblinear, dual=False'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','audit','predict'])
    parser.add_argument('--method',choices=METHODS);parser.add_argument('--tokens');parser.add_argument('--text')
    args=parser.parse_args()
    if args.stage=='predict':
        directory=ROOT/read_json(BASE/'latest.json')['experiment'];plan=read_json(directory/'plan.json')
        if args.method is None:parser.error('--method is required')
        if args.tokens:tokens=args.tokens.split()
        elif args.text:
            import jieba
            import unicodedata
            tokens=[w for w in jieba.lcut(unicodedata.normalize('NFKC',args.text)) if w.strip() and not all(unicodedata.category(c)[0] in 'PS' for c in w)]
        else:parser.error('Supply --tokens or --text')
        if not tokens:raise ValueError('No valid tokens')
        source=plan['sources'][args.method]
        for n,h in source['files'].items():assert file_hash(ROOT/source['directory']/n)==h
        representation=FrozenRepresentation(args.method,source)
        x,_=representation.transform([{'id':'new','month':'new','tokens':tokens}],1);x=normalize(x,norm='l2',copy=False)
        out=directory/'runs'/read_json(directory/'index.json')['selection'][args.method]['run']
        assert cached_run(out,read_json(out/'spec.json'))
        model=joblib.load(out/'classifier.joblib');decision=model.decision_function(x)[0]
        print(__import__('json').dumps({'topic':model.predict(x)[0],'score_kind':'decision_function',
             'scores':dict(zip(model.classes_,map(float,decision))),'representation':args.method},ensure_ascii=False))
        return
    with artifact_lock(),threadpool_limits(limits=settings()['numerical_threads']):
        directory,docs=prepare()
        if args.stage=='run':features(directory,docs);train(directory,docs);finalize(directory,docs)
        elif args.stage=='audit':audit(directory,docs)


if __name__=='__main__':main()
