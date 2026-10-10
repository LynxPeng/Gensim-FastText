"""Native supervised fastText only; PFR topic labels, validation-only selection."""
from __future__ import annotations

import argparse
import collections
import copy
import gc
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from .common import ROOT, PeakMemory, environment, file_hash, fingerprint, read_json, stamp, write_json
from .evaluation_support import save_csv
from .fasttext_transfer import export_initialization, training_vocabulary
from .integrity import artifact_lock
from .pfr_frozen_classification import metrics, cached_run, seal_run, validate_partitions
from .pfr_topics import BASE, CONFIG, configuration, load_jsonl, make_splits, prepare as prepare_corpus, validate_item
from .runtime import settings as runtime_settings

VARIANTS = ['native_scratch', 'native_pretrained']


def source_records(cfg):
    result = {}
    for method, relative in cfg['sources'].items():
        folder = ROOT / relative
        seal = read_json(folder / 'seal.json')
        summary = read_json(folder / 'summary.json')
        assert summary['spec']['method'] == method
        names = ['model.gensim', 'vectors.kv', 'summary.json']
        hashes = {n: file_hash(folder / n) for n in names}
        assert all(seal['files'].get(n) == h for n, h in hashes.items())
        result[method] = {'directory': relative, 'files': hashes, 'spec': summary['spec']}
    return result


def prepare():
    cfg = configuration()
    corpus = prepare_corpus()
    seal = read_json(corpus / 'annotation_seal.json')
    for name, field in [('articles.jsonl','articles_sha256'), ('annotations.jsonl','annotations_sha256'),
                        ('requests.jsonl','requests_sha256')]:
        assert file_hash(corpus / name) == seal[field]
    raw = load_jsonl(corpus / 'articles.jsonl')
    annotations = load_jsonl(corpus / 'annotations.jsonl')
    labelled = {r['id']:r for r in annotations}
    assert len(labelled) == len(annotations) == len(raw)
    docs, excluded = [], []
    for d in raw:
        item = validate_item(labelled[d['id']], d)
        if item['status'] == 'out_of_scope':
            excluded.append({'id':d['id'], 'source_id':d['source_id'], 'month':d['month'],
                             'reason':item['reason'], 'evidence':item['evidence']})
        else:
            docs.append(dict(d, topic=item['topic'], annotation_status=item['status'],
                             annotation_confidence=item['confidence']))
    splits, duplicate_test = make_splits(docs, cfg)
    lookup = {d['id']:d for d in docs}
    for i in duplicate_test:
        excluded.append({'id':i,'source_id':lookup[i]['source_id'],'month':lookup[i]['month'],
                         'reason':'March group present in training'})
    docs = [d for d in docs if d['id'] not in set(duplicate_test)]
    protocol_names = ['pfr_topics.py','pfr_topic_native.py','data.py','fasttext_transfer.py',
                      'pfr_frozen_classification.py','common.py','runtime.py']
    plan = {'configuration':cfg, 'configuration_sha256':file_hash(CONFIG), 'runtime':runtime_settings(),
            'sources':source_records(cfg), 'dataset':corpus.relative_to(ROOT).as_posix(),
            'annotation_seal':seal, 'labels':cfg['topics'], 'splits':splits,
            'document_order':[d['id'] for d in docs],
            'protocol':{n:file_hash(ROOT/'gensim_lab'/n) for n in protocol_names},
            'environment':environment(), 'scope':'Native fastText ONLY; no SVM or classifier comparison'}
    identity = {k:v for k,v in plan.items() if k!='environment'}
    directory = BASE/'experiments'/fingerprint(identity)[:24]
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'plan.json').exists():
        assert {k:v for k,v in read_json(directory/'plan.json').items() if k!='environment'} == identity
    else:
        write_json(directory/'plan.json',plan)
        with (directory/'labelled_articles.jsonl').open('w',encoding='utf-8',newline='\n') as f:
            for d in docs:
                f.write(__import__('json').dumps(d,ensure_ascii=False)+'\n')
        save_csv(directory/'topic_exclusions.csv',excluded)
        save_csv(directory/'split_counts.csv',[{'partition':part,'topic':topic,'articles':
                 sum(lookup[i]['topic']==topic for i in ids)} for part,ids in splits.items() for topic in cfg['topics']])
    if not (directory/'index.json').exists():
        write_json(directory/'index.json',{'status':'prepared','started':stamp(),'runs':{},'selection':{}})
    write_json(BASE/'latest_experiment.json',{'experiment':directory.relative_to(ROOT).as_posix()})
    print('Topic partitions:',{p:len(ids) for p,ids in splits.items()},flush=True)
    return directory,docs


def native_distribution(model,tokens,labels):
    """Recover softmax probabilities from v0.9.2 exp(log(p + 1e-5))."""
    native,raw=model.predict(' '.join(tokens),k=len(labels))
    raw=np.asarray(raw,dtype=np.float64)
    assert len(native)==len(labels) and np.isfinite(raw).all()
    probability=np.maximum(raw-1e-5,0.)
    total=probability.sum()
    assert total>0
    probability/=total
    themes=[labels[int(label.removeprefix('__label__t'))] for label in native]
    assert set(themes)==set(labels)
    return themes,probability,raw


def native_prediction(model,docs,labels):
    predicted, scores, raw_scores = [], [], []
    for d in docs:
        themes,probability,raw=native_distribution(model,d['tokens'],labels)
        predicted.append(themes[0]);scores.append(float(probability[0]));raw_scores.append(float(raw[0]))
    return predicted,np.asarray(scores),np.asarray(raw_scores)


def evaluate(out,part,docs,predicted,scores,labels,raw_scores):
    truth=[d['topic'] for d in docs]
    result=metrics(truth,predicted,labels)
    assert len(docs)==len(predicted)==len(scores)==len(raw_scores) and np.isfinite(scores).all()
    assert np.all((scores>=0)&(scores<=1))
    write_json(out/f'metrics_{part}.json',result)
    save_csv(out/f'predictions_{part}.csv',[{'id':d['id'],'source_id':d['source_id'],
              'true_topic':y,'predicted_topic':p,'score':float(s),'raw_native_score':float(raw),'score_kind':'probability'}
              for d,y,p,s,raw in zip(docs,truth,predicted,scores,raw_scores)])
    save_csv(out/f'per_class_{part}.csv',[{'topic':label,**result['per_class'][label]} for label in labels])
    save_csv(out/f'confusion_{part}.csv',[{'true_topic':label,**dict(zip(labels,row))}
              for label,row in zip(labels,result['confusion_matrix'])])
    return result


def train(directory,docs):
    import fasttext
    from gensim.models import KeyedVectors
    plan=read_json(directory/'plan.json')
    cfg,labels=plan['configuration'],plan['labels']
    lookup={d['id']:d for d in docs}
    train_docs=[lookup[i] for i in plan['splits']['train']]
    val=[lookup[i] for i in plan['splits']['validation']]
    folder=directory/'native_input';folder.mkdir(exist_ok=True)
    text=''.join(f"__label__t{labels.index(d['topic'])} {' '.join(d['tokens'])}\n" for d in train_docs)
    (folder/'train.txt').write_text(text,encoding='utf-8',newline='\n')
    if not (folder/'seal.json').exists():
        wv=KeyedVectors.load(str(ROOT/plan['sources']['fasttext']['directory']/'vectors.kv'))
        assert wv.vector_size==cfg['native']['dim']
        record=export_initialization(wv,training_vocabulary(train_docs),folder/'fasttext.vec',generate_oov=True)
        del wv;gc.collect()
        write_json(folder/'initialization.json',record)
        native={k:v for k,v in cfg['native'].items() if k!='epochs'}
        probe=fasttext.train_supervised(input=str(folder/'train.txt'),epoch=1,lr=0.,
                    **{k:v for k,v in native.items() if k!='lr'},thread=plan['runtime']['training_threads'],
                    pretrainedVectors=str(folder/'fasttext.vec'),seed=42,verbose=0)
        assert all(np.array_equal(probe.get_word_vector(w),np.array(v,dtype=np.float32))
                   for w,v in record['probe_vectors'].items())
        write_json(folder/'initialization_probe.json',{'lr_zero_import_exact':True,'words':len(record['probe_vectors'])})
        del probe
        write_json(folder/'seal.json',{n:file_hash(folder/n) for n in
                   ['train.txt','fasttext.vec','initialization.json','initialization_probe.json']})
    for n,h in read_json(folder/'seal.json').items():
        assert file_hash(folder/n)==h
    native={k:v for k,v in cfg['native'].items() if k!='epochs'}
    native['thread']=plan['runtime']['training_threads']
    for variant in VARIANTS:
        selected_epoch=None
        for seed in cfg['seeds']:
            grid=cfg['native']['epochs'] if selected_epoch is None else [selected_epoch]
            candidates=[]
            for epoch in grid:
                spec={'method':variant,'head':'native','seed':seed,'epoch':epoch,'parameters':native,
                      'experiment':directory.name,'labels':labels,'input_seal':read_json(folder/'seal.json')}
                run=fingerprint(spec)[:24];out=directory/'runs'/run;out.mkdir(parents=True,exist_ok=True)
                if not cached_run(out,spec):
                    kwargs=dict(native,input=str(folder/'train.txt'),epoch=epoch,seed=seed,verbose=0)
                    if variant=='native_pretrained':
                        kwargs['pretrainedVectors']=str(folder/'fasttext.vec')
                    print(f'Fit {variant} epoch={epoch} seed={seed}',flush=True)
                    with PeakMemory() as memory:
                        model=fasttext.train_supervised(**kwargs)
                    assert set(model.get_words())==set(training_vocabulary(train_docs))|{'</s>'}
                    assert set(model.get_labels())=={f'__label__t{i}' for i in range(len(labels))}
                    prediction,scores,raw_scores=native_prediction(model,val,labels)
                    result=evaluate(out,'validation',val,prediction,scores,labels,raw_scores)
                    model.save_model(str(out/'native.bin'))
                    loaded=fasttext.load_model(str(out/'native.bin'))
                    pp,ss,rr=native_prediction(loaded,val,labels)
                    assert pp==prediction and np.array_equal(ss,scores) and np.array_equal(rr,raw_scores)
                    changes=None
                    if variant=='native_pretrained':
                        probes=read_json(folder/'initialization.json')['probe_vectors']
                        values=[float(np.linalg.norm(model.get_word_vector(w)-np.array(v))) for w,v in probes.items()]
                        changes={'words':len(values),'mean_l2_change':float(np.mean(values)),
                                 'maximum_l2_change':max(values),'changed_words':sum(x>1e-6 for x in values)}
                    write_json(out/'spec.json',spec)
                    write_json(out/'summary.json',{'spec':spec,'fit_seconds':memory.elapsed,
                        'peak_rss_mib':memory.peak/2**20,'model_bytes':(out/'native.bin').stat().st_size,
                        'validation':result,'reload_validation_exact':True,'source_vector_change':changes,
                        'training_vocabulary_only':True})
                    seal_run(out,spec)
                    del model,loaded;gc.collect()
                result=read_json(out/'metrics_validation.json')
                candidates.append({'run':run,'parameter':epoch,'macro_f1':result['macro_f1']})
                index=read_json(directory/'index.json')
                index['runs'][run]={'method':variant,'seed':seed,'epoch':epoch}
                write_json(directory/'index.json',index)
            best=max(candidates,key=lambda r:r['macro_f1'])
            if selected_epoch is None:
                selected_epoch=best['parameter']
            index=read_json(directory/'index.json')
            index['selection'][f'{variant}/{seed}']={**best,'seed':seed}
            write_json(directory/'index.json',index)
            print('Selected',variant,seed,'epoch',best['parameter'],'val Macro-F1',round(best['macro_f1'],4),flush=True)


def paired_bootstrap(truth,predictions,labels,repetitions,seed):
    truth=np.asarray(truth);rng=np.random.default_rng(seed)
    strata=[np.flatnonzero(truth==label) for label in labels]
    if any(not len(s) for s in strata):
        raise ValueError('Every theme is required for the bootstrap')
    values={m:np.empty(repetitions) for m in VARIANTS}
    for r in range(repetitions):
        chosen=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        for m in VARIANTS:
            values[m][r]=f1_score(truth[chosen],np.asarray(predictions[m])[chosen],labels=labels,average='macro',zero_division=0)
    difference=values['native_pretrained']-values['native_scratch']
    return {'repetitions':repetitions,'seed':seed,'training_seed':42,
            'kind':'paired theme-stratified percentile bootstrap',
            'intervals':{m:np.quantile(v,[.025,.975]).tolist() for m,v in values.items()},
            'difference':{'a':'native_pretrained','b':'native_scratch',
                'estimate':float(f1_score(truth,predictions['native_pretrained'],labels=labels,average='macro')-
                                 f1_score(truth,predictions['native_scratch'],labels=labels,average='macro')),
                'ci_low':float(np.quantile(difference,.025)),'ci_high':float(np.quantile(difference,.975))}}


def finalize(directory,docs):
    import fasttext
    plan,index=read_json(directory/'plan.json'),read_json(directory/'index.json')
    cfg,labels=plan['configuration'],plan['labels']
    assert len(index['selection'])==len(VARIANTS)*len(cfg['seeds'])
    selected=copy.deepcopy(index['selection'])
    frozen=directory/'selection_frozen.json'
    if frozen.exists():
        assert read_json(frozen)['selection']==selected
    else:
        write_json(frozen,{'frozen':stamp(),'selection':selected,'metric':'validation_macro_f1'})
    index['status']='selection_frozen';write_json(directory/'index.json',index)
    lookup={d['id']:d for d in docs}
    test=[lookup[i] for i in plan['splits']['test']]
    predictions,rows={},[]
    for key,rec in selected.items():
        variant,seed=key.split('/');out=directory/'runs'/rec['run']
        assert cached_run(out,read_json(out/'spec.json'))
        start=time.perf_counter();model=fasttext.load_model(str(out/'native.bin'))
        predicted,scores,raw_scores=native_prediction(model,test,labels)
        elapsed=time.perf_counter()-start
        loaded=fasttext.load_model(str(out/'native.bin'))
        pp,ss,rr=native_prediction(loaded,test,labels)
        assert pp==predicted and np.array_equal(ss,scores) and np.array_equal(rr,raw_scores)
        result=evaluate(out,'test',test,predicted,scores,labels,raw_scores)
        write_json(out/'reload_test.json',{'n':len(test),'labels_exact':True,'scores_exact':True})
        summary=read_json(out/'summary.json')
        rows.append({'method':variant,'seed':int(seed),'epoch':rec['parameter'],'run':rec['run'],
              'validation_macro_f1':rec['macro_f1'],**{k:result[k] for k in
              ['n','accuracy','macro_f1','micro_f1','weighted_f1']},'fit_seconds':summary['fit_seconds'],
              'predict_and_load_seconds':elapsed,'peak_rss_mib':summary['peak_rss_mib'],'model_bytes':summary['model_bytes']})
        if int(seed)==42:
            predictions[variant]=predicted
        del model,loaded;gc.collect()
    truth=[d['topic'] for d in test]
    assert len({d['group'] for d in test})==len(test),'Repeated test groups need a grouped bootstrap'
    write_json(directory/'bootstrap.json',paired_bootstrap(truth,predictions,labels,
               cfg['bootstrap']['repetitions'],cfg['bootstrap']['seed']))
    counts=collections.Counter(lookup[i]['topic'] for i in plan['splits']['train'])
    majority=max(labels,key=counts.get)
    write_json(directory/'majority.json',{'topic':majority,'metrics':metrics(truth,[majority]*len(test),labels)})
    save_csv(directory/'results.csv',rows)
    checks={m:all(file_hash(ROOT/s['directory']/n)==h for n,h in s['files'].items()) for m,s in plan['sources'].items()}
    assert all(checks.values())
    validate_partitions(plan['splits'],lookup)
    write_json(directory/'test_seal.json',{str(p.relative_to(directory)):file_hash(p)
               for p in directory.glob('runs/*/*') if p.is_file() and 'test' in p.name})
    write_json(directory/'verification.json',{'status':'passed','verified':stamp(),'selected_runs':len(selected),
               'candidate_runs':len(index['runs']),'classification_corpus':'PFR 9801--9803 ONLY',
               'task':'Eight news themes, single-pass DeepSeek weak labels',
               'train_months':['199801','199802'],'validation_and_test_month':'199803',
               'sources_unchanged':checks,'partition_id_content_group_disjoint':True,
               'selection_before_test':True,'all_saved_predictions_reload_exact':True,'svm_runs':0,'other_classifier_runs':0})
    index['status']='complete';index['completed']=stamp();write_json(directory/'index.json',index)
    print('Native topic experiment complete.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run','finalize','predict'])
    parser.add_argument('--run',type=Path)
    parser.add_argument('--tokens')
    parser.add_argument('--text')
    args=parser.parse_args()
    if args.stage=='predict':
        import fasttext
        out=args.run if args.run.is_absolute() else ROOT/args.run
        plan=read_json(out.parent.parent/'plan.json')
        assert cached_run(out,read_json(out/'spec.json'))
        if args.tokens:
            tokens=args.tokens.split()
        elif args.text:
            import jieba
            tokens=[t for t in jieba.lcut(args.text) if t.strip() and not all(__import__('unicodedata').category(c)[0] in 'PS' for c in t)]
        else:
            parser.error('Supply --tokens for exact reproduction or --text for a new article')
        if not tokens or any(t.startswith('__label__') or t=='</s>' for t in tokens):
            raise ValueError('Invalid input tokens')
        model=fasttext.load_model(str(out/'native.bin'))
        themes,prob,_=native_distribution(model,tokens,plan['labels'])
        scores={theme:float(s) for theme,s in zip(themes,prob)}
        print(__import__('json').dumps({'topic':next(iter(scores)),'score_kind':'probability','scores':scores,
                  'model_method':read_json(out/'spec.json')['method']},ensure_ascii=False))
        return
    with artifact_lock(),threadpool_limits(limits=runtime_settings()['numerical_threads']):
        directory,docs=prepare()
        if args.stage=='run':
            train(directory,docs);finalize(directory,docs)
        elif args.stage=='finalize':
            finalize(directory,docs)


if __name__=='__main__':
    main()
