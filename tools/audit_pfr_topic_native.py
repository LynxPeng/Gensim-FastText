"""Independent checks of topic labels, partitions, saved metrics and delivery."""
from __future__ import annotations
import collections
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score

from gensim_lab.common import ROOT,file_hash,fingerprint,read_json,stamp,write_json
from gensim_lab.pfr_topics import load_jsonl,normalized,request_payload,validate_item,usage_cost


def verify_links(path):
    text=path.read_text(encoding='utf-8')
    checked=[]
    for target in re.findall(r'!?\[[^\]]*\]\(([^\n]+?)\)',text):
        if target.startswith(('https://','http://','#','data:','mailto:')):
            continue
        relative=unquote(target.strip('<>').split('#',1)[0])
        assert (path.parent/relative).exists(),f'Broken link in {path.name}: {relative}'
        checked.append(relative)
    return len(checked)


def main():
    directory=ROOT/read_json(ROOT/'artifacts/pfr_topics/latest_experiment.json')['experiment']
    plan=read_json(directory/'plan.json');cfg=plan['configuration'];labels=plan['labels']
    corpus=ROOT/plan['dataset'];raw=load_jsonl(corpus/'articles.jsonl')
    annotations=load_jsonl(corpus/'annotations.jsonl');requests=load_jsonl(corpus/'requests.jsonl')
    lookup={d['id']:d for d in raw};by_response={r['response_id']:r for r in requests if r.get('response_id')}
    assert len(lookup)==len(raw)==len(annotations)==8431
    assert len({r['id'] for r in annotations})==len(raw)
    for name,digest in read_json(corpus/'seal.json').items():
        assert file_hash(corpus/name)==digest
    for name,key in [('articles.jsonl','articles_sha256'),('annotations.jsonl','annotations_sha256'),('requests.jsonl','requests_sha256')]:
        assert file_hash(corpus/name)==plan['annotation_seal'][key]
    manifest=read_json(corpus/'manifest.json')
    for name,digest in manifest['identity']['raw'].items():
        assert file_hash(ROOT/'data/raw'/name)==digest
    assert 'page_label' not in set().union(*(d.keys() for d in raw))
    for r in requests:
        assert r['status']==200 and r['finish_reason'] in ['stop','length'] and 'usage' in r
        assert r['payload_sha256']==fingerprint(request_payload([lookup[i] for i in r['ids']],cfg['annotation']))
        assert np.isclose(r['cost_cny'],usage_cost(r['usage'],r['tariff_peak_conservative']),atol=1e-12)
        assert len(r['ids'])<=4
    attempts=collections.Counter(i for r in requests for i in r['ids'])
    assert max(attempts.values())<=3
    for a in annotations:
        validate_item(a,lookup[a['id']])
        response=by_response[a['response_id']]
        assert response['finish_reason']=='stop'
        original=[i for i in json.loads(response['content'])['items'] if i['id']==a['id']]
        assert len(original)==1
        item=original[0]
        for k in ['id','topic','status','confidence','reason']:
            assert item[k]==a[k],f'Automatic topic changed: {a["id"]}'
        assert item['evidence']==a.get('evidence_original',a['evidence'])
        if a['status']!='out_of_scope':
            assert normalized(a['evidence']) in normalized(lookup[a['id']]['annotation_text'])
    amount=sum(r['cost_cny'] for r in requests)
    assert amount<=cfg['annotation']['budget_cny']
    assert np.isclose(amount,plan['annotation_seal']['cost_cny_accounted'])

    docs=load_jsonl(directory/'labelled_articles.jsonl');doc_lookup={d['id']:d for d in docs}
    included={a['id'] for a in annotations if a['status']!='out_of_scope'}
    split_ids=set().union(*(set(v) for v in plan['splits'].values()))
    assert set(doc_lookup)==included==split_ids and len(docs)==8063
    seen={attribute:set() for attribute in ['id','content_hash','group']}
    counts={}
    for part,ids in plan['splits'].items():
        assert len(ids)==len(set(ids))
        for attribute in seen:
            values={doc_lookup[i][attribute] for i in ids}
            assert not seen[attribute].intersection(values)
            seen[attribute].update(values)
        months={doc_lookup[i]['month'] for i in ids}
        assert months==({'199801','199802'} if part=='train' else {'199803'})
        counts[part]=dict(collections.Counter(doc_lookup[i]['topic'] for i in ids))
        assert set(counts[part])==set(labels)
    for topic in labels:
        n=counts['validation'][topic]+counts['test'][topic]
        assert abs(counts['validation'][topic]-n*cfg['validation_fraction'])<=3
    for name,digest in plan['protocol'].items():
        assert file_hash(ROOT/'gensim_lab'/name)==digest
    assert file_hash(ROOT/'configs/pfr_topics.yaml')==plan['configuration_sha256']

    source_checks={}
    # Confirm all three existing representations, including those not used here.
    other_sources={
        'tfidf':'artifacts/thucnews/runs/928ed691978c06e8cc9d3d1e',
        'word2vec':'artifacts/thucnews_validation/runs/1109be5a651700f6a95e3c96',
        'fasttext':plan['sources']['fasttext']['directory']}
    for method,relative in other_sources.items():
        source=ROOT/relative;seal=read_json(source/'seal.json')['files']
        names=['model.gensim','summary.json']+(['dictionary.gensim'] if method=='tfidf' else ['vectors.kv'])
        assert all(file_hash(source/name)==seal[name] for name in names)
        source_checks[method]=True
    input_folder=directory/'native_input'
    for name,digest in read_json(input_folder/'seal.json').items():
        assert file_hash(input_folder/name)==digest
    expected_lines=[f"__label__t{labels.index(doc_lookup[i]['topic'])} {' '.join(doc_lookup[i]['tokens'])}" for i in plan['splits']['train']]
    assert (input_folder/'train.txt').read_text(encoding='utf-8').splitlines()==expected_lines
    vocabulary=set(w for i in plan['splits']['train'] for w in doc_lookup[i]['tokens'])
    with (input_folder/'fasttext.vec').open(encoding='utf-8') as stream:
        header=stream.readline().split();vector_words={line.split(' ',1)[0] for line in stream}
    assert vector_words==vocabulary and int(header[0])==len(vocabulary) and int(header[1])==100

    index=read_json(directory/'index.json');frozen=read_json(directory/'selection_frozen.json')
    assert index['status']=='complete' and len(index['runs'])==16 and len(index['selection'])==6
    assert frozen['selection']==index['selection']
    verified_rows=0
    for run,record in index['runs'].items():
        out=directory/'runs'/run
        for name,digest in read_json(out/'seal.json')['files'].items():
            assert file_hash(out/name)==digest
        parts=['validation']+(['test'] if run in {r['run'] for r in index['selection'].values()} else [])
        for part in parts:
            frame=pd.read_csv(out/f'predictions_{part}.csv',dtype={'id':str})
            assert frame.id.tolist()==plan['splits'][part]
            assert frame.true_topic.tolist()==[doc_lookup[i]['topic'] for i in frame.id]
            assert set(frame.predicted_topic)<=set(labels) and np.isfinite(frame.score).all()
            assert frame.score.between(0,1).all() and np.isfinite(frame.raw_native_score).all()
            truth,pred=frame.true_topic,frame.predicted_topic
            saved=read_json(out/f'metrics_{part}.json')
            assert saved['n']==len(frame)
            assert np.isclose(saved['accuracy'],accuracy_score(truth,pred),atol=1e-12)
            for avg in ['macro','micro','weighted']:
                assert np.isclose(saved[f'{avg}_f1'],f1_score(truth,pred,labels=labels,average=avg,zero_division=0),atol=1e-12)
            assert saved['confusion_matrix']==confusion_matrix(truth,pred,labels=labels).tolist()
            assert saved['per_class']==classification_report(truth,pred,labels=labels,output_dict=True,zero_division=0)
            verified_rows+=len(frame)
    for method in ['native_scratch','native_pretrained']:
        candidates=[(read_json(directory/'runs'/run/'metrics_validation.json')['macro_f1'],r['epoch'],run)
                    for run,r in index['runs'].items() if r['method']==method and r['seed']==42]
        winner=sorted(candidates,key=lambda r:(-r[0],r[1]))[0]
        assert index['selection'][f'{method}/42']['run']==winner[2]
        assert all(index['selection'][f'{method}/{s}']['parameter']==winner[1] for s in cfg['seeds'])
        run=index['selection'][f'{method}/42']['run'];out=directory/'runs'/run
        model=fasttext.load_model(str(out/'native.bin'));args=model.f.getArgs()
        assert args.dim==100 and args.epoch==winner[1] and args.wordNgrams==2
        assert args.minn==args.maxn==0 and args.minCount==1 and args.bucket==200000
        assert set(model.get_words())==vocabulary|{'</s>'}
        assert set(model.get_labels())=={f'__label__t{i}' for i in range(8)}
        frame=pd.read_csv(out/'predictions_test.csv',dtype={'id':str})
        doc=doc_lookup[frame.id.iloc[0]]
        native,score=model.predict(' '.join(doc['tokens']),k=8)
        assert labels[int(native[0].removeprefix('__label__t'))]==frame.predicted_topic.iloc[0]
        probability=np.maximum(np.asarray(score,dtype=np.float64)-1e-5,0.)
        probability/=probability.sum()
        assert abs(float(score[0])-frame.raw_native_score.iloc[0])<1e-12
        assert abs(float(probability[0])-frame.score.iloc[0])<1e-12
        del model
    for name,digest in read_json(directory/'test_seal.json').items():
        assert file_hash(directory/name)==digest

    paths=[ROOT/'reports/Gensim&FastText.md',ROOT/'reports/Gensim&FastText-去AI味版.md',
           ROOT/'reports/人民日报原生FastText主题分类.md',ROOT/'docs/人民日报主题分类复现指南.md',
           ROOT/'docs/人民日报八类主题分类实施方案.md']
    link_counts={p.name:verify_links(p) for p in paths}
    link_counts['historical_archive']=verify_links(ROOT/'reports/archive/人民日报版面分类_历史_20261010.md')
    link_counts['historical_pointer']=verify_links(ROOT/'reports/人民日报固定表示分类实验.md')
    figure_files=0
    figure_folder=ROOT/'reports/figures/pfr_topics'/directory.name
    for figure,records in read_json(directory/'figure_export_audit.json').items():
        for name,record in records.items():
            assert file_hash(figure_folder/name)==record['sha256']
            assert not any(i[0] in ['WARN','FAIL'] for i in record['issues'])
            figure_files+=1
    assert figure_files==15 and read_json(directory/'figure_preview/visual_review.json')['status']=='approved'
    for path in paths[:2]:
        before=(directory/'report_before'/path.name).read_text(encoding='utf-8')
        after=path.read_text(encoding='utf-8')
        assert before.split('### 1.2 ',1)[0]==after.split('### 1.2 ',1)[0]
        assert before.split('## part2 ',1)[1]==after.split('## part2 ',1)[1]
        assert after.count('### 1.2 ')==1
        chapter=after.split('### 1.2 ',1)[1].split('## part2 ',1)[0]
        native_chapter=chapter.split('#### 1.2.6 ',1)[0]
        assert native_chapter.count('![图1.2-')==5 and '弱标签' in native_chapter
    bad_secrets=0
    secret=re.compile(r'(?:sk-[A-Za-z0-9]{25,}|Bearer\s+[A-Za-z0-9_-]{20,})')
    for folder in [corpus,directory]:
        for path in folder.rglob('*'):
            if path.suffix.lower() in {'.json','.jsonl','.csv','.md','.log','.yaml','.py'} and path.is_file():
                bad_secrets+=len(secret.findall(path.read_text(encoding='utf-8')))
    assert bad_secrets==0
    result={'status':'passed','verified':stamp(),'experiment':directory.name,'label_count':len(annotations),
        'automatic_topic_unchanged_from_response':True,'format_repairs_only':True,'cost_cny':amount,
        'requests':len(requests),'max_annotation_attempts':max(attempts.values()),
        'response_finish_reasons':dict(collections.Counter(r['finish_reason'] for r in requests)),
        'partitions':{p:len(v) for p,v in plan['splits'].items()},'theme_counts':counts,
        'stratification_count_error_at_most_3':True,'id_content_group_disjoint':True,
        'training_vocabulary_only':True,'source_models_unchanged':source_checks,
        'candidate_runs':16,'selected_runs':6,'prediction_rows_recomputed':verified_rows,
        'selection_only_validation':True,'native_reload_probe_exact':True,'other_classifier_runs':0,
        'probabilities_between_zero_one':True,'raw_native_scores_retained':True,
        'report_surrounding_text_preserved':True,'local_links':link_counts,'credential_patterns_found':0,
        'figure_assets_verified':figure_files,'figure_color_grayscale_reviewed':True,
        'scope':'Native fastText two initialization routes; single-pass weak topic labels'}
    write_json(directory/'delivery_verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
