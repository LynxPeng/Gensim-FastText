"""External word-vector evaluation, with item-level OOV accounting and isolated caches."""
from __future__ import annotations

import argparse
import collections
import copy
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from threadpoolctl import threadpool_limits

from .common import ARTIFACTS, PROCESSED, ROOT, file_hash, fingerprint, read_json, stamp, write_json
from .models import Bundle
from .reporting import records, uses
from .runner import worker_environment
from .integrity import artifact_lock, validate_splits
from .evaluation_support import save_csv

BASE = ARTIFACTS / 'word_evaluation'
BENCH = ROOT / 'references' / 'word_benchmarks'


def prepare_benchmarks():
    manifest = read_json(BENCH / 'manifest.json')
    for f in manifest['files']:
        if file_hash(ROOT / f['path']) != f['sha256']:
            raise ValueError('Benchmark archive checksum mismatch')
    out = BASE / 'benchmarks'
    out.mkdir(parents=True, exist_ok=True)
    sources = {'wordsim240': BENCH/'CWE/data/240.txt', 'wordsim297': BENCH/'CWE/data/297.txt'}
    result = {}
    for name, source in sources.items():
        triples = []
        for line in source.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            a, b, val = line.split()
            assert math.isfinite(float(val))
            triples.append((a, b, val))
        if len(triples) != int(name[-3:]):
            raise AssertionError(f'Unexpected benchmark size: {name}')
        target = out / f'{name}.tsv'
        target.write_text(''.join('\t'.join(x)+'\n' for x in triples), encoding='utf-8')
        result[name] = {'kind': 'similarity', 'n': len(triples), 'path': str(target.relative_to(ROOT)),
                        'sha256': file_hash(target), 'source_sha256': file_hash(source)}
        if name == 'wordsim297':
            reduced = [x for x in triples if x[:2] != ('OPEC', '石油')]
            assert len(reduced) == 296
            target = out / 'wordsim296.tsv'
            target.write_text(''.join('\t'.join(x)+'\n' for x in reduced), encoding='utf-8')
            result['wordsim296'] = {'kind': 'similarity', 'n': len(reduced), 'path': str(target.relative_to(ROOT)),
                                    'sha256': file_hash(target), 'removed_pair': ['OPEC', '石油']}
    analogies = {'CA_translated': BENCH/'Chinese-Word-Vectors/testsets/CA_translated/ca_translated.txt',
                'CA8_morphological': BENCH/'Chinese-Word-Vectors/testsets/CA8/morphological.txt',
                'CA8_semantic': BENCH/'Chinese-Word-Vectors/testsets/CA8/semantic.txt'}
    for name, source in analogies.items():
        lines = [x.strip() for x in source.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        assert lines[0].startswith(': ')
        n = sum(not x.startswith(':') for x in lines)
        for x in lines:
            if not x.startswith(':') and len(x.split()) != 4:
                raise ValueError(f'Malformed analogy: {x}')
        target = out / f'{name}.txt'
        target.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        result[name] = {'kind': 'analogy', 'n': n, 'path': str(target.relative_to(ROOT)),
                        'sha256': file_hash(target), 'source_sha256': file_hash(source)}
    write_json(out/'manifest.json', result)
    return result




def seal(directory, spec):
    files={str(p.relative_to(directory)): file_hash(p) for p in directory.rglob('*')
           if p.is_file() and p.name not in ['seal.json','console.log']}
    write_json(directory/'seal.json', {'spec_fingerprint':fingerprint(spec),'files':files,'completed':stamp()})


def cache_valid(directory, spec):
    if not (directory/'seal.json').exists():
        return False
    manifest=read_json(directory/'seal.json')
    return manifest['spec_fingerprint']==fingerprint(spec) and all(
        (directory/p).is_file() and file_hash(directory/p)==sha for p,sha in manifest['files'].items())




def finite(x):
    return float(x) if np.isfinite(x) else None


def correlation_status(gold, predicted):
    if len(gold)<3:
        return 'insufficient_pairs'
    if np.ptp(gold)==0 or np.ptp(predicted)==0:
        return 'constant_values'
    return 'ok'


def evaluate_pairs(wv, source, output, name, common_words=None):
    rows=[]
    vocab=wv.key_to_index
    for i,line in enumerate(source.read_text(encoding='utf-8').splitlines(),1):
        a,b,val=line.split('\t')
        covered=a in vocab and b in vocab
        generated = None
        # Separate exploratory FastText subword result; never mixed into the official IV metric.
        if not covered and hasattr(wv, 'vectors_ngrams'):
            try:
                va,vb=wv.get_vector(a),wv.get_vector(b)
                if np.linalg.norm(va)>0 and np.linalg.norm(vb)>0:
                    generated=float(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)))
            except KeyError:
                pass
        rows.append({'item':i,'word1':a,'word2':b,'human_similarity':float(val),
                     'cosine_similarity':float(wv.similarity(a,b)) if covered else None,
                     'oov_subword_cosine':generated,
                     'status':'ok' if covered else 'oov',
                     'common_subset':bool(covered and (common_words is None or a in common_words and b in common_words))})
    save_csv(output/f'{name}_pairs.csv',rows)
    valid=[r for r in rows if r['status']=='ok']
    status=correlation_status([r['human_similarity'] for r in valid],[r['cosine_similarity'] for r in valid])
    metrics={'benchmark':name,'kind':'similarity','total':len(rows),'covered':len(valid),
             'oov_count':len(rows)-len(valid),'oov_ratio_percent':100*(len(rows)-len(valid))/len(rows),
             'coverage':len(valid)/len(rows),'status':status}
    if status=='ok':
        pearson,spearman,oov=wv.evaluate_word_pairs(str(source),delimiter='\t',case_insensitive=False,
                                                  restrict_vocab=len(wv),dummy4unknown=False)
        p2=pearsonr([r['human_similarity'] for r in valid],[r['cosine_similarity'] for r in valid])
        s2=spearmanr([r['human_similarity'] for r in valid],[r['cosine_similarity'] for r in valid])
        assert np.allclose(pearson,p2) and np.allclose(spearman,s2) and np.isclose(oov,metrics['oov_ratio_percent'])
        metrics.update(pearson_r=finite(pearson[0]),pearson_p=finite(pearson[1]),
                       spearman_rho=finite(spearman[0]),spearman_p=finite(spearman[1]))
    else:
        metrics.update(pearson_r=None,pearson_p=None,spearman_rho=None,spearman_p=None)
    shared=[r for r in rows if r['common_subset']]
    cs=correlation_status([r['human_similarity'] for r in shared],[r['cosine_similarity'] for r in shared])
    metrics.update(common_n=len(shared),common_status=cs,
                   common_spearman_rho=finite(spearmanr([r['human_similarity'] for r in shared],
                     [r['cosine_similarity'] for r in shared])[0]) if cs=='ok' else None)
    generatable=[r for r in rows if r['cosine_similarity'] is not None or r['oov_subword_cosine'] is not None]
    gg=[r['human_similarity'] for r in generatable]
    gp=[r['cosine_similarity'] if r['cosine_similarity'] is not None else r['oov_subword_cosine'] for r in generatable]
    gs=correlation_status(gg,gp)
    metrics.update(subword_generated_oov_pairs=sum(r['oov_subword_cosine'] is not None for r in rows),
                   generatable_n=len(generatable),generatable_status=gs,
                   generatable_spearman_rho=finite(spearmanr(gg,gp)[0]) if gs=='ok' else None)
    return metrics


def evaluate_analogies(wv,source,output,name):
    rows=[]
    section=None
    for i,line in enumerate(source.read_text(encoding='utf-8').splitlines(),1):
        if line.startswith(':'):
            section=line[1:].strip()
            continue
        q=tuple(line.split())
        covered=all(w in wv.key_to_index for w in q)
        rows.append({'line':i,'section':section,'a':q[0],'b':q[1],'c':q[2],'expected':q[3],
                     'status':'covered' if covered else 'oov','correct':False})
    # OOV-as-wrong uses the exact official Gensim method; conditional score derived from covered rows.
    overall,sections=wv.evaluate_word_analogies(str(source),restrict_vocab=len(wv),case_insensitive=False,
                                               dummy4unknown=True)
    correctness={}
    for s in sections[:-1]:
        for q in s['correct']:
            correctness[(s['section'],tuple(q))]=True
    for r in rows:
        r['correct']=correctness.get((r['section'],(r['a'],r['b'],r['c'],r['expected'])),False)
    n=len(rows); covered=sum(r['status']=='covered' for r in rows); correct=sum(r['correct'] for r in rows)
    assert np.isclose(overall,correct/n) and sum(len(s['correct'])+len(s['incorrect']) for s in sections[:-1])==n
    save_csv(output/f'{name}_questions.csv',rows)
    write_json(output/f'{name}_gensim_sections.json',sections)
    grouped=[]
    for sec in dict.fromkeys(r['section'] for r in rows):
        group=[r for r in rows if r['section']==sec]
        cv=sum(r['status']=='covered' for r in group); cc=sum(r['correct'] for r in group)
        grouped.append({'section':sec,'total':len(group),'covered':cv,'oov_count':len(group)-cv,
                        'correct':cc,'conditional_accuracy':cc/cv if cv else None,
                        'all_accuracy':cc/len(group)})
    save_csv(output/f'{name}_section_metrics.csv',grouped)
    return {'benchmark':name,'kind':'analogy','total':n,'covered':covered,'oov_count':n-covered,
            'coverage':covered/n,'oov_ratio_percent':100*(n-covered)/n,'correct':correct,
            'conditional_accuracy':correct/covered if covered else None,'all_accuracy':correct/n,
            'status':'ok' if covered else 'no_covered_questions','sections':grouped}


def sources(rows,plan=None):
    selected=[]
    for r in rows:
        if r['method'] not in ['word2vec','fasttext'] or uses(r,'smoke') or 'source_run' in r['spec']:
            continue
        selected.append({'id':r['run'],'origin':'original','directory':str(r['directory']),
                         'method':r['method'],'classes':r['classes'],'seed':r['seed'],'uses':r['uses'],
                         'spec':r['spec']})
    return selected


def evaluate(selected,benchmarks):
    # Common-covered similarity comparison for the six original 12-class main models.
    primary=[x for x in selected if x['origin']=='original' and x['classes']==12 and
             any(u['profile']=='main' for u in x['uses'])]
    wordsets=[]
    for x in primary:
        wordsets.append(set(Bundle.load(Path(x['directory'])).representation.model.wv.key_to_index))
    common=set.intersection(*wordsets)
    common_hash=fingerprint(sorted(common))
    write_json(BASE/'common_vocabulary.json',{'model_ids':[x['id'] for x in primary],
                                            'n':len(common),'sha256':common_hash})
    all_rows=[]
    for i,item in enumerate(selected,1):
        source=Path(item['directory'])
        spec={'model_sha256':file_hash(source/'bundle.joblib'),'benchmarks':benchmarks,
              'protocol_sha256':file_hash(Path(__file__)),'common_vocab':common_hash,
              'case_insensitive':False,'restrict_vocab':'full','dummy4unknown':True}
        out=BASE/'evaluations'/item['id']
        out.mkdir(parents=True,exist_ok=True)
        print(f'INTRINSIC {i}/{len(selected)} {item["id"]} {item["method"]}',flush=True)
        if cache_valid(out,spec):
            result=read_json(out/'metrics.json')
        else:
            started=time.perf_counter()
            bundle=Bundle.load(source)
            wv=bundle.representation.model.wv
            wv.fill_norms(force=True)
            if not np.all(np.isfinite(wv.vectors)) or np.any(wv.norms<=0):
                raise ValueError('Degenerate vectors')
            result=[]
            for name,b in benchmarks.items():
                path=ROOT/b['path']
                r=evaluate_pairs(wv,path,out,name,common) if b['kind']=='similarity' else evaluate_analogies(wv,path,out,name)
                result.append(r)
            queries={}
            for word in ['中国','经济','教育','农业','人民','体育','科学','文化']:
                if word in wv.key_to_index:
                    queries[word]={'nearest_neighbors':[(w,float(s)) for w,s in wv.most_similar(word,topn=10)]}
                else:
                    queries[word]={'status':'oov'}
            write_json(out/'query_diagnostics.json',queries)
            write_json(out/'metrics.json',result)
            write_json(out/'metadata.json',{'elapsed_seconds':time.perf_counter()-started,'vocabulary_size':len(wv),
                                           'spec':spec,'completed':stamp()})
            seal(out,spec)
        for r in result:
            all_rows.append({k:v for k,v in {**r,'id':item['id'],'origin':item['origin'],'method':item['method'],
                                            'classes':item['classes'],'seed':item['seed']}.items() if k!='sections'})
    fields=list(dict.fromkeys(k for r in all_rows for k in r))
    save_csv(BASE/'metrics.csv',all_rows,fields)
    write_json(BASE/'metrics.json',all_rows)
    write_json(BASE/'evaluated_models.json',selected)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','train','evaluate','report'],default='all')
    parser.add_argument('--export',action='store_true',help='Export visually reviewed figures')
    args=parser.parse_args()
    if args.stage == 'train':
        parser.error('旧训练计账入口已移除。逐轮验证使用 python -m gensim_lab.thucnews_validation。')
    BASE.mkdir(parents=True,exist_ok=True)
    from .runtime import numerical_threads
    with artifact_lock(), threadpool_limits(limits=numerical_threads()):
        rows=records()
        write_json(BASE/'pipeline.json',{'status':'running','stage':args.stage,'started':stamp()})
        if args.stage in ['all','evaluate']:
            evaluate(sources(rows),prepare_benchmarks())
        if args.stage in ['all','report']:
            from .word_figures import report
            report(export=args.export)
        write_json(BASE/'pipeline.json',{'status':'complete','stage':args.stage,'completed':stamp(),
                                       'instrumentation':'validation checkpoints replace training counters'})


if __name__=='__main__':
    main()
