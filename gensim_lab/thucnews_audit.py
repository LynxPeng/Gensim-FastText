"""Independent checkpoint, split and intrinsic-metric audit; optional HS diagnostics."""
from __future__ import annotations
import argparse
import collections
import csv
import math
import numpy as np
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.models.word2vec import LineSentence
from scipy.stats import pearsonr, spearmanr
from threadpoolctl import threadpool_limits
from .common import ROOT, file_hash, read_json, stamp, write_json
from .evaluation_support import save_csv, score_event_count
from .thucnews_validation import BASE, update_selection
from .runtime import numerical_threads


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def metric_check(directory, metric):
    name=metric['benchmark']
    if metric['kind']=='similarity':
        items=rows(directory/f'{name}_pairs.csv')
        valid=[r for r in items if r['status']=='ok']
        assert len(items)==metric['total'] and len(valid)==metric['covered']
        if metric['status']=='ok':
            gold=[float(r['human_similarity']) for r in valid]
            predicted=[float(r['cosine_similarity']) for r in valid]
            for actual, keys in [(pearsonr(gold,predicted),('pearson_r','pearson_p')),
                                 (spearmanr(gold,predicted),('spearman_rho','spearman_p'))]:
                assert np.allclose(actual,[metric[k] for k in keys],rtol=1e-10,atol=1e-12)
    else:
        items=rows(directory/f'{name}_questions.csv')
        covered=sum(r['status']=='covered' for r in items)
        correct=sum(r['correct']=='True' for r in items)
        assert (len(items),covered,correct)==(metric['total'],metric['covered'],metric['correct'])
        assert all(r['correct']=='False' for r in items if r['status']=='oov')
        assert math.isclose(correct/len(items),metric['all_accuracy'],abs_tol=1e-12)
        assert (metric['conditional_accuracy'] is None if not covered else
                math.isclose(correct/covered,metric['conditional_accuracy'],abs_tol=1e-12))
    return len(items)


def score_posthoc(directory, model, manifest):
    output=BASE/'posthoc'/directory.name
    summary=output/'summary.json'
    digest=file_hash(directory/'model.gensim')
    if summary.exists():
        previous=read_json(summary)
        if previous['model_sha256']==digest and all(file_hash(output/p)==h for p,h in previous['files'].items()):
            return previous
    data=ROOT/manifest['directory']; output.mkdir(parents=True,exist_ok=True)
    sentences=list(LineSentence(str(data/'holdout.txt')))
    metadata=rows(data/'holdout_sentences.csv')
    assert len(sentences)==len(metadata)==manifest['partitions']['holdout']['sentences']
    scores=model.score(sentences,total_sentences=len(sentences),chunksize=256)
    recorded=[]
    for meta,sentence,score in zip(metadata,sentences,scores,strict=True):
        iv=sum(w in model.wv.key_to_index for w in sentence)
        events=score_event_count(iv,model.window,model.sg)
        assert len(sentence)==int(meta['tokens']) and len(sentence)<=256
        assert np.isfinite(score) and score<=1e-6
        recorded.append({**meta,'iv_tokens':iv,'oov_tokens':len(sentence)-iv,'events':events,
            'context_log_score':float(score),'log_score_per_iv_token':float(score)/iv if events else None,
            'log_score_per_event':float(score)/events if events else None,
            'status':'ok' if events else 'no_context',
            'score_kind':'word2vec_hs_compatibility' if isinstance(model,FastText) else 'word2vec_hs_context'})
    save_csv(output/'heldout_sentence_scores.csv',recorded)
    # Read back every float and independently score a fixed subset after reload.
    loaded=rows(output/'heldout_sentence_scores.csv')
    assert np.array_equal(scores,np.array([float(r['context_log_score']) for r in loaded],dtype=np.float32))
    positions=np.linspace(0,len(sentences)-1,128,dtype=int)
    cls=FastText if isinstance(model,FastText) else Word2Vec
    reloaded=cls.load(str(directory/'model.gensim'))
    check=reloaded.score([sentences[i] for i in positions],total_sentences=len(positions))
    assert np.array_equal(check,scores[positions])
    result={'model_sha256':digest,'selected_epoch':read_json(directory/'summary.json')['selected_epoch'],
            'sentences':len(scores),'tokens':sum(len(s) for s in sentences),
            'score_kind':recorded[0]['score_kind'],'all_csv_floats_exact':True,'reload_128_scores_exact':True,
            'not_sentence_generative_probability':True,'not_used_for_epoch_selection':True,
            'files':{'heldout_sentence_scores.csv':file_hash(output/'heldout_sentence_scores.csv')},'completed':stamp()}
    write_json(summary,result)
    del reloaded
    return result


def audit(posthoc=True):
    from .thucnews import records
    entries=[(d,s) for d,s in records() if s['spec']['method']!='tfidf']
    assert {s['spec']['method'] for _,s in entries}=={'word2vec','fasttext'}
    assert len({s['spec']['benchmark_manifest_sha256'] for _,s in entries})==1
    manifest=read_json(ROOT/entries[0][1]['spec']['manifest'])
    split=read_json(ROOT/entries[0][1]['spec']['benchmark_manifest'])
    assignments=rows(ROOT/split['directory']/'assignments.csv')
    partitions={part:{r['question_key'] for r in assignments if r['part']==part} for part in ['validation','test']}
    assert not partitions['validation'] & partitions['test'] and split['overlap_groups']==0
    assert len(partitions['validation'])==split['validation_unique_groups']
    assert len(partitions['test'])==split['test_unique_groups']
    for benchmark,item in split['benchmarks'].items():
        assert item['validation']['n']+item['test']['n']+item['duplicates_removed']==item['original_n']
        for part in ['validation','test','validation_probe']:
            if part in item:
                assert file_hash(ROOT/item[part]['path'])==item[part]['sha256']
    # Recount the entire actual training stream, never the holdout or benchmark words.
    corpus=ROOT/entries[0][1]['spec']['training_corpus']
    frequencies=collections.Counter(); sentences=0
    for sentence in LineSentence(str(corpus)):
        frequencies.update(sentence); sentences+=1
        if sentences % 4000000 == 0:
            print(f'Training vocabulary recount: {sentences:,} sentences',flush=True)
    assert sum(frequencies.values())==manifest['partitions']['train']['tokens']
    assert sentences==manifest['partitions']['train']['sentences']
    checks=[]
    for directory,s in entries:
        spec=s['spec']; rules=spec['validation']; seal=read_json(directory/'seal.json')
        assert all(file_hash(directory/p)==sha for p,sha in seal['files'].items())
        assert file_hash(ROOT/spec['training_corpus'])==spec['training_corpus_sha256']
        epochs=read_json(directory/'epochs.json'); state={}; checkpoint_checks=[]
        assert rules['min_epochs']<=len(epochs)<=rules['max_epochs']
        assert len(epochs)==s['trained_epochs']==s['checkpoints']
        cls=Word2Vec if spec['method']=='word2vec' else FastText
        for i,r in enumerate(epochs,1):
            assert i==r['epoch'] and r['raw_words']==sum(frequencies.values())
            assert not any('loss' in k for k in r)
            checkpoint=directory/r['checkpoint']
            assert file_hash(checkpoint)==r['checkpoint_sha256']
            model=cls.load(str(checkpoint)); assert model.compute_loss is False
            assert model.workers==spec['embedding']['workers']==20
            primary=next(m for m in r['metrics'] if m['benchmark']==rules['primary_benchmark'])
            path=ROOT/split['benchmarks'][rules['primary_benchmark']]['validation']['path']
            pearson,spearman,_=model.wv.evaluate_word_pairs(str(path),delimiter='\t',restrict_vocab=len(model.wv),case_insensitive=False)
            assert math.isclose(spearman[0],r['primary_score'],abs_tol=1e-12)
            for m in r['metrics']:
                metric_check(directory/'validation'/f'epoch_{i:02d}',m)
            state=update_selection(state,r['primary_score'],i,rules)
            assert (state['best_epoch'],state['stale_epochs'])==(r['best_epoch_so_far'],r['stale_epochs'])
            assert not state['stop'] or i==len(epochs)
            checkpoint_checks.append({'epoch':i,'native_load':True,'primary_equal':True,'all_item_metrics_equal':True})
            del model
        assert state==s['selection'] and s['selected_epoch']==state['best_epoch']
        assert s['stop_reason']==('validation_patience' if state['stop'] else 'max_epochs')
        selected=epochs[s['selected_epoch']-1]
        assert file_hash(directory/'model.gensim')==selected['checkpoint_sha256']
        model=cls.load(str(directory/'model.gensim')); saved=KeyedVectors.load(str(directory/'vectors.kv'))
        expected={w for w,n in frequencies.items() if n>=spec['embedding']['min_count']}
        assert set(model.wv.key_to_index)==expected
        assert all(int(model.wv.get_vecattr(w,'count'))==frequencies[w] for w in expected)
        assert np.array_equal(model.wv.vectors,saved.vectors) and model.wv.index_to_key==saved.index_to_key
        assert np.isfinite(model.wv.vectors).all() and np.all(np.linalg.norm(model.wv.vectors,axis=1)>0)
        assert model.hs==1 and model.negative==0
        if isinstance(model,FastText):
            assert all(np.array_equal(getattr(model.wv,k),getattr(saved,k)) for k in ['vectors_vocab','vectors_ngrams'])
            for word in ['新能源汽车产业测试未登录','量子计算芯片测试未登录']:
                assert word not in expected and np.array_equal(model.wv[word],saved[word])
        for metric in s['metrics']:
            metric_check(directory/'test',metric)
            assert metric['total']==split['benchmarks'][metric['benchmark']]['test']['n']
            if metric['kind']=='analogy':
                covered=[r for r in rows(directory/'test'/f"{metric['benchmark']}_questions.csv") if r['status']=='covered']
                for r in covered[::max(1,len(covered)//16)][:16]:
                    predicted=model.wv.most_similar(positive=[r['b'],r['c']],negative=[r['a']],topn=1)[0][0]
                    assert (predicted==r['expected'])==(r['correct']=='True')
        assert read_json(directory/'save_load_check.json')['all_validation_metrics_equal']
        detail={'model':spec['method'],'seal_sha256':file_hash(directory/'seal.json'),
                'training_only_vocab_and_counts_exact':True,'saved_vectors_equal':True,
                'checkpoint_checks':checkpoint_checks,'selection_rule_exact':True,
                'test_item_metrics_and_sample_analogies_equal':True,'selected_epoch':s['selected_epoch']}
        if posthoc:
            detail['posthoc']=score_posthoc(directory,model,manifest)
        checks.append(detail); del model,saved
        print(f'Independent audit passed: {spec["method"]}',flush=True)
    result={'status':'passed','verified':stamp(),'benchmark_group_overlap':0,
            'primary_validation_pairs':split['benchmarks']['wordsim240']['validation']['n'],
            'training_sentences':sentences,'training_tokens':sum(frequencies.values()),'checks':checks}
    write_json(BASE/'verification.json',result)
    print('All checkpoint, split and metric audits passed',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--no-posthoc',action='store_true'); args=parser.parse_args()
    with threadpool_limits(limits=numerical_threads()):
        audit(posthoc=not args.no_posthoc)
