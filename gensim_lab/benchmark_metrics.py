"""Intrinsic evaluation with item-level coverage and independently checked metrics."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from .common import write_json
from .evaluation_support import save_csv

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


