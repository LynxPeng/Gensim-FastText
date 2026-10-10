"""Qualitative Gensim APIs on each validation-selected embedding; no model fitting."""
from gensim.models import KeyedVectors
from .common import ROOT, file_hash, write_json, stamp
from .thucnews import records
from .thucnews_validation import BASE


def run():
    for directory,summary in records():
        if summary['spec']['method']=='tfidf':
            continue
        wv=KeyedVectors.load(str(directory/'vectors.kv'))
        cases={
            'most_similar':(['经济'],lambda:wv.most_similar('经济',topn=10)),
            'similarity':(['经济','股票'],lambda:float(wv.similarity('经济','股票'))),
            'n_similarity':(['经济','股票','新闻','科技'],lambda:float(wv.n_similarity(['经济','股票'],['新闻','科技']))),
            'most_similar_cosmul':(['中国','北京','日本'],lambda:wv.most_similar_cosmul(positive=['北京','日本'],negative=['中国'],topn=10)),
            'doesnt_match':(['经济','股票','财经','苹果'],lambda:wv.doesnt_match(['经济','股票','财经','苹果']))}
        result={}
        for name,(words,fn) in cases.items():
            missing=[w for w in words if w not in wv.key_to_index]
            result[name]={'words':words,'strict_iv':True,'status':'missing_words' if missing else 'ok','missing':missing}
            if not missing:
                value=fn()
                if isinstance(value,list):
                    value=[(word,float(score)) for word,score in value]
                result[name]['value']=value
        output=BASE/'posthoc'/directory.name
        write_json(output/'qualitative_probes.json',{'selected_epoch':summary['selected_epoch'],
            'vectors_sha256':file_hash(directory/'vectors.kv'),'method':summary['spec']['method'],
            'diagnostic_only':True,'not_used_for_selection':True,'probes':result,'created':stamp()})
        print(f'Qualitative APIs saved: {summary["spec"]["method"]}',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    from .runtime import numerical_threads
    with threadpool_limits(limits=numerical_threads()):
        run()
