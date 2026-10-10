"""Article-frequency TF-IDF fitted in parallel on training articles only."""
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import time
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from threadpoolctl import threadpool_limits
from .common import ROOT, PeakMemory, file_hash, fingerprint, stamp, write_json
from .runtime import numerical_threads
from .evaluation_support import save_csv
from .thucnews_documents import chunks, ordered_results
from gensim.models.word2vec import LineSentence


def counts(lines):
    df, cf = collections.Counter(), collections.Counter()
    positions, nnz = 0, 0
    for line in lines:
        document = json.loads(line)
        tokens = [word for sentence in document['sentences'] for word in sentence]
        occurrences = collections.Counter(tokens)
        df.update(occurrences.keys())
        cf.update(occurrences)
        positions += len(tokens)
        nnz += len(occurrences)
    return df, cf, len(lines), positions, nnz


def fit_dictionary(path, parameters, expected):
    df, cf, n, positions, nnz = collections.Counter(), collections.Counter(), 0, 0, 0
    workers = parameters['workers']
    with ProcessPoolExecutor(max_workers=workers) as executor:
        last = time.monotonic()
        for a,b,c,d,e in ordered_results(executor, counts, chunks(path,'train'), workers*2):
            df.update(a)
            cf.update(b)
            n, positions, nnz = n+c, positions+d, nnz+e
            if time.monotonic()-last > 15:
                print(f'TF-IDF document frequencies {n}/{expected}',flush=True)
                last=time.monotonic()
    assert n == expected
    dictionary = Dictionary()
    dictionary.token2id = {word:index for index,word in enumerate(sorted(df))}
    dictionary.dfs = {index:df[word] for word,index in dictionary.token2id.items()}
    dictionary.cfs = {index:cf[word] for word,index in dictionary.token2id.items()}
    dictionary.num_docs, dictionary.num_pos, dictionary.num_nnz = n, positions, nnz
    dictionary.filter_extremes(**{key:parameters[key] for key in ['no_below','no_above','keep_n']})
    assert len(dictionary) and dictionary.num_docs == n
    return dictionary


def worker(directory, spec, manifest):
    started=time.perf_counter()
    with threadpool_limits(limits=numerical_threads()), PeakMemory() as memory:
        dictionary=fit_dictionary(ROOT/manifest['directory']/'documents.jsonl',spec['dictionary'],manifest['partitions']['train']['documents'])
        assert dictionary.num_pos == manifest['partitions']['train']['tokens']
        model=TfidfModel(dictionary=dictionary,normalize=True)
        fit_seconds=time.perf_counter()-started
        dictionary.save(str(directory/'dictionary.gensim'))
        model.save(str(directory/'model.gensim'))
        loaded_dictionary=Dictionary.load(str(directory/'dictionary.gensim'))
        loaded=TfidfModel.load(str(directory/'model.gensim'))
        assert loaded.idfs == model.idfs and loaded_dictionary.token2id == dictionary.token2id
        assert loaded_dictionary.dfs == dictionary.dfs and loaded_dictionary.cfs == dictionary.cfs
        save_csv(directory/'training_vocabulary.csv',[{'token':word,'id':index,'df':dictionary.dfs[index],
                  'cf':dictionary.cfs[index],'idf':model.idfs[index]} for word,index in sorted(dictionary.token2id.items(),key=lambda item:item[1])])
        frequencies=collections.Counter(t for s in LineSentence(str(ROOT/manifest['directory']/'holdout.txt')) for t in s)
        total=sum(frequencies.values()); iv=sum(n for t,n in frequencies.items() if t in dictionary.token2id)
        coverage={'tokens':total,'iv_tokens':iv,'oov_tokens':total-iv,
                  'vocabulary_token_coverage':iv/total,'usable_vector_token_coverage':iv/total,'generated_oov_tokens':0,
                  'score_kind':'not_applicable_tfidf_document_representation'}
        write_json(directory/'heldout_summary.json',coverage)
        write_json(directory/'save_load_check.json',{'dictionary_exact':True,'dfs_and_cfs_exact':True,'idfs_exact':True,
                   'no_classification_head':True})
        epoch={'epoch':1,'epoch_seconds':fit_seconds,'effective_words':dictionary.num_pos,'raw_words':dictionary.num_pos,
               'representation':'statistical_document_tfidf'}
        write_json(directory/'epochs.json',[epoch])
        save_csv(directory/'epochs.csv',[epoch])
        write_json(directory/'metrics.json',[])
        write_json(directory/'queries.json',{'kind':'tfidf_document_representation','word_vector_benchmarks':'not_applicable'})
        summary={'spec':spec,'completed':stamp(),'fit_seconds':fit_seconds,
                 'elapsed_this_process_seconds':time.perf_counter()-started,'vocabulary_size':len(dictionary),
                 'vocabulary_sha256':fingerprint(sorted(dictionary.token2id)),'metrics':[],'heldout':coverage,
                 'model_bytes':sum((directory/name).stat().st_size for name in ['dictionary.gensim','model.gensim']),
                 'representation_kind':'sparse document TF-IDF; no neural word embeddings'}
    summary['peak_rss_mib']=memory.peak/2**20
    write_json(directory/'summary.json',summary)
    files={path.name:file_hash(path) for path in directory.iterdir() if path.is_file() and path.name not in {'run.log','seal.json'}}
    write_json(directory/'seal.json',{'spec_fingerprint':fingerprint(spec),'files':files,'sealed':stamp()})
    print(f'Completed tfidf: {len(dictionary)} terms',flush=True)
