import json
from pathlib import Path
import numpy as np
from gensim.models import Word2Vec, FastText
from gensim.models.word2vec import LineSentence
from gensim_lab.thucnews import save_training_model
from gensim_lab.thucnews_tfidf import fit_dictionary


def test_native_parallel_queue_and_checkpoint(tmp_path):
    path=tmp_path/'queue.txt'
    path.write_text('中国 经济 市场 新闻 科技\n'*10000,encoding='utf-8',newline='\n')
    for cls in [Word2Vec,FastText]:
        extra={'min_n':1,'max_n':3,'bucket':1000} if cls is FastText else {}
        model=cls(vector_size=8,min_count=1,workers=4,batch_words=1000,hs=1,negative=0,
                  sample=0,shrink_windows=False,**extra)
        model.build_vocab(corpus_file=str(path))
        effective,raw=model.train(corpus_iterable=LineSentence(str(path)),total_examples=10000,total_words=50000,epochs=1)
        assert effective==raw==50000 and model.workers==4 and not model.compute_loss
        checkpoint=tmp_path/f'{cls.__name__}.gensim'
        save_training_model(model,checkpoint)
        loaded=cls.load(str(checkpoint))
        assert np.array_equal(loaded.wv.vectors,model.wv.vectors)
        assert np.array_equal(loaded.syn1,model.syn1)
        if cls is FastText:
            assert np.array_equal(loaded.wv.vectors_ngrams,model.wv.vectors_ngrams)
            assert np.array_equal(loaded.wv['尚未出现的新词'],model.wv['尚未出现的新词'])


def test_tfidf_counts_articles_and_excludes_holdout(tmp_path):
    path=tmp_path/'documents.jsonl'
    rows=[{'part':'train','sentences':[['中国','中国','新闻'],['中国','市场']]},
          {'part':'train','sentences':[['中国','新闻']]},
          {'part':'holdout','sentences':[['只在留出','只在留出']]}]
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    dictionary=fit_dictionary(path,{'workers':2,'no_below':1,'no_above':1.0,'keep_n':100},2)
    assert dictionary.num_docs==2 and dictionary.num_pos==7
    assert '只在留出' not in dictionary.token2id
    assert dictionary.dfs[dictionary.token2id['中国']]==2
    assert dictionary.cfs[dictionary.token2id['中国']]==4
