import numpy as np
import joblib
from scipy import sparse
from gensim_lab import pfr_topic_svm as task
from gensim_lab.common import write_json,read_json,file_hash
from gensim_lab.pfr_topics import TOPICS


def test_actual_same_svc_grid_never_reads_test_labels_and_selects_smallest_tied_c(tmp_path):
    docs=[];rows=[]
    for part,n in [('train',3),('validation',1),('test',1)]:
        for j,topic in enumerate(TOPICS):
            for k in range(n):
                docs.append({'id':f'{part}/{j}/{k}','topic':topic if part!='test' else 'UNREAD_TEST_LABEL'})
                rows.append(np.eye(8)[j])
    ids={part:[d['id'] for d in docs if d['id'].startswith(part+'/')] for part in ['train','validation','test']}
    write_json(tmp_path/'plan.json',{'labels':TOPICS,'document_order':[d['id'] for d in docs],'splits':ids,
          'configuration':{'svm':{'C':[.1,1,10],'class_weight':'balanced','dual':False,'tol':1e-4,'max_iter':20000,'random_state':42}}})
    write_json(tmp_path/'index.json',{'runs':{},'selection':{}})
    folder=tmp_path/'features';folder.mkdir();x=np.asarray(rows,dtype=np.float32)
    sparse.save_npz(folder/'tfidf.npz',sparse.csr_matrix(x))
    for method in ['word2vec','fasttext']:np.save(folder/f'{method}.npy',x)
    write_json(folder/'seal.json',{p.name:file_hash(p) for p in folder.iterdir()})
    task.train(tmp_path,docs)
    index=read_json(tmp_path/'index.json')
    assert len(index['runs'])==9 and len(index['selection'])==3
    assert all(v['C']==.1 for v in index['selection'].values())
    assert not list(tmp_path.glob('runs/*/metrics_test.json'))
    assert not (tmp_path/'selection_frozen.json').exists()
    for v in index['selection'].values():
        model=joblib.load(tmp_path/'runs'/v['run']/'classifier.joblib')
        assert set(model.classes_)==set(TOPICS) and model.dual is False and model.class_weight=='balanced'


def test_prediction_records_full_decision_vector_in_saved_class_order(tmp_path):
    labels=['体育','财经'];write_json(tmp_path/'class_order.json',{'classes':['财经','体育']})
    task.evaluate(tmp_path,'validation',['a'],['体育'],['体育'],np.array([[-2.,-.5]]),labels)
    import pandas as pd
    frame=pd.read_csv(tmp_path/'predictions_validation.csv')
    assert frame.score.iloc[0]==-.5 and frame.decision_财经.iloc[0]==-2
    assert frame.score_kind.iloc[0]=='decision_function'
