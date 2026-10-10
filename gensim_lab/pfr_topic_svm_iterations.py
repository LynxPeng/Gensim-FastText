"""Validation accuracy against independent LinearSVC solver iteration budgets.

These are restarted fits, not epochs or live checkpoints of a single fit.
The selected C values and immutable article split are fixed before this diagnostic.
"""
from __future__ import annotations
import gc
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from .common import ROOT,file_hash,fingerprint,read_json,stamp,write_json
from .evaluation_support import save_csv
from .integrity import artifact_lock
from .pfr_topic_svm import BASE,METHODS,load_features,evaluate
from .pfr_topics import load_jsonl
from .runtime import settings

BUDGETS=list(range(1,21))


def run():
    directory=ROOT/read_json(BASE/'latest.json')['experiment']
    plan=read_json(directory/'plan.json');index=read_json(directory/'index.json')
    assert index['status']=='complete'
    for n,h in read_json(directory/'features/seal.json').items():assert file_hash(directory/'features'/n)==h
    assert file_hash(ROOT/plan['article_file'])==plan['article_sha256']
    docs=load_jsonl(ROOT/plan['article_file']);lookup={d['id']:d for d in docs}
    positions={i:j for j,i in enumerate(plan['document_order'])}
    train_ids=plan['splits']['train'];ids=plan['splits']['validation']
    truth=[lookup[i]['topic'] for i in ids];y=[lookup[i]['topic'] for i in train_ids]
    identity={'selection_sha256':file_hash(directory/'selection_frozen.json'),
              'features':read_json(directory/'features/seal.json'),'budgets':BUDGETS,
              'protocol_sha256':file_hash(Path(__file__)),
              'kind':'Independent primal solver budgets; validation only; not epochs'}
    folder=directory/'iteration_diagnostic'/fingerprint(identity)[:24];folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'plan.json',identity)
    rows=[]
    for method in METHODS:
        selected=directory/'runs'/index['selection'][method]['run']
        params=read_json(selected/'spec.json')['parameters']
        full=joblib.load(selected/'classifier.joblib')
        x=load_features(directory,method)
        xt=x[[positions[i] for i in train_ids]];xv=x[[positions[i] for i in ids]]
        for budget in BUDGETS:
            out=folder/method/f'budget_{budget:02d}';out.mkdir(parents=True,exist_ok=True)
            seal_file=out/'seal.json'
            if seal_file.exists():
                assert all(file_hash(out/n)==h for n,h in read_json(seal_file).items())
                record=read_json(out/'summary.json')
            else:
                model=LinearSVC(**{**params,'max_iter':budget})
                start=time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    model.fit(xt,y)
                seconds=time.perf_counter()-start
                warnings_expected=[w for w in caught if issubclass(w.category,ConvergenceWarning)]
                assert len(warnings_expected)==len(caught),'Unexpected training warning'
                predicted=model.predict(xv);decision=model.decision_function(xv)
                write_json(out/'class_order.json',{'classes':model.classes_.tolist()})
                result=evaluate(out,'validation',ids,truth,predicted,decision,plan['labels'])
                joblib.dump(model,out/'classifier.joblib');loaded=joblib.load(out/'classifier.joblib')
                assert np.array_equal(loaded.predict(xv),predicted)
                assert np.array_equal(loaded.decision_function(xv),decision)
                record={'method':method,'C':params['C'],'budget':budget,'n_iter':int(model.n_iter_),
                        'converged':not bool(warnings_expected),'validation_accuracy':result['accuracy'],
                        'validation_macro_f1':result['macro_f1'],'fit_seconds':seconds,
                        'reload_exact':True,'checkpoint':out.relative_to(directory).as_posix()}
                if budget==max(BUDGETS):
                    assert record['converged']
                    assert np.array_equal(predicted,full.predict(xv))
                    assert np.array_equal(decision,full.decision_function(xv))
                    record['final_selected_model_exact']=True
                write_json(out/'summary.json',record)
                write_json(seal_file,{p.name:file_hash(p) for p in out.iterdir() if p.is_file()})
            rows.append(record)
        print(method,'iteration budgets complete; first converged:',min(r['budget'] for r in rows if r['method']==method and r['converged']),flush=True)
        del full,x,xt,xv;gc.collect()
    save_csv(folder/'accuracy_curve.csv',rows)
    assert len(rows)==len(METHODS)*len(BUDGETS)
    write_json(folder/'verification.json',{'status':'passed','completed':stamp(),
        'fits':len(rows),'validation_articles_per_fit':len(ids),'test_evaluations':0,
        'selected_C_unchanged':True,'all_budget20_models_equal_selected':True,
        'all_saved_validation_predictions_reload_exact':True,'not_training_epochs':True})
    write_json(folder/'seal.json',{n:file_hash(folder/n) for n in ['plan.json','accuracy_curve.csv','verification.json']})
    write_json(directory/'iteration_diagnostic/latest.json',{'directory':folder.relative_to(directory).as_posix()})


if __name__=='__main__':
    with artifact_lock(),threadpool_limits(limits=settings()['numerical_threads']):run()
