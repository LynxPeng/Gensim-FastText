"""Independent saved-model, diagnostic-curve, and document delivery checks."""
from __future__ import annotations
import collections
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from threadpoolctl import threadpool_limits

from gensim_lab.common import ROOT,file_hash,read_json,stamp,write_json
from gensim_lab.pfr_topic_svm import BASE,METHODS,audit,load_features
from gensim_lab.pfr_topic_svm_report import FIGURES,HEADING,iterations
from gensim_lab.pfr_topics import load_jsonl
from audit_pfr_topic_native import verify_links


def main():
    directory=ROOT/read_json(BASE/'latest.json')['experiment']
    plan=read_json(directory/'plan.json');index=read_json(directory/'index.json')
    docs=load_jsonl(ROOT/plan['article_file']);lookup={d['id']:d for d in docs}
    audit(directory,docs)
    positions={i:j for j,i in enumerate(plan['document_order'])}
    counts={p:dict(collections.Counter(lookup[i]['topic'] for i in ids)) for p,ids in plan['splits'].items()}
    assert {p:len(ids) for p,ids in plan['splits'].items()}=={'train':5333,'validation':820,'test':1910}
    assert {lookup[i]['month'] for i in plan['splits']['train']}=={'199801','199802'}
    assert {lookup[i]['month'] for i in plan['splits']['validation']+plan['splits']['test']}=={'199803'}
    folder,curve=iterations(directory)
    identity=read_json(folder/'plan.json')
    assert identity['selection_sha256']==file_hash(directory/'selection_frozen.json')
    assert identity['protocol_sha256']==file_hash(ROOT/'gensim_lab/pfr_topic_svm_iterations.py')
    assert identity['features']==read_json(directory/'features/seal.json')
    assert len(curve)==60
    extra_rows=0;final_probes=0
    for method in METHODS:
        features=load_features(directory,method)
        chosen=directory/'runs'/index['selection'][method]['run']
        selected=joblib.load(chosen/'classifier.joblib')
        for part in ['validation','test']:
            frame=pd.read_csv(chosen/f'predictions_{part}.csv',dtype={'id':str})
            x=features[[positions[i] for i in plan['splits'][part]]]
            assert np.array_equal(selected.predict(x),frame.predicted_topic)
            expected=frame[[f'decision_{t}' for t in selected.classes_]].to_numpy()
            assert np.allclose(selected.decision_function(x),expected,atol=1e-12,rtol=0)
            final_probes+=len(frame)
        ids=plan['splits']['validation'];x=features[[positions[i] for i in ids]]
        sub=curve[curve.method==method].sort_values('budget')
        assert sub.budget.tolist()==list(range(1,21))
        for r in sub.itertuples():
            out=directory/r.checkpoint
            assert all(file_hash(out/n)==h for n,h in read_json(out/'seal.json').items())
            model=joblib.load(out/'classifier.joblib')
            assert model.get_params()=={**selected.get_params(),'max_iter':r.budget}
            assert model.n_iter_==r.n_iter and r.n_iter<=r.budget
            frame=pd.read_csv(out/'predictions_validation.csv',dtype={'id':str})
            assert frame.id.tolist()==ids
            truth=[lookup[i]['topic'] for i in ids]
            assert frame.true_topic.tolist()==truth
            predicted=model.predict(x)
            assert np.array_equal(predicted,frame.predicted_topic)
            assert np.allclose(model.decision_function(x),frame[[f'decision_{t}' for t in model.classes_]],atol=1e-12,rtol=0)
            assert np.isclose(r.validation_accuracy,accuracy_score(truth,predicted),atol=1e-12)
            assert np.isclose(r.validation_macro_f1,f1_score(truth,predicted,labels=plan['labels'],average='macro',zero_division=0),atol=1e-12)
            assert not (out/'metrics_test.json').exists()
            if r.budget==20:
                assert r.converged
                assert np.array_equal(model.decision_function(x),selected.decision_function(x))
            extra_rows+=len(frame)
        del features,x,selected
    assert extra_rows==49200 and final_probes==8190
    review=read_json(directory/'figure_preview/visual_review.json')
    assert review['status']=='approved' and set(review['previews'])==set(FIGURES)
    assert all(not r['issues'] for r in read_json(directory/'figure_preview/layout_audit.json').values())
    for name,record in review['previews'].items():
        assert file_hash(directory/f'figure_preview/{name}.png')==record['sha256']
        assert file_hash(directory/f'figure_preview/{name}_grayscale.png')==record['grayscale_sha256']
    assets=0;figures=ROOT/'reports/figures/pfr_topic_svm'/directory.name
    for records in read_json(directory/'figure_export_audit.json').values():
        for name,rec in records.items():
            assert file_hash(figures/name)==rec['sha256']
            assert not any(issue[0] in ['WARN','FAIL'] for issue in rec['issues'])
            assets+=1
    assert assets==15
    links={}
    for path in [ROOT/'reports/Gensim&FastText.md',ROOT/'reports/Gensim&FastText-去AI味版.md',
                 ROOT/'reports/人民日报三种表示LinearSVC对比.md',ROOT/'docs/人民日报主题分类复现指南.md',ROOT/'README.md']:
        links[path.name]=verify_links(path)
        if path.name.startswith('Gensim&FastText'):
            before=(directory/'report_before'/path.name).read_text(encoding='utf-8')
            after=path.read_text(encoding='utf-8')
            assert before.split('### 1.2 ',1)[0]==after.split('### 1.2 ',1)[0]
            assert before.split('## part2 ',1)[1]==after.split('## part2 ',1)[1]
            assert after.count(HEADING)==1 and after.count('![图1.2-10')==1
            assert '不是按epoch' in after and '弱标签' in after
    result={'status':'passed','verified':stamp(),'experiment':directory.name,'splits':{p:len(v) for p,v in plan['splits'].items()},
            'topic_counts':counts,'source_models_unchanged':True,'same_partition_as_native':True,'new_annotation_requests':0,
            'candidate_svc_fits':9,'selected_svc_fits':3,'baseline_metric_rows_recomputed':13110,
            'selected_model_rows_reloaded':final_probes,'diagnostic_fits':60,'diagnostic_prediction_rows_reloaded':extra_rows,
            'diagnostic_test_evaluations':0,'curve_budget20_equals_selected_exact':True,'curve_axis_is_solver_budget_not_epoch':True,
            'first_converged_budget':{m:int(curve[(curve.method==m)&curve.converged].budget.min()) for m in METHODS},
            'figure_assets_verified':assets,'color_and_grayscale_reviewed':True,'report_1_1_and_part2_preserved':True,'local_links':links,
            'uncertainty':'Paired topic-stratified article bootstrap; no training replicate uncertainty'}
    write_json(directory/'delivery_verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
