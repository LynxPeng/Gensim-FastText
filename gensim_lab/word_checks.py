"""Independently recompute archived intrinsic metrics from item-level CSVs."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from .common import ROOT,read_json,write_json,stamp,file_hash
from .word_evaluation import BASE


def numeric_csv(path):
    return pd.read_csv(path,encoding='utf-8-sig',float_precision='round_trip')


def run():
    selected=read_json(BASE/'evaluated_models.json')
    results=read_json(BASE/'metrics.json')
    checks=[]
    for model in selected:
        directory=BASE/'evaluations'/model['id']
        metadata=read_json(directory/'metadata.json')
        assert file_hash(ROOT/model['directory']/'bundle.joblib')==metadata['spec']['model_sha256']
        for row in read_json(directory/'metrics.json'):
            assert row['covered']+row['oov_count']==row['total']
            if row['kind']=='similarity':
                items=numeric_csv(directory/f"{row['benchmark']}_pairs.csv")
                valid=items[items.status=='ok']
                assert len(items)==row['total'] and len(valid)==row['covered']
                if row['status']=='ok':
                    assert np.allclose(pearsonr(valid.human_similarity,valid.cosine_similarity),[row['pearson_r'],row['pearson_p']])
                    assert np.allclose(spearmanr(valid.human_similarity,valid.cosine_similarity),[row['spearman_rho'],row['spearman_p']])
            else:
                items=numeric_csv(directory/f"{row['benchmark']}_questions.csv")
                assert len(items)==row['total'] and int(items.correct.sum())==row['correct']
                assert np.isclose(row['all_accuracy'],row['correct']/row['total'])
            checks.append({'model':model['id'],'benchmark':row['benchmark'],'passed':True})
    assert len(checks)==len(results)
    write_json(BASE/'verification.json',{'status':'complete','completed':stamp(),'checks':checks,'models':len(selected)})
    print(f'Intrinsic metrics independently verified: {len(checks)}')


if __name__=='__main__':
    run()
