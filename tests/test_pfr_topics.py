import json
import numpy as np
import pytest
from gensim_lab import pfr_topics as task


def test_payload_does_not_expose_source_ids_or_page_metadata():
    doc = {'id': 'anonymous', 'source_id': '19980101-08-001', 'month': '199801',
           'annotation_text': '运动员在比赛中取得了冠军。', 'page_label': '08'}
    body = task.request_payload([doc], {'model': 'deepseek-flash', 'temperature': 0, 'max_tokens': 1024})
    encoded = json.dumps(body, ensure_ascii=False)
    assert '19980101-08-001' not in encoded and 'page_label' not in encoded
    assert body['thinking']['type'] == 'disabled'
    assert json.loads(body['messages'][1]['content'])['articles'] == [{'id': 'anonymous', 'text': doc['annotation_text']}]


def test_evidence_must_be_contiguous_original_and_topic_valid():
    doc = {'id': 'a', 'annotation_text': '北京举办了一场精彩的田径比赛。\n运动员表现出色。'}
    item = {'id': 'a', 'status': 'ok', 'topic': '体育', 'confidence': 'high',
            'reason': '报道田径比赛', 'evidence': '一场精彩的田径比赛。'}
    assert task.validate_item(item, doc) == item
    with pytest.raises(ValueError, match='original'):
        task.validate_item(dict(item, evidence='运动员获得了奥运金牌'), doc)
    with pytest.raises(ValueError, match='topic'):
        task.validate_item(dict(item, topic='08'), doc)
    assert task.validate_item(dict(item,status='out_of_scope',topic=None,evidence=''),doc)['topic'] is None


def test_saved_quote_recovery_does_not_invent_a_new_label_or_call_api(tmp_path,monkeypatch):
    doc={'id':'a','source_id':'origin','annotation_text':'企业完成资产重组，通过资源整合的方式建立新的航运公司。'}
    item={'id':'a','status':'ok','topic':'财经','confidence':'high','reason':'企业重组',
          'evidence':'企业完成资产重组，通过资源整合的方式建立新的航运公司并提高利润'}
    task.append_jsonl(tmp_path/'articles.jsonl',doc)
    task.append_jsonl(tmp_path/'requests.jsonl',{'ids':['a'],'status':200,'finish_reason':'stop',
         'content':json.dumps({'items':[item]},ensure_ascii=False),'prompt_sha256':'p','response_id':'r','model':'deepseek-flash'})
    monkeypatch.setattr(task,'call_api',lambda *args:pytest.fail('recovery made a paid call'))
    task.recover_saved_outputs(tmp_path)
    saved=task.load_jsonl(tmp_path/'annotations.jsonl')[0]
    assert saved['topic']==item['topic']
    assert task.normalized(saved['evidence']) in task.normalized(doc['annotation_text'])
    assert len(saved['evidence'])>=20
    assert saved['evidence_original']==item['evidence']


def test_long_quote_recovery_preserves_topic_and_prior_audit(tmp_path):
    text='中共中央和国务院领导同志参加代表团审议政府工作报告。'*6
    doc={'id':'a','source_id':'origin','annotation_text':text}
    item={'id':'a','status':'ok','topic':'政治','confidence':'high','reason':'审议报告','evidence':text}
    task.append_jsonl(tmp_path/'articles.jsonl',doc)
    task.append_jsonl(tmp_path/'requests.jsonl',{'ids':['a'],'status':200,'finish_reason':'stop',
         'content':json.dumps({'items':[item]},ensure_ascii=False),'prompt_sha256':'p','response_id':'r','model':'deepseek-flash'})
    task.write_json(tmp_path/'evidence_recovery.json',{'recovered':[{'id':'earlier','repair':'retained'}]})
    task.recover_saved_outputs(tmp_path)
    saved=task.load_jsonl(tmp_path/'annotations.jsonl')[0]
    assert saved['topic']=='政治' and saved['evidence']==text[:40]
    assert saved['evidence_original']==text
    assert len(task.read_json(tmp_path/'evidence_recovery.json')['recovered'])==2


def test_tariff_uses_cache_hit_and_miss_counts():
    usage = {'prompt_tokens': 2000, 'prompt_cache_hit_tokens': 1000,
             'prompt_cache_miss_tokens': 1000, 'completion_tokens': 100}
    assert task.usage_cost(usage) == pytest.approx(.00142)
    assert task.usage_cost(usage, True) == pytest.approx(.00284)


def test_march_is_held_out_and_train_keeps_both_months():
    docs=[]
    for j, topic in enumerate(task.TOPICS):
        for month, n in [('199801', 4), ('199802', 4), ('199803', 20)]:
            for i in range(n):
                key=f'{month}_{j}_{i}'
                docs.append({'id': key, 'month': month, 'topic': topic, 'group': key, 'content_hash': key})
    cfg={'train_months': ['199801','199802'], 'holdout_month': '199803',
         'split_seed':42, 'validation_fraction':.3, 'topics':task.TOPICS}
    splits, excluded = task.make_splits(docs, cfg)
    lookup={d['id']:d for d in docs}
    assert len(splits['train'])==64 and not excluded
    assert len(splits['validation'])==48 and len(splits['test'])==112
    assert all(lookup[i]['month']=='199803' for part in ['validation','test'] for i in splits[part])
    assert task.make_splits(docs, cfg)[0] == splits


def test_stratification_counts_stay_attached_to_groups():
    docs=[]
    sizes=[127,98,55,43,36,185,29,62]
    for topic,n in zip(task.TOPICS,sizes):
        for month,count in [('199801',4),('199802',4),('199803',n)]:
            for i in range(count):
                key=f'{month}/{topic}/{i}'
                docs.append({'id':key,'month':month,'topic':topic,'group':key,'content_hash':key})
    cfg={'train_months':['199801','199802'],'holdout_month':'199803',
         'split_seed':42,'validation_fraction':.3,'topics':task.TOPICS}
    splits,_=task.make_splits(docs,cfg)
    lookup={d['id']:d for d in docs}
    for topic,n in zip(task.TOPICS,sizes):
        count=sum(lookup[i]['topic']==topic for i in splits['validation'])
        assert abs(count-.3*n)<=3


def test_resume_uses_successful_annotations_once(tmp_path, monkeypatch):
    # No network or credentials: a sealed success must not generate another request.
    doc={'id':'a','annotation_text':'运动员参加了比赛。'}
    item={'id':'a','status':'ok','topic':'体育','confidence':'high','reason':'比赛','evidence':'运动员参加了比赛。'}
    task.append_jsonl(tmp_path/'articles.jsonl',doc)
    task.append_jsonl(tmp_path/'annotations.jsonl',item)
    task.append_jsonl(tmp_path/'requests.jsonl',{'ids':['a'],'cost_cny':0.0,'status':200})
    monkeypatch.setenv('DEEPSEEK_API_KEY','test-not-a-real-key')
    monkeypatch.setattr(task,'call_api',lambda *args: pytest.fail('successful article was annotated again'))
    task.annotate(tmp_path)
    assert len(task.load_jsonl(tmp_path/'annotations.jsonl'))==1


def test_budget_prevents_dispatching_reserved_requests(tmp_path, monkeypatch):
    doc={'id':'a','annotation_text':'运动员参加了比赛。'}
    task.append_jsonl(tmp_path/'articles.jsonl',doc)
    cfg=task.configuration(); cfg['annotation']['budget_cny']=.000001
    monkeypatch.setenv('DEEPSEEK_API_KEY','test-not-a-real-key')
    monkeypatch.setattr(task,'configuration',lambda:cfg)
    monkeypatch.setattr(task,'call_api',lambda *args: pytest.fail('over-budget request dispatched'))
    with pytest.raises(RuntimeError,match='budget'):
        task.annotate(tmp_path)
