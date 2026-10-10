"""CSV persistence and optional context-score diagnostics; no training instrumentation."""
from pathlib import Path
import csv
import numpy as np
from gensim.models import Word2Vec
def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)




def score_event_count(n, window, sg):
    if n < 2:
        return 0
    if not sg:
        return n
    return sum(min(window, i) + min(window, n - i - 1) for i in range(n))


def stable_hs_score(model, sentence):
    """Exact log-sigmoid reference for the same IV context events (no lookup/saturation skip)."""
    if model.hs != 1 or model.negative != 0:
        raise ValueError('HS-only scoring')
    ids = [model.wv.key_to_index[t] for t in sentence if t in model.wv.key_to_index]
    if len(ids) > 10000:
        raise ValueError('Reference must respect the builtin 10000-IV-token limit')
    if len(ids) < 2:
        return 0.0, 0, 0
    value, saturated, nodes = 0.0, 0, 0
    for i, target in enumerate(ids):
        contexts = ids[max(0, i-model.window):i] + ids[i+1:i+1+model.window]
        code = model.wv.get_vecattr(model.wv.index_to_key[target], 'code')
        points = model.wv.get_vecattr(model.wv.index_to_key[target], 'point')
        vectors = model.wv.vectors[contexts].astype(np.float64)
        if model.sg == 0:
            vectors = vectors.mean(axis=0, keepdims=True) if model.cbow_mean else vectors.sum(axis=0, keepdims=True)
        z = vectors @ model.syn1[points].astype(np.float64).T
        z *= (1.0 - 2.0 * code)
        value += float((-np.logaddexp(0.0, -z)).sum())
        saturated += int((np.abs(z) >= 6).sum())
        nodes += z.size
    return value, saturated, nodes


def score_sentences(model, docs, part, directory):
    if model.hs != 1 or model.negative != 0 or not isinstance(model, Word2Vec):
        raise ValueError('Only native Gensim Word2Vec HS is eligible for score')
    meta, sentences = [], []
    for doc in docs:
        for i, sentence in enumerate(doc['sentences']):
            n = sum(t in model.wv.key_to_index for t in sentence)
            if n > 10000:
                raise ValueError('Scoring sentence would silently truncate')
            sentences.append(sentence)
            meta.append({'part': part, 'document_id': doc['id'], 'sentence_index': i,
                         'date': doc['date'], 'tokens': len(sentence), 'iv_tokens': n,
                         'oov_tokens': len(sentence)-n, 'events': score_event_count(n, model.window, model.sg)})
    scores = model.score(sentences, total_sentences=len(sentences))
    audit = []
    for row, sentence, val in zip(meta, sentences, scores):
        score = float(val)
        if not np.isfinite(score) or score > 1e-6:
            raise ValueError('Invalid sentence score')
        row.update(builtin_log_score=score, status='ok' if row['events'] else 'no_context',
                   log_score_per_iv_token=score/row['iv_tokens'] if row['events'] else None,
                   log_score_per_event=score/row['events'] if row['events'] else None)
        if row['events'] and len(audit) < 100:
            ref, sat, nodes = stable_hs_score(model, sentence)
            audit.append({**row, 'stable_log_score': ref, 'builtin_minus_stable': score-ref,
                          'saturated_nodes_skipped_by_builtin': sat, 'hs_nodes': nodes})
    save_csv(directory / f'sentence_scores_{part}.csv', meta)
    save_csv(directory / f'score_numerical_audit_{part}.csv', audit)
    return {'sentences': len(meta), 'valid_context_sentences': sum(r['events'] > 0 for r in meta),
            'oov_tokens': sum(r['oov_tokens'] for r in meta), 'audited': len(audit),
            'max_abs_reference_difference': max((abs(r['builtin_minus_stable']) for r in audit), default=0),
            'saturated_nodes': sum(r['saturated_nodes_skipped_by_builtin'] for r in audit)}


