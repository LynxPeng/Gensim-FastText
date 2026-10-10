"""Verify every published plot statistic, epoch point and dimension table."""
import csv
import math
import re
import statistics
from pathlib import Path
from urllib.parse import unquote

from gensim_lab.common import ROOT, file_hash, fingerprint, read_json, stamp, write_json
from gensim_lab.thucnews_dimension_grid import rank_grid


def read_csv(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def check(experiment):
    experiment = Path(experiment)
    plan = read_json(experiment / 'index.json')
    assert plan['status'] == 'complete' and rank_grid(plan['runs'], plan['grid']) == plan['dimension_ranking']
    assert read_json(experiment / 'verification.json')['status'] == 'passed'
    rows = plan['runs']
    summaries = {r['run']: read_json(ROOT / r['run'] / 'summary.json') for r in rows}
    receipt = read_json(experiment / 'figures/approved.json')
    assert len(receipt['figures']) == 8 and receipt['data_fingerprint'] == fingerprint(rows)
    assert all(file_hash(ROOT / p) == h for p, h in receipt['files'].items())
    assert read_json(experiment / 'figures/layout_checks.json')['status'] == 'passed'
    plot_data = read_csv(experiment / 'figures/aggregated_plot_data.csv')
    assert len(plot_data) == 110
    for item in plot_data:
        group = [r for r in rows if r['method'] == item['method'] and r['vector_size'] == int(item['vector_size'])]
        assert len(group) == int(item['n']) == 3
        key = item['metric']
        if '/' in key:
            benchmark, metric = key.split('/')
            values = [next(m[metric] for m in summaries[r['run']]['validation_metrics'] if m['benchmark'] == benchmark)
                      for r in group]
        elif key == 'per_epoch_seconds':
            values = [r['fit_seconds']/r['trained_epochs'] for r in group]
        elif key == 'model_mib':
            values = [r['model_bytes']/2**20 for r in group]
        else:
            field = key if key in ['fit_seconds', 'peak_rss_mib'] else 'validation_score'
            values = [r[field] for r in group]
        assert math.isclose(float(item['mean']), statistics.mean(values), rel_tol=0, abs_tol=1e-12)
        assert math.isclose(float(item['sample_sd']), statistics.stdev(values), rel_tol=0, abs_tol=1e-12)
    points = read_csv(experiment / 'figures/epoch_plot_data.csv')
    assert len(points) == sum(r['trained_epochs'] for r in rows)
    for row in rows:
        actual = [p for p in points if p['method'] == row['method'] and int(p['vector_size']) == row['vector_size']
                  and int(p['seed']) == row['model_seed']]
        epochs = read_json(ROOT / row['run'] / 'epochs.json')
        assert len(actual) == len(epochs)
        for point, epoch in zip(actual, epochs, strict=True):
            assert int(point['epoch']) == epoch['epoch']
            assert float(point['validation_spearman']) == epoch['primary_score']
            assert (point['selected'] == 'True') == (epoch['epoch'] == row['selected_epoch'])
    report = ROOT / 'reports/THUCNews向量维数搜索.md'
    text = report.read_text(encoding='utf-8')
    for method, name in [('word2vec', 'Word2Vec'), ('fasttext', 'Gensim FastText')]:
        body = text.split('## ' + name + '\n', 1)[1].split('\n## ', 1)[0]
        for dimension in plan['grid']['vector_sizes']:
            values = [next(r['validation_score'] for r in rows if r['method'] == method
                           and r['vector_size'] == dimension and r['model_seed'] == s) for s in plan['grid']['seeds']]
            expected = '| ' + str(dimension) + ' | ' + ' | '.join(f'{v:.4f}' for v in values)
            expected += f' | {statistics.mean(values):.4f}±{statistics.stdev(values):.4f} |'
            assert expected in body
    paths = [report, ROOT / 'README.md', ROOT / 'docs/THUCNews向量维数搜索方案.md',
             ROOT / 'reports/Gensim&FastText.md', ROOT / 'reports/Gensim&FastText-去AI味版.md']
    links = 0
    for path in paths:
        content = path.read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content):
            local = unquote(target.strip('<>').split('#', 1)[0])
            if not local or re.match(r'[a-zA-Z]+://', local):
                continue
            assert (path.parent / local).exists(), (path, target)
            links += 1
    result = {'status': 'passed', 'plot_statistics': len(plot_data), 'epoch_points': len(points),
              'dimension_tables': 2, 'figures': 8, 'local_links': links, 'checked': stamp()}
    write_json(experiment / 'publication_checks.json', result)
    print(result)


if __name__ == '__main__':
    base = ROOT / 'artifacts/thucnews_small_search'
    check(ROOT / read_json(base / 'latest.json')['experiment'])
