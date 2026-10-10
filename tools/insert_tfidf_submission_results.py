"""Add TF-IDF results locally while preserving manually edited report prose."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gensim_lab.common import ROOT, file_hash, read_json, stamp, write_json
from tools.tfidf_submission_diagnostics import QA, SOURCE


def main():
    result = read_json(QA / 'summary.json')
    source = read_json(SOURCE / 'summary.json')
    assert read_json(QA / 'visual_review.json')['status'] == 'approved'
    assert len(read_json(QA / 'figure_audit.json')['exports']) == 2
    for path, digest in result['sources'].items():
        assert file_hash(ROOT / path) == digest
    n = result['nonzero_features']
    section = f'''##### 1.1.4.3 TF-IDF 稀疏表示与留出文章诊断

TF-IDF 在 821,609 篇训练文章上拟合词表与 IDF，最终保留 50,000 个特征。它按文档统计词频和文档频次，不经过梯度更新，因此无需逐轮选取检查点。本次复用已经保存的词典和模型，将同一批 5,000 篇留出文章映射到固定特征空间；留出文章不参与词表筛选或 IDF 计算。

留出文章共包含 {result['word_occurrences']:,} 词次，其中 {result['retained_word_occurrences']:,} 词次进入词表，整体覆盖率为 {result['weighted_vocabulary_percent']:.2f}%。每篇文章的非零特征数中位数为 {n['median']:.0f}，四分位区间为 {n['q25']:.0f}–{n['q75']:.0f}，均值为 {n['mean']:.2f}。相对于 50,000 维特征空间，中位数对应的非零比例仅为 {result['density_percent']['median']:.2f}%，说明单篇文章只激活词表中的少量特征。图 1.3 展示逐篇分布，避免用一个总体覆盖率掩盖文章之间的差异。

![图 1.3 TF-IDF 在全部 5,000 篇留出文章上的表示诊断。a：每篇文章的非零特征数；b：每篇文章进入词表的词次占比。纵轴均为文章数，完整保留观测范围；这些文章属于同一次模型运行，不作为独立训练重复。](figures/section11/tfidf_document_diagnostics.png)

[SVG 矢量图](figures/section11/tfidf_document_diagnostics.svg)

全部留出文章均得到非零表示，零向量比例为 0%。L2 归一化后的向量范数均为 1，最大绝对偏差为 {result['max_normalization_error']:.2e}；重新加载模型后，全部文章的稀疏向量逐项一致。统计模型拟合耗时 {source['fit_seconds']:.1f} 秒，历史主进程峰值 RSS 为 {source['peak_rss_mib']:.1f} MiB，词典与模型合计 {source['model_bytes']/2**20:.1f} MiB。这些资源与覆盖指标可和词向量方法并列比较；词相似度、词类比及上下文评分针对词向量，不适用于当前文档级 TF-IDF。覆盖率和稀疏度描述表示是否可用及其存储特征，不能直接证明语义质量或分类性能。

[逐篇原始指标](../artifacts/tfidf_submission_diagnostics/{SOURCE.name}/holdout_documents.csv)、[诊断汇总与来源核验](../artifacts/tfidf_submission_diagnostics/{SOURCE.name}/summary.json)、[图表核验](../artifacts/tfidf_submission_diagnostics/{SOURCE.name}/figure_audit.json)、[拟合记录](../artifacts/thucnews/runs/{SOURCE.name}/summary.json)、[词表及 IDF](../artifacts/thucnews/runs/{SOURCE.name}/training_vocabulary.csv)。

'''
    edits = []
    backup = QA / 'report_before'
    backup.mkdir(exist_ok=True)
    for name in ['Gensim&FastText-去AI味版.md', 'Gensim&FastText.md']:
        path = ROOT / 'reports' / name
        before = path.read_bytes()
        text = before.decode('utf-8')
        assert 'TF-IDF 稀疏表示与留出文章诊断' not in text, 'Already updated; avoid duplicate insertion.'
        newline = '\r\n' if '\r\n' in text else '\n'
        start = text.index('#### 1.1.4 训练结果与评估')
        end = text.index('### 1.2 ', start)
        old = text[start:end]
        updated = re.sub(r'(##### 1\.1\.4\.)([3-8])\b',
                         lambda m: m[1] + str(int(m[2]) + 1), old)
        updated = re.sub(r'(图 1\.)([3-7])\b',
                         lambda m: m[1] + str(int(m[2]) + 1), updated)
        marker = '##### 1.1.4.4 训练耗时、峰值内存与模型大小'
        assert updated.count(marker) == 1
        updated = updated.replace(marker, section.replace('\n', newline) + marker, 1)
        new = text[:start] + updated + text[end:]
        assert new[:start] == text[:start]
        assert new[new.index('### 1.2 ', start):] == text[end:]
        for identifier, next_identifier in [('1', '2'), ('2', '4')]:
            old_part = old[old.index(f'##### 1.1.4.{identifier} '):]
            old_part = old_part[:old_part.index(f'##### 1.1.4.{int(identifier)+1} ')]
            new_part = updated[updated.index(f'##### 1.1.4.{identifier} '):]
            new_part = new_part[:new_part.index(f'##### 1.1.4.{"2" if identifier=="1" else "3"} ')]
            assert old_part == new_part, 'Existing model prose changed.'
        headings = re.findall(r'^##### 1\.1\.4\.(\d+) ', updated, flags=re.M)
        assert headings == [str(i) for i in range(1, 10)], headings
        assert file_hash(path) == __import__('hashlib').sha256(before).hexdigest()
        (backup / name).write_bytes(before)
        path.write_bytes(new.encode('utf-8'))
        edits.append({'path': str(path.relative_to(ROOT)), 'before_sha256': file_hash(backup/name),
                      'after_sha256': file_hash(path), 'section_characters_before': len(old),
                      'section_characters_after': len(updated), 'outside_section_preserved': True,
                      'existing_neural_results_preserved': True})
        print('Updated', path)
    write_json(QA/'report_edit.json', {'updated': stamp(), 'edits': edits,
               'added_section': '1.1.4.3 TF-IDF 稀疏表示与留出文章诊断',
               'generator_sha256': file_hash(Path(__file__))})


if __name__ == '__main__':
    main()
