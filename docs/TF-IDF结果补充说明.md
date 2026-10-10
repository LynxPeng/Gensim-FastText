# TF-IDF 结果补充说明

## 修改范围

在两份提交文档的 1.1.4 中新增 1.1.4.3“TF-IDF 稀疏表示与留出文章诊断”，原 1.1.4.3–1.1.4.8 顺延为 1.1.4.4–1.1.4.9。加入一幅两面板直方图，其余图号相应顺延。1.1.4 之外的内容及既有 Word2Vec、FastText 结果逐字保留。

## 证据与表述

| 证据 | 功能 | 保存位置 |
| --- | --- | --- |
| 非零特征数、稀疏度和逐篇覆盖分布 | TF-IDF 表示结果 | 新增正文与图 1.3 |
| 零向量数、L2 范数和重载一致性 | 表示可用性核验 | 正文概述、诊断汇总 |
| 每篇文章指标、直方图分箱与来源指纹 | 可追溯记录 | CSV／JSON 资源文件 |
| 词相似度和类比不适用当前文档级表示 | 评价范围 | 正文限定 |

本次仅复用已拟合 TF-IDF 模型，未重训，也未使用类别标签。没有采用历史类别中心诊断的分类准确率，未将覆盖率解释为语义质量或分类性能。正文只保留主要描述指标，逐篇完整数值放在原始记录中。既有资源和覆盖率比较保留，新增分布回答文章间差异，避免只重复总量。

## 局部核查

两份文档分别核对 9 个小节、8 幅图和 35 个本地链接。5,000 篇文章 ID 唯一，数值有限、归一化通过、全部重载向量一致，两个直方图分箱各计入 5,000 篇文章；彩色和灰度预览均通过人工读图与程序布局检查。

| 文档 | 1.1.4 修改前字符数 | 修改后字符数 |
| --- | ---: | ---: |
| Gensim&FastText-修改版.md | 4718 | 6003 |
| Gensim&FastText.md | 4718 | 6003 |

字符数含 Markdown 标记，记录用于追踪本次局部增补。

[诊断汇总](../artifacts/tfidf_submission_diagnostics/928ed691978c06e8cc9d3d1e/summary.json) · [编辑记录](../artifacts/tfidf_submission_diagnostics/928ed691978c06e8cc9d3d1e/report_edit.json) · [交付核查](../artifacts/tfidf_submission_diagnostics/928ed691978c06e8cc9d3d1e/delivery_verification.json)

复现诊断：`conda run -n Gensim python tools/tfidf_submission_diagnostics.py --stage analyze`；绘图按 `preview`、视觉核验、`export` 顺序执行。报告局部编辑入口 `tools/insert_tfidf_submission_results.py` 已完成一次性应用，会阻止重复插入。
