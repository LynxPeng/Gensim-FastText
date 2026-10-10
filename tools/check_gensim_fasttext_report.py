from pathlib import Path
import hashlib
import re
from collections import Counter
from inline_math_text import plain_inline_math

root = Path(__file__).resolve().parents[1]
source_path = root / 'reports/Gensim&FastText-修改版.md'
target_path = root / 'reports/Gensim&FastText-终稿.md'
appendix_path = root / 'reports/Gensim&FastText-附录.md'
source = source_path.read_text(encoding='utf-8-sig')
main = target_path.read_text(encoding='utf-8')
appendix = appendix_path.read_text(encoding='utf-8')
target = main + '\n' + appendix
assert plain_inline_math(main)[1] == 0, 'Inline LaTeX remains in the final report.'

source_images = re.findall(r'^!\[.*\]\((.*)\)$', source, re.M)
target_images = re.findall(r'^!\[.*\]\((.*)\)$', target, re.M)
assert Counter(source_images) == Counter(target_images)
assert len(target_images) == len(set(target_images)) == 37
assert all((target_path.parent / path).is_file() for path in target_images)
links = re.findall(r'(?<!!)\[[^\n]*?\]\(([^\n]*?)\)', target)
for link in links:
    assert link.startswith('https://') or (target_path.parent / link).is_file(), link
assert '(Gensim&FastText-附录.md)' in main
assert '(Gensim&FastText-终稿.md)' in appendix
assert '## 附录 A ' not in main
assert '## 摘要\n' in main
assert 'https://github.com/LynxPeng/Gensim-FastText' in main
assert not re.search(r'<!--|\bpart[123]\b|SVG|\.svg|artifacts/|\.csv\b|\.json\b|\.log\b|C:/Projects|原始结果|原始指标|原始数据', target)

in_code = False
parent = 0
expected_fourth = expected_fifth = expected_sixth = 0
third_label = ''
headings = []
for lineno, line in enumerate(target.splitlines(), 1):
    if line.startswith('```'):
        in_code = not in_code
        continue
    if in_code:
        continue
    heading = re.match(r'^(#{1,6}) (.+)$', line)
    if not heading:
        continue
    level, text = len(heading[1]), heading[2]
    assert level <= parent + 1 or parent == 0, (lineno, text, parent)
    parent = level
    headings.append((lineno, level, text))
    if level <= 3:
        expected_fourth = expected_fifth = expected_sixth = 0
        if level == 3:
            third_label = text.split()[0]
    elif level == 4:
        expected_fourth += 1
        expected_fifth = expected_sixth = 0
        assert text.startswith(f'{third_label}.{expected_fourth} '), (lineno, text)
    elif level == 5:
        expected_fifth += 1
        expected_sixth = 0
        assert text.startswith(f'{expected_fifth}. '), (lineno, text)
    elif level == 6:
        expected_sixth += 1
        assert text.startswith(chr(96 + expected_sixth) + ') '), (lineno, text)
assert not in_code

source_code = re.findall(r'```(?:python|cython)\n(.*?)\n```', source, re.S)
target_code = re.findall(r'```(?:python|cython)\n(.*?)\n```', target, re.S)
assert len(source_code) == len(target_code) == 13
assert source_code == target_code
implementation = main.split('## 3 ', 1)[1]
assert len(re.findall(r'```(?:python|cython)\n', implementation)) == 13
assert '## 附录 E ' not in target
assert '附录 E' not in target
assert target.count('\n$$\n') % 2 == 0
assert re.findall(r'\\tag\{(\d+)\}', target) == ['1', '2', '3', '4', '5']

source_tables = re.findall(r'(?:^\|[^\n]*\n)+', source, re.M)
target_tables = re.findall(r'(?:^\|[^\n]*\n)+', target, re.M)
assert len(target_tables) == len(source_tables) + 1 == 23

def numbers_in_rows(table):
    # Ignore rewritten headers, normalize thousands separators, preserve exact decimals.
    rows = table.splitlines()[2:]
    return [re.findall(r'-?\d+(?:\.\d+)?', row.replace(',', '')) for row in rows]

remaining_tables = [numbers_in_rows(t) for t in target_tables]
for old in source_tables:
    numbers = numbers_in_rows(old)
    assert numbers in remaining_tables, numbers
    remaining_tables.remove(numbers)
assert len(remaining_tables) == 1

for table in target_tables:
    rows = table.strip().splitlines()
    widths = [len(row.split('|')) for row in rows]
    assert len(set(widths)) == 1, (widths, rows[0])

for phrase in ('[−0.45, 2.99]', '[-3.64, 0.78]', '[-4.43, 0.03]', '[-2.30, 1.04]', '未经独立人工核验'):
    if phrase == '未经独立人工核验':
        assert '主题正确性仍未经过独立人工核验' in main
    else:
        assert phrase in main
assert '当前测试子集仍属探索性评价' in main
assert '单种子运行' in main
assert '未按候选配置启动全量重训' not in target
assert '候选参数全量重训与表示比较' not in target or '候选配置全量重训与表示比较' in target

main_figures = len(re.findall(r'^!\[', main, re.M))
caption_count = len(re.findall(r'^图 [A-D]?\d+\. ', target, re.M))
table_caption_count = len(re.findall(r'^表 [A-D]?\d+\. ', target, re.M))
assert caption_count == 37
assert table_caption_count == 23

def region(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]

counts = [
    ('无监督表示', len(region(source, '### 1.1 ', '### 1.2 ')), len(region(target, '### 1.1 ', '### 1.2 '))),
    ('八类主题分类', len(region(source, '### 1.2 ', '## part2 ')), len(region(target, '### 1.2 ', '## 2 '))),
    ('模型原理', len(region(source, '## part2 ', '## part3 ')), len(region(target, '## 2 ', '## 3 '))),
    ('实现分析', len(source.split('## part3 ', 1)[1]), len(region(main, '## 3 ', '## 4 '))),
]

notes = '''# Gensim 与 FastText 终稿润色说明

处理模式：报告终稿。正文保持中文及克制表述；原稿保留，新稿另存。H1 为报告标题，正文层级从 H2 开始计算。

## 修改范围

- 正文四级标题即 Markdown H5，改为父节内的 `1.、2.、3.`；正文五级标题即 H6，改为 `a)、b)、c)`。H4 恢复 `1.1.1` 等完整章节编号，补齐原先跳级的标题层次。
- 删除运行索引、CSV／JSON、日志、配置、源码路径及 SVG 下载等过程链接，保留正文图像与论文引用。
- 明确区分全量 CBOW＋HS 基线、固定子集参数搜索和候选配置全量重训，清理搜索阶段尚未重训与小结已经重训之间的时态冲突。
- 主文保留主要比较、资源成本和影响结论的限定；逐种子明细、辅助曲线、逐主题完整指标及分歧案例移入附录 A–D。
- 第三部分恢复 13 段关键源码，按对应机制就近展示，保留固定版本文件名与行号，不再另设源码附录。
- 增加摘要，交代表示学习、参数搜索、主题分类及源码分析的范围，保留自动标签与性能排序的不确定性。
- 附录 A–D 拆为独立 Markdown，正文与附录互相链接；另列数据来源、固定版本工具和 GitHub 公开仓库。
- 按作者当前终稿追加第 4 章，说明小规模选参的全量外推限制、未充分验证的参数及其交互，以及不同类别数量的实验缺口。
- 两份编辑稿统一命名为“修改版”，同步更新引用与处理入口；生成脚本默认保护已有终稿，显式传入 `--overwrite` 才会重建。
- 终稿的行内 LaTeX 改为普通字母、下划标记和可直接阅读的算式，独立公式块与源码片段保留。
- 统一图表编号，以短图像替代文本配一条可见图注；不改动图像文件、原有表格数值或源码片段。
- 删除请求费用、调用预算、测试日志条目及反复提示读者查看本地记录的措辞；压缩讲义式定义、重复总结和过长案例。

## 术语约定

| 术语 | 统一用法 |
| --- | --- |
| Gensim FastText | Gensim 4.4.0 无监督子词词向量模型 |
| 原生 FastText | fastText 0.9.2 原生实现 |
| 字符子词／词 n-gram | 分别指词内部特征／相邻词组合 |
| 最佳检查点／停止轮次 | 区分验证最高分模型和实际训练终点 |
| 下采样、向量维数 | 统一原文的术语变体 |
| 训练标准差／bootstrap 区间 | 分别描述训练波动和条件性文章抽样不确定性 |

## 主张与证据位置

| 主张 | 证据 | 位置与限定 |
| --- | --- | --- |
| 参数选择随模型与任务变化 | 固定子集的三个种子验证结果 | 正文 1.1；逐次结果和辅助评价在附录 A。未检验完整参数交互 |
| 预训练初始化改善较低预算验证表现 | 两条初始化路线的独立预算实验 | 正文 1.2；保留额外预训练数据成本的限定 |
| 预训练路线测试均值较高 | 三种子测试与种子 42 配对 bootstrap | 正文 1.2；配对区间跨零，未确认稳定优势 |
| TF-IDF 冻结分类点估计最高 | 同一 LinearSVC 的测试及配对差值 | 正文 1.2；三组差值区间跨零，未宣称等效或稳定排序 |
| Gensim 不同分支存在尺度与更新差异 | 固定版本源码和已有微型探针 | 正文第 3 节直接展示代码，未推断独立的全量性能贡献 |

## 正文与附录分配

| 内容 | 处理 |
| --- | --- |
| 核心比较和结论边界 | 保留正文 |
| 逐种子数值、训练轨迹、次要指标 | 附录 A |
| 全量基线的完整图示、近邻与评分诊断 | 附录 B |
| 主题定义、标注技术细节、原生逐类指标与案例 | 附录 C |
| SVC 迭代诊断、逐类指标与案例 | 附录 D |
| 13 段固定版本代码片段 | 正文第三部分对应小节 |
| 过程文件导航及调用账单 | 从成稿删除，原文件及资源保留 |

## 篇幅对照

以下按节统计 Unicode 字符数，包含 Markdown 结构与图表文字，移入附录的内容不再计入对应正文节。

| 正文部分 | 原稿字符数 | 终稿字符数 |
| --- | ---: | ---: |
'''
for label, before, after in counts:
    notes += f'| {label} | {before:,} | {after:,} |\n'
notes += f'''
## 完成核验

- 原稿 SHA-256：`{hashlib.sha256(source_path.read_bytes()).hexdigest()}`。
- 原稿 37 幅图全部保留；正文 {main_figures} 幅、附录 {37 - main_figures} 幅，文件路径全部有效，无重复引用。
- 原稿 22 张表的逐行数值全部保留，新增一张候选全量模型汇总表，共 23 张。表格列数一致。
- 13 段 Python／Cython 代码与原稿逐字一致，全部在正文第三部分；构造文本示例在附录 A。
- 标题层级、H4 章节编号、H5／H6 序号、图表编号、数学显示块和公式编号通过结构检查。
- 正文与附录互链有效；资源链接仅指向数据、工具和项目来源，无过程文件路径、SVG 下载链接或生成标记。
- 自动标签、历史观察、单种子全量训练及配对区间跨零等必要限定保留在正文。

新增汇总表与全量最佳轮次依据本项目既有的候选参数全量重训练报告核对，未重新训练或计算实验。两篇论文的分析沿用原稿，未扩展引文范围，也未执行新的文献或统计审查。

## 生成与核验入口

在项目根目录使用 Gensim 环境运行以下命令。核验脚本检查当前终稿与附录的标题、图像依赖、数值及源码一致性，并更新本文档。终稿已有作者手工修改，生成脚本默认拒绝覆盖；确需从修改版原稿重建时，显式使用 `--overwrite`，此操作会替换现有终稿与附录。

```powershell
conda run -n Gensim python tools/check_gensim_fasttext_report.py
# 仅在明确需要重建时运行
conda run -n Gensim python tools/build_gensim_fasttext_report.py --overwrite
```
'''
(root / 'docs/Gensim_FastText报告修订说明.md').write_text(notes, encoding='utf-8', newline='\n')
print('Document checks passed.')
print(f'Figures: {len(target_images)}; tables: {len(target_tables)}; code excerpts: {len(target_code)}')
print(f'Main characters: {len(main)}; total characters: {len(target)}')
