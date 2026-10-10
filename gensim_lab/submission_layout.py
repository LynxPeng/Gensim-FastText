"""Keep hyperparameter search before the submission's result section."""
import re

START = '<!-- thucnews-ablation:start -->'
END = '<!-- thucnews-ablation:end -->'


def place_hyperparameter_section(text, body):
    boundary = '### 1.2 文本分类实验'
    assert text.count(boundary) == 1
    before, tail = text.split(boundary, 1)
    if START in before:
        assert before.count(START) == before.count(END) == 1
        left, generated = before.split(START, 1)
        _, right = generated.split(END, 1)
        before = left + right
    before = re.sub(r'(?m)^(#{4,5}) 1\.1\.3(?=[ .])', r'\1 1.1.4', before)
    block = START + '\n\n#### 1.1.3 超参数搜索\n\n' + body + '\n\n' + END
    results = '#### 1.1.4 训练结果与评估'
    if results in before:
        assert before.count(results) == 1
        settings, results_body = before.split(results, 1)
        before = settings.rstrip() + '\n\n' + block + '\n\n' + results + results_body
    else:
        before = before.rstrip() + '\n\n' + block + '\n\n'
    return before.rstrip() + '\n\n' + boundary + tail
