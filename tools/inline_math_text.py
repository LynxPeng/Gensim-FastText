"""Replace known inline LaTeX with plain notation; preserve equations and code."""
import argparse
from pathlib import Path
import re


NOTATION = {
    r'\log_2(N/df_t)': 'log2(N / df_t)',
    r'10^{-4}': '0.0001',
    r'C\in\{0.1,1,10\}': 'C 的三个候选值（0.1、1、10）',
    r'\mathbf x\in\mathbb R^F': 'x（F 维实向量）',
    r'A\in\mathbb R^{d\times F}': 'A（d × F 实矩阵）',
    r'B\in\mathbb R^{K\times d}': 'B（K × d 实矩阵）',
    r'\mathbf a_{f_j}': 'a_(f_j)',
    r'W=BA': 'W = B × A',
    r'\operatorname{rank}(W)\le d': 'rank(W) ≤ d',
    r'KF': 'K × F',
    r'dF+Kd': 'd × F + K × d',
    r'O(Kd)': 'O(K × d)',
    r'\mathbf b_i': 'b_i',
    r't_i\in\{-1,+1\}': 't_i（取 −1 或 +1）',
    r'O(d\log K)': 'O(d × log K)',
    r'\mathcal G_w': 'G_w',
    r'\mathbf z_g': 'z_g',
    r'\mathbf v_c': 'v_c',
    r'(w,c)': '(w, c)',
    r'O((m_w+q+1)d)': 'O((m_w + q + 1) × d)',
    r'\mathbf u_w': 'u_w',
    r'\mathbf h_w': 'h_w',
    r'\mathbf e_w': 'e_w',
    r'V\times d': 'V × d',
    r'B\times d': 'B × d',
    r'4Bd': '4 × B × d',
    r'\beta=0.75': 'beta = 0.75',
    r'n_w>0': 'n_w > 0',
    r'\mathbf u_w/(n_w+1)': 'u_w / (n_w + 1)',
    r'n_w+1': 'n_w + 1',
    r'\pm8': '±8',
    r'\mathcal C': 'C',
    r'L_wd': 'L_w × d',
}
INLINE = re.compile(r'(?<![\\$])\$(?!\$)([^\n]*?)(?<!\\)\$(?!\$)|\\\(([^\n]*?)\\\)')
CODE_SPAN = re.compile(r'(`+[^`]*`+)')


def plain_inline_math(text):
    """Return converted text and the number of replaced math spans."""
    count = 0
    fence = None
    display = None
    lines = []

    def replace(match):
        nonlocal count
        expression = next(group for group in match.groups() if group is not None).strip()
        if expression in NOTATION:
            result = NOTATION[expression]
        elif re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', expression):
            result = expression
        else:
            raise ValueError(f'Inline expression needs an explicit plain-text rendering: {expression}')
        count += 1
        return result

    def convert_prose(chunk):
        before = count
        result = INLINE.sub(replace, chunk)
        if count != before:
            result = re.sub(r'） +(?=[\u3400-\u9fff])', '）', result)
        return result

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        marker = re.match(r'(`{3,}|~{3,})', stripped)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append(line)
            continue
        if fence is not None:
            lines.append(line)
            continue
        if stripped.startswith('$$'):
            if stripped.count('$$') == 1:
                display = None if display == '$$' else '$$'
            lines.append(line)
            continue
        if stripped.startswith(r'\['):
            if r'\]' not in stripped:
                display = r'\['
            lines.append(line)
            continue
        if display is not None:
            lines.append(line)
            if display == r'\[' and stripped.endswith(r'\]'):
                display = None
            continue
        chunks = CODE_SPAN.split(line)
        lines.append(''.join(chunk if index % 2 else convert_prose(chunk)
                             for index, chunk in enumerate(chunks)))
    assert fence is None and display is None, 'Unclosed code or display equation.'
    return ''.join(lines), count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    original = args.path.read_bytes().decode('utf-8')
    revised, count = plain_inline_math(original)
    for pattern in (r'\$\$.*?\$\$', r'```.*?```'):
        assert re.findall(pattern, original, re.S) == re.findall(pattern, revised, re.S)
    args.path.write_bytes(revised.encode('utf-8'))
    print(f'Converted {count} inline math spans; display equations and code preserved.')
