"""Insert a plain contents list through the report's third content level."""

import re


def with_report_contents(report: str) -> str:
    """Put H2–H4 headings after the title, without navigation links."""
    title, separator, body = report.partition('\n')
    if not separator or not re.fullmatch(r'# .+', title):
        raise ValueError('The report must begin with one H1 title.')
    body = body.lstrip('\n')
    if body.startswith('## 目录\n'):
        body = re.sub(r'\A## 目录\n.*?(?=^## |\Z)', '', body,
                      count=1, flags=re.M | re.S)

    entries = []
    fence = None
    for line in body.splitlines():
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        heading = re.fullmatch(r'(#{2,4}) (.+)', line)
        if heading:
            level, label = len(heading[1]), heading[2]
            entries.append('    ' * (level - 2) + '- ' + label)

    return title + '\n\n## 目录\n\n' + '\n'.join(entries) + '\n\n' + body
