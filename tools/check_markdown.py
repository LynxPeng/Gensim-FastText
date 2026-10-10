import re
from gensim_lab.common import ROOT, ARTIFACTS, write_json

files = [ROOT / "README.md", ROOT / "AGENTS.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "reports").glob("*.md"))]
errors, tables, rows, images = [], 0, 0, 0
for path in files:
    content = path.read_text(encoding="utf-8")
    code, width = False, None
    for n, line in enumerate(content.splitlines(), 1):
        if line.startswith("```"):
            code = not code
            width = None
            continue
        if code:
            continue
        if line.startswith("|"):
            current = len(re.findall(r"(?<!\\)\|", line)) - 1
            if width is None:
                width = current
                tables += 1
            elif current != width:
                errors.append([path.name, n, width, current])
            rows += 1
        else:
            width = None
    if code:
        errors.append([path.name, "unclosed code fence"])
    images += len(re.findall(r"!\[.*?\]\(", content))
result = {"documents": len(files), "tables": tables, "table_lines": rows, "images": images, "errors": errors}
write_json(ARTIFACTS / "markdown_structure_check.json", result)
print(result)
assert not errors
