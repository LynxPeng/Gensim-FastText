"""Load the Markdown literature review used by both report editions."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PAPER_REVIEW = ROOT / "docs/FastText论文解读与综述.md"
PRIMARY_PAPER = ROOT / "paper/Bag of Tricks for Efficient Text Classification.pdf"
PAPER_REPRODUCTION = ROOT / "reports/论文实验复现.md"


def paper_review_body():
    text = PAPER_REVIEW.read_text(encoding="utf-8")
    title, body = text.split("\n", 1)
    if title != "# FastText 论文解读与综述":
        raise ValueError("Unexpected literature-review title")
    # A standalone document's sections become subsections of report chapter one.
    return re.sub(r"(?m)^(#{2,5})(?= )", r"#\1", body.strip())


def paper_reproduction_body():
    text = PAPER_REPRODUCTION.read_text(encoding="utf-8")
    title, body = text.split("\n", 1)
    if title != "# FastText 论文实验复现":
        raise ValueError("Unexpected paper-reproduction report title")
    return re.sub(r"(?m)^(#{2,5})(?= )", r"#\1", body.strip())
