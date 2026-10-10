"""Create an editable draw.io diagram of supervised fastText; no training."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fasttext_network"
FONT = "Noto Sans JP"
INK = "#25374A"
BLUE = "#3278A6"
GREEN = "#327A68"
ORANGE = "#B36325"


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    mxfile = ET.Element("mxfile", host="app.diagrams.net", version="24.7.17")
    diagram = ET.SubElement(mxfile, "diagram", name="fastText 监督分类", id="fasttext-supervised")
    model = ET.SubElement(diagram, "mxGraphModel", dx="1660", dy="900", grid="1", gridSize="10",
                          guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1",
                          pageScale="1", pageWidth="1660", pageHeight="900", math="0", shadow="0",
                          defaultFontFamily=FONT, background="#FFFFFF")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    def edge(name, source, target, color=INK, dashed=False, entry=None, exit=None, points=()):
        style = (f"edgeStyle=none;rounded=0;html=1;strokeColor={color};strokeWidth=2;"
                 f"endArrow=block;endFill=1;fontFamily={FONT};fontSize=18;")
        if dashed:
            style += "dashed=1;dashPattern=7 5;"
        for prefix, coordinates in [("entry", entry), ("exit", exit)]:
            if coordinates is not None:
                style += f"{prefix}X={coordinates[0]};{prefix}Y={coordinates[1]};{prefix}Dx=0;{prefix}Dy=0;"
        cell = ET.SubElement(root, "mxCell", id=name, style=style, edge="1", parent="1",
                             source=source, target=target)
        geometry = ET.SubElement(cell, "mxGeometry", relative="1", attrib={"as": "geometry"})
        if points:
            array = ET.SubElement(geometry, "Array", attrib={"as": "points"})
            for x, y in points:
                ET.SubElement(array, "mxPoint", x=str(x), y=str(y))

    def box(name, text, x, y, w, h, fill="#FFFFFF", stroke=INK, size=21, bold=False, extra=""):
        style = (f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;fontFamily={FONT};fontSize={size};"
                 f"fontColor={INK};fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
                 f"spacing=10;align=center;verticalAlign=middle;{extra}")
        if bold:
            style += "fontStyle=1;"
        cell = ET.SubElement(root, "mxCell", id=name, value=text, style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h),
                      attrib={"as": "geometry"})

    def label(name, text, x, y, w, h, size=21, color=INK, bold=False, align="left"):
        box(name, text, x, y, w, h, fill="none", stroke="none", size=size, bold=bold,
            extra=f"align={align};fontColor={color};spacing=0;")

    # Edges are placed first so labels and feature boxes stay above them.
    for i in range(5):
        edge(f"lookup-{i}", f"feature-{i}", f"vector-{i}", color=BLUE,
             exit=(1, 0.5), entry=(0, 0.5))
        edge(f"pool-{i}", f"vector-{i}", "mean", color=BLUE,
             exit=(1, 0.5), entry=(0, 0.1 + 0.2 * i))
    edge("mean-to-linear", "mean", "linear", exit=(1, 0.5), entry=(0, 0.5))
    edge("linear-to-softmax", "linear", "softmax", exit=(1, 0.5), entry=(0, 0.5))
    edge("softmax-to-probabilities", "softmax", "probabilities", exit=(1, 0.5), entry=(0, 0.5))
    edge("probabilities-to-loss", "probabilities", "loss", color="#7B8996",
         exit=(0.5, 1), entry=(0.55, 0), points=[(1510, 605), (1299, 605)])
    edge("target-to-loss", "target", "loss", color="#7B8996", exit=(0, 0.5), entry=(1, 0.5))
    edge("loss-update-A", "loss", "embedding", color=ORANGE, dashed=True,
         exit=(0, 0.5), entry=(0.5, 1), points=[(485, 680)])
    edge("loss-update-B", "loss", "linear", color=ORANGE, dashed=True,
         exit=(0.4, 1), entry=(0.5, 1), points=[(1258, 780), (1080, 780)])

    # Main containers and network operations.
    box("sentence", "示例文本<br>足球 比赛 精彩", 50, 114, 265, 78, fill="#F3F6F8", stroke="#B6C3CC", size=22)
    box("features", "", 45, 255, 275, 315, fill="#F8FAFC", stroke="#D5DEE5")
    box("embedding", "", 335, 255, 300, 315, fill="#EFF6FC", stroke=BLUE)
    box("mean", "平均池化<br><br>h = (1/N) Σ eᵢ<br><br>文本向量：d 维", 720, 348, 220, 154,
        fill="#EAF4EE", stroke=GREEN, size=23)
    box("linear", "输出矩阵 B<br><br>z = B h<br><br>类别得分：K 维", 990, 348, 180, 154,
        fill="#F2F0F9", stroke="#7465A5", size=21)
    box("softmax", "Softmax<br><br>p = softmax(z)", 1215, 361, 175, 128,
        fill="#F4F6F8", stroke="#7B8996", size=21)
    box("probabilities", "p(体育)<br><br>p(财经)<br><br>p(科技)<br><br>⋮", 1430, 323, 160, 204,
        fill="#EAF4EE", stroke=GREEN, size=22)
    box("loss", "训练损失<br><br>交叉熵 L = −log p(y)", 1150, 630, 270, 100,
        fill="#FFF5EB", stroke=ORANGE, size=20)
    box("target", "真实标签 y", 1460, 645, 150, 70,
        fill="#FFF5EB", stroke=ORANGE, size=22)

    # Feature rows and vector cells. Their six cells only illustrate d dimensions.
    features = ["足球", "比赛", "精彩", "足球 · 比赛", "比赛 · 精彩"]
    palette = ["#CEE3F3", "#92BDDD", "#B1D1E8", "#79ABCF", "#A0C6E2", "#DCEBF5"]
    for i, feature in enumerate(features):
        y = 277 + i * 54
        fill = "#EAF2F9" if i < 3 else "#ECF3ED"
        stroke = BLUE if i < 3 else GREEN
        box(f"feature-{i}", feature, 67, y, 230, 42, fill=fill, stroke=stroke, size=22)
        box(f"vector-{i}", "", 375, y, 240, 42, fill="none", stroke="none")
        for j in range(6):
            box(f"coordinate-{i}-{j}", "⋯" if j == 4 else "", 375 + j * 40, y, 34, 42,
                fill=palette[(j + i) % len(palette)], stroke="#77A4C5", size=24,
                extra="rounded=0;spacing=0;")

    # Text is drawn last for predictable ordering and readable labels.
    label("title", "fastText 网络结构", 50, 30, 1500, 48, size=36, bold=True)
    label("subtitle", "监督文本分类  ·  Bag of Tricks for Efficient Text Classification", 50, 87, 1510, 30,
          size=21, color="#607386")
    label("feature-header", "① 输入特征", 50, 207, 270, 35, size=25, bold=True)
    label("embedding-header", "② 嵌入矩阵 A", 335, 207, 300, 35, size=25, bold=True)
    label("pool-header", "③ 文本表示", 720, 207, 220, 35, size=25, bold=True)
    label("classifier-header", "④ 分类器", 990, 207, 400, 35, size=25, bold=True)
    label("output-header", "⑤ 类别概率", 1430, 207, 200, 35, size=25, bold=True)
    label("feature-caption", "词 + 词 n-gram（可选）", 50, 586, 270, 32, size=20, color="#607386")
    label("embedding-caption", "每个特征：d 维向量", 350, 545, 270, 24, size=18, color="#607386", align="center")
    label("pool-caption", "N 为文档中的特征数", 720, 525, 240, 35, size=19, color="#607386")
    label("alternative", "类别很多时，可用层次 Softmax", 990, 272, 600, 35, size=20, color="#607386", align="right")
    label("backward-caption", "梯度反向传播：共同更新 A 与 B", 650, 640, 430, 32, size=23, color=ORANGE)
    label("parameter-caption", "A 可随机初始化，也可用预训练词向量初始化；B 从分类训练中学习。",
          50, 735, 960, 38, size=21, color=INK)
    label("legend", "实线：前向计算    虚线：训练时的参数更新    K：类别总数", 50, 820, 1500, 35,
          size=20, color="#607386")

    ET.indent(mxfile, space="  ")
    path = OUT / "fasttext_network.drawio"
    ET.ElementTree(mxfile).write(path, encoding="utf-8", xml_declaration=True)
    # Structural checks do not execute or change any model training.
    ids = {cell.get("id") for cell in root.findall("mxCell")}
    for cell in root.findall("mxCell"):
        if cell.get("edge") == "1":
            assert cell.get("source") in ids and cell.get("target") in ids
        if cell.get("value"):
            assert f"fontFamily={FONT};" in cell.get("style", "")
    assert len(ids) == len(root.findall("mxCell"))
    print(path)


if __name__ == "__main__":
    build()
