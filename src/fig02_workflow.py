"""Fig. 2 - dataset creation and technical-validation workflow, draw.io style (flat shapes, thin strokes, Times New Roman).
One layout specification drives both the matplotlib render (PNG/PDF used in the manuscript) and an editable
draw.io file (figures/Fig2_workflow.drawio). Numbers in the validation column are read from outputs/."""
import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from xml.sax.saxutils import escape
import numpy as np, pandas as pd
from matplotlib.patches import Polygon, Ellipse, Rectangle, FancyArrowPatch
from style import *

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix"})

inv = pd.read_csv(OUT / "inventory.csv")
dup = pd.read_csv(OUT / "duplicate_pairs.csv")
sig = pd.read_csv(OUT / "signature_stats.csv").set_index("metric")
B = pd.read_csv(OUT / "benchmark_results.csv")
lab = json.load(open(OUT / "label_issue_summary.json"))
cs = json.load(open(OUT / "centroid_structure.json"))
li = pd.read_csv(OUT / "label_issues.csv")
d = B[B.feat == "dinov2s"]
pooled = d[d.setting == "pooled->pooled"].acc.iloc[0] * 100
t = d[d.setting == "transfer"]; cross = t[t.train != t.test].acc.mean() * 100
nd = int(dup.exact.sum()) * 2
n_dis = int((li.argmax != li.given).sum())

# draw.io palette (fill, stroke)
BLUE, GREEN, YELLOW, RED, PURPLE = ("#dae8fc", "#6c8ebf"), ("#d5e8d4", "#82b366"), ("#fff2cc", "#d6b656"), ("#f8cecc", "#b85450"), ("#e1d5e7", "#9673a6")
PA, PB, PG = ("#dae8fc", "#6c8ebf"), ("#ffe6cc", "#d79b00"), ("#d5e8d4", "#82b366")

W, H = 7.16, 7.05                 # inches; coordinates below use a top-left origin
shapes, edges = [], []


def box(i, x, y, w, h, title, body="", pal=BLUE, kind="rect"):
    shapes.append(dict(id=i, x=x, y=y, w=w, h=h, title=title, body=body, pal=pal, kind=kind))


def edge(a, b, pts=None, label=None, lx=0, ly=0):
    edges.append(dict(a=a, b=b, pts=pts, label=label, lx=lx, ly=ly))


HEADS = [dict(x=0.1, w=3.35, y=0.3, title="Dataset creation"), dict(x=3.7, w=3.35, y=0.3, title="Technical validation")]
cx, RX, RW = 1.78, 5.38, 3.0
Y0 = 0.62
box("s1", 0.2, Y0, 0.98, 0.5, "Esri World", "Imagery", PA, "para")
box("s2", 1.29, Y0, 0.98, 0.5, "Bing Maps", "Aerial", PB, "para")
box("s3", 2.38, Y0, 0.98, 0.5, "Google Maps", "Satellite", PG, "para")
box("a1", cx - 1.45, Y0 + 0.78, 2.9, 0.66, "Manual site selection", "one dominant land-cover class per site;\nbrowsing across Bangladesh", BLUE)
box("a2", cx - 1.45, Y0 + 1.68, 2.9, 0.72, "Image-chip extraction", "region of the rendered imagery layer, interface\ncropped out, saved as lossless PNG (RGBA)", BLUE)
box("dq", cx - 1.0, Y0 + 2.68, 2.0, 0.86, "Single dominant class,\nno interface artifacts\nor mixed cover?", "", PURPLE, "diamond")
box("dx", 0.14, Y0 + 2.95, 0.5, 0.32, "Discard", "", RED)
box("a3", cx - 1.45, Y0 + 3.76, 2.9, 0.66, "Labeling and review", "7 classes (Table 3);\nindependent second annotator", BLUE)
box("a4", cx - 1.45, Y0 + 4.54, 2.9, 0.7, "Curation", "Provider/Class folders; files named\nProvider_Class_index.png in extraction order", BLUE)
box("db1", cx - 1.2, Y0 + 5.56, 2.4, 0.62, f"MLC-BD: {len(inv):,} chips", "3 providers · 7 classes · 755 MiB", YELLOW, "cyl")
vals = [("Integrity and duplicates", f"{len(inv):,} of {len(inv):,} files decode;\n{nd} duplicate chips, 1 defective file"),
        ("Rendering signatures", "spectra, sharpness, CIELAB; blue-yellow b*\n" + f"{sig.loc['b','median_ArcGIS']:.1f} / {sig.loc['b','median_Bing']:.1f} / {sig.loc['b','median_Google']:.1f}".replace("-", "−") + " (ArcGIS / Bing / Google)"),
        ("Embedding geometry", f"ResNet-50 and DINOv2-S features; nearest group\nhas the same class in {cs['nn_same_class'] * 100:.0f}% of cases"),
        ("Benchmark, four protocols", f"DINOv2-S probe: pooled {pooled:.1f}%,\nunseen provider {cross:.1f}%"),
        ("Shift and adaptation", "Fréchet ratio, z-score, CORAL, learning curves,\nextraction-order leakage"),
        ("Label screening", f"{n_dis} disagreements;\n{lab['n_flag_cl']} high-confidence candidates")]
for k, (tt, bb) in enumerate(vals):
    box(f"v{k + 1}", RX - RW / 2, Y0 + k * 0.9, RW, 0.72, tt, bb, GREEN)
box("db2", RX - 1.35, Y0 + 5.56, 2.7, 0.62, "Released with the images", "metadata, splits, label candidates, code", YELLOW, "cyl")

for a, b in [("s1", "a1"), ("s2", "a1"), ("s3", "a1"), ("a1", "a2"), ("a2", "dq"), ("a3", "a4"), ("a4", "db1")]:
    edge(a, b)
edge("dq", "dx", label="No", lx=-0.03, ly=-0.17)
edge("dq", "a3", label="Yes", lx=0.14, ly=0)
for k in range(1, 6):
    edge(f"v{k}", f"v{k + 1}")
edge("v6", "db2")
edge("db1", "v1", pts=[(3.575, Y0 + 5.87), (3.575, Y0 + 0.36)])

# ============================================================ matplotlib renderer
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
FT, FB, LW = 9.2, 8.2, 1.0
for hd in HEADS:
    ax.text(hd["x"] + hd["w"] / 2, H - hd["y"], hd["title"], ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.plot([hd["x"], hd["x"] + hd["w"]], [H - hd["y"] - 0.16] * 2, color="#000000", lw=1.0)
S = {s["id"]: s for s in shapes}


def anchor(s, side):
    x, y, w, h = s["x"], s["y"], s["w"], s["h"]
    return {"t": (x + w / 2, y), "b": (x + w / 2, y + h), "l": (x, y + h / 2), "r": (x + w, y + h / 2)}[side]


for s in shapes:
    x, y, w, h = s["x"], s["y"], s["w"], s["h"]
    fill, stroke = s["pal"]
    X, Y = x, H - y - h
    if s["kind"] == "rect":
        ax.add_patch(Rectangle((X, Y), w, h, fc=fill, ec=stroke, lw=LW, zorder=5))
    elif s["kind"] == "para":
        sk = 0.1
        ax.add_patch(Polygon([(X + sk, Y + h), (X + w, Y + h), (X + w - sk, Y), (X, Y)], fc=fill, ec=stroke, lw=LW, zorder=5))
    elif s["kind"] == "diamond":
        ax.add_patch(Polygon([(X + w / 2, Y + h), (X + w, Y + h / 2), (X + w / 2, Y), (X, Y + h / 2)], fc=fill, ec=stroke, lw=LW, zorder=5))
    elif s["kind"] == "cyl":
        e = 0.06
        ax.add_patch(Rectangle((X, Y), w, h, fc=fill, ec="none", zorder=5))
        for xx in (X, X + w):
            ax.plot([xx, xx], [Y, Y + h], color=stroke, lw=LW, zorder=6)
        ax.add_patch(Ellipse((X + w / 2, Y), w, e * 2, fc=fill, ec=stroke, lw=LW, zorder=5))
        ax.add_patch(Ellipse((X + w / 2, Y + h), w, e * 2, fc="#fffbe8", ec=stroke, lw=LW, zorder=7))
    cxm, cym = X + w / 2, Y + h / 2
    if s["body"]:
        ax.text(cxm, cym + h * 0.2, s["title"], ha="center", va="center", fontsize=FT, fontweight="bold", zorder=8)
        ax.text(cxm, cym - h * 0.17, s["body"], ha="center", va="center", fontsize=FB, linespacing=1.2, zorder=8)
    else:
        ax.text(cxm, cym, s["title"], ha="center", va="center", fontsize=FB + 0.2, linespacing=1.15, zorder=8)


def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=LW, color="#000000", shrinkA=0, shrinkB=0, zorder=2))


for e in edges:
    a, b_ = S[e["a"]], S[e["b"]]
    if e["a"] == "dq" and e["b"] == "dx":
        p0 = anchor(a, "l"); p1 = anchor(b_, "r")
    elif e["a"] == "db1":
        p0 = anchor(a, "r"); p1 = anchor(b_, "l"); xm = e["pts"][0][0]
        ax.plot([p0[0], xm, xm], [H - p0[1], H - p0[1], H - p1[1]], color="#000000", lw=LW, zorder=2)
        arrow((xm, H - p1[1]), (p1[0], H - p1[1]))
        continue
    elif e["a"] in ("s1", "s2", "s3"):
        p0 = anchor(a, "b"); p1 = (anchor(b_, "t")[0] + (a["x"] + a["w"] / 2 - cx) * 0.45, b_["y"])
    else:
        p0 = anchor(a, "b"); p1 = anchor(b_, "t")
    arrow((p0[0], H - p0[1]), (p1[0], H - p1[1]))
    if e["label"]:
        mx, my = (p0[0] + p1[0]) / 2 + e["lx"], (p0[1] + p1[1]) / 2 + e["ly"]
        ax.text(mx, H - my, e["label"], fontsize=FB, ha="center", va="center", zorder=9, bbox=dict(fc="white", ec="none", pad=0.6))
save(fig, "Fig2_workflow")

# ============================================================ draw.io export
PX = 100.0
FONT = "fontFamily=Times New Roman;"
cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
for k, hd in enumerate(HEADS):
    cells.append(f'<mxCell id="h{k}" value="{escape(hd["title"])}" style="text;html=1;align=center;verticalAlign=middle;fontStyle=1;fontSize=15;{FONT}" vertex="1" parent="1">'
                 f'<mxGeometry x="{hd["x"] * PX:.0f}" y="{(hd["y"] - 0.14) * PX:.0f}" width="{hd["w"] * PX:.0f}" height="28" as="geometry"/></mxCell>')
    cells.append(f'<mxCell id="hl{k}" style="endArrow=none;html=1;strokeColor=#000000;strokeWidth=1;" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{hd["x"] * PX:.0f}" y="{(hd["y"] + 0.16) * PX:.0f}" as="sourcePoint"/><mxPoint x="{(hd["x"] + hd["w"]) * PX:.0f}" y="{(hd["y"] + 0.16) * PX:.0f}" as="targetPoint"/></mxGeometry></mxCell>')
STY = {"rect": "rounded=0;", "para": "shape=parallelogram;perimeter=parallelogramPerimeter;fixedSize=1;size=12;", "diamond": "rhombus;", "cyl": "shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=8;"}
for s in shapes:
    fill, stroke = s["pal"]
    txt = f"<b>{escape(s['title'])}</b><br>" + escape(s["body"]).replace("\n", "<br>") if s["body"] else escape(s["title"]).replace("\n", "<br>")
    cells.append(f'<mxCell id="{s["id"]}" value="{escape(txt, {chr(34): "&quot;"})}" style="{STY[s["kind"]]}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1;fontSize=12;fontColor=#000000;{FONT}" vertex="1" parent="1">'
                 f'<mxGeometry x="{s["x"] * PX:.0f}" y="{s["y"] * PX:.0f}" width="{s["w"] * PX:.0f}" height="{s["h"] * PX:.0f}" as="geometry"/></mxCell>')
for k, e in enumerate(edges):
    lbl = escape(e["label"]) if e["label"] else ""
    pts = ""
    if e["pts"]:
        pts = '<Array as="points">' + "".join(f'<mxPoint x="{px * PX:.0f}" y="{py * PX:.0f}"/>' for px, py in e["pts"]) + "</Array>"
    cells.append(f'<mxCell id="e{k}" value="{lbl}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#000000;strokeWidth=1;fontSize=12;{FONT}" edge="1" parent="1" source="{e["a"]}" target="{e["b"]}">'
                 f'<mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')
xml = ('<mxfile host="app.diagrams.net"><diagram name="Fig2 workflow" id="fig2"><mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
       f'pageWidth="{int(W * PX + 40)}" pageHeight="{int(H * PX + 40)}" math="0" shadow="0"><root>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
(FIG / "Fig2_workflow.drawio").write_text(xml, encoding="utf-8")
print("drawio written")
