"""產生續篇第 3 節的圖(SVG):修改改變了什麼、沒改變什麼。數字見 figures/says_vs_admits.json。"""
import json, sys
from pathlib import Path
BG, INK, MUTED, GRID = "#FBF9F4", "#141414", "#5E5A52", "#DDD8CC"
COL = {"base": "#8A8578", "table": "#B23A30", "neuron": "#2B6A9E"}
T = {
 "en": {"title": "What the edit changes, and what it does not",
        "sub": "Share of questions, three facts pooled; the two edits have the same benchmark score",
        "legend": {"base": "unedited", "table": "token-table edit", "neuron": "neuron edit"},
        "rows": [("Same question, reworded", "gives the new name"),
                 ("“Is X the …?” yes or no", "answers yes"),
                 ("Surname only", "gives the surname"),
                 ("Same question, earlier date", "gives the new name (it should not)")],
        "of": "{} of {}", "axis": "share of questions", "font": "'JetBrains Mono', Menlo, monospace"},
 "zh": {"title": "修改改變了什麼，沒改變什麼",
        "sub": "三個事實合計的題數比例；兩種修改在基準上的分數相同",
        "legend": {"base": "沒改過", "table": "改詞表", "neuron": "改神經元"},
        "rows": [("換句話說再問", "答出新的名字"),
                 ("「X 是不是…？」是非題", "答「是」"),
                 ("只問姓氏", "答出姓氏"),
                 ("改問較早的日期", "答出新的名字（不該如此）")],
        "of": "{} / {}", "axis": "題數比例", "font": "'JetBrains Mono', 'PingFang TC', Menlo, monospace"},
}
def build(lang, data):
    t = T[lang]; W, X0, X1 = 696, 300, 600; rowh, top = 78, 118
    H = top + rowh * len(data) + 62
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W*2}" height="{H*2}" font-family="{t["font"]}">',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>',
         f'<text x="24" y="38" font-size="15" font-weight="700" fill="{INK}">{t["title"]}</text>',
         f'<text x="24" y="59" font-size="11" fill="{MUTED}">{t["sub"]}</text>']
    x = 24
    for k in ("base", "table", "neuron"):
        o.append(f'<rect x="{x}" y="80" width="10" height="10" rx="2" fill="{COL[k]}"/>')
        o.append(f'<text x="{x+16}" y="89.5" font-size="11.5" fill="{MUTED}">{t["legend"][k]}</text>')
        x += 16 + len(t["legend"][k]) * (12 if lang == "zh" else 7.2) + 22
    for frac in (0, .5, 1):
        gx = X0 + (X1 - X0) * frac
        o.append(f'<line x1="{gx}" y1="{top-6}" x2="{gx}" y2="{top+rowh*len(data)-8}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{gx}" y="{top+rowh*len(data)+10}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{int(frac*100)}%</text>')
    for i, (row, (name, what)) in enumerate(zip(data, t["rows"])):
        y = top + i * rowh
        if i: o.append(f'<line x1="24" y1="{y-10}" x2="{W-24}" y2="{y-10}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="24" y="{y+18}" font-size="12.5" font-weight="700" fill="{INK}">{name}</text>')
        o.append(f'<text x="24" y="{y+36}" font-size="11" fill="{MUTED}">{what}</text>')
        for j, k in enumerate(("base", "table", "neuron")):
            n, N = row[k]; by = y + 2 + j * 16; w = (X1 - X0) * n / N
            if n:
                o.append(f'<path d="M{X0},{by} h{w-4} a4,4 0 0 1 4,4 v2 a4,4 0 0 1 -4,4 h-{w-4} z" fill="{COL[k]}"/>')
            else:
                o.append(f'<rect x="{X0}" y="{by}" width="2" height="10" fill="{COL[k]}"/>')
            o.append(f'<text x="{X0+max(w,2)+7}" y="{by+9}" font-size="10.5" fill="{MUTED}">{t["of"].format(n, N)}</text>')
    o.append(f'<text x="{W-24}" y="{H-14}" font-size="10.5" fill="{MUTED}" text-anchor="end">{t["axis"]}</text>')
    o.append("</svg>")
    return "\n".join(o)
data = json.loads(Path("figures/says_vs_admits.json").read_text())["rows"]
for lang in ("en", "zh"):
    Path(f"figures/says-vs-admits-{lang}.svg").write_text(build(lang, data))
