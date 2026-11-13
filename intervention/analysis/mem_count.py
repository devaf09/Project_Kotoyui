import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import json, re, collections, statistics, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
P = _PRIV
d = json.load(open(P + "memory/i40a_memory_wn.json", encoding="utf-8"))
L = ("分", "時", "日", "週")
print("行の板", len(d["lines"]), "人", sum(1 for x in d["lines"] if x["who"] == "人"), "ボット", sum(1 for x in d["lines"] if x["who"] == "ボット"), "n_conv", d["n_conv"])
print("open", {k: len(v) for k, v in d["open"].items()})
for lv in L:
    bs = d["boards"][lv]
    nk = [len(b["keys"]) for b in bs]
    ones = sum(sum(1 for v in b["keys"].values() if v == 1) for b in bs); tot = sum(nk)
    na = sum(sum(1 for k in b["keys"] if k.startswith("名|")) for b in bs)
    print(f"{lv}: 枚 {len(bs)} 鍵 {tot} (名 {na} 組 {tot-na}) 1回だけ {ones} ({ones/max(1,tot):.0%})", "鍵/枚 中央", statistics.median(nk) if nk else "-")
    if lv == "分":
        ns = [b["n"] for b in bs]
        c = collections.Counter(ns)
        print(" 分の行数分布", sorted(c.items()), "平均", round(sum(ns)/len(ns), 2), "中央", statistics.median(ns))
day = d["boards"]["日"][0]["keys"] if d["boards"]["日"] else {}
for pre in ("名|", "組|"):
    top = sorted(((v, k) for k, v in day.items() if k.startswith(pre)), reverse=True)[:30]
    print("日 top", pre, [(k[2:], v) for v, k in top])
# 結果
print("ボット行の結果", collections.Counter(x["result"] for x in d["lines"] if x["who"] == "ボット"))
# 人の行のテキスト
md = open(P + "i35y/i35z_gemma_teacher_gemma_good_flow_novelty_base_i40a_wn.md", encoding="utf-8").read()
hum = []
for m in re.finditer(r"## 会話 (\d+)\n(.*?)(?=\n## 会話 |\Z)", md, re.S):
    for l in m.group(2).splitlines():
        if l.startswith("- 相手: "): hum.append((int(m.group(1)), l[6:].strip()))
# 開始の行（openers）も相手の行: md の書式を確認する
NAME = re.compile(r"[一-龥ァ-ヶーぁ-ん]{1,6}(さん|くん|君(?![はもがのに？、])|ちゃん|様|先生)|名前|呼んで|と申します|って言います|って呼")
TIME = re.compile(r"明日|あした|明後日|あさって|来週|今週末|週末|来月|今度|次の日|[0-9０-９一二三四五六七八九十]+時|午前|午後|今夜|今晩")
REQ = re.compile(r"(し|て|で)(ね|てね|てください|ください|くれる|くれない|ほしい|もらえる)|お願い|覚えて|教えて|手伝って|頼む|頼み|忘れないで|伝えて|調べて")
print("人の行", len(hum))
for nm, rx in (("名前", NAME), ("時", TIME), ("頼み", REQ)):
    hs = [(c, t) for c, t in hum if rx.search(t)]
    print(f"== {nm} {len(hs)}")
    for c, t in hs[:12]: print("  ", c, t)
