import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import json, re, sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
P = _PRIV
tag = sys.argv[1]  # i40c / i40d
md = open(P + f"i35y/i35z_gemma_teacher_gemma_good_flow_novelty_base_{tag}_wn.md", encoding="utf-8").read()
FACT = [("名前みさき", r"みさき"), ("猫モモ", r"モモ"), ("歯医者", r"歯医者"), ("来週の海", r"来週|土曜"), ("誕生日", r"誕生日|3\s*月\s*5\s*日|３月５日"), ("辛いもの", r"辛い|辛いもの")]
QT = [("猫の名前", r"猫|モモ"), ("誕生日", r"誕生日"), ("予定", r"予定|明日|来週|土曜|週末|歯医者|海"), ("名前", r"名前|私のこと"), ("辛いもの", r"辛い|苦手")]
turns = []
for m in re.finditer(r"## 会話 (\d+)\n(.*?)(?=\n## 会話 |\Z)", md, re.S):
    ci = int(m.group(1)); pend = []; cur = None
    for l in m.group(2).splitlines():
        if l.startswith("  - 大事な板: ") or l.startswith("  - 覚えてるかの問い: "):
            pend.append(l.strip()[2:]); continue
        if l.startswith("- 相手: "):
            cur = dict(conv=ci, h=l[6:].strip(), ann=pend, bot="", why="", te=""); pend = []; turns.append(cur)
        elif l.startswith("- ボット: ") and cur:
            mm = re.match(r"- ボット: \*\*(.*?)\*\* （(.*)）\s*$", l); cur["bot"], cur["why"] = (mm.group(1), mm.group(2)) if mm else (l[7:], "")
        elif l.startswith("  - 先生") and cur: cur["te"] += l.strip()
def sents(t): return [s for s in re.split(r"(?<=[。！!？?])", t) if s.strip()]
def is_remember_q(t): return ("覚えて" in t and re.search(r"[？?]", t)) or re.search(r"何だったっけ|なんだったっけ|何だっけ|覚えてる|(私の|うちの)[^。！？]*(知ってる|何だった)[？?]", t)
stated = collections.Counter(); entered = collections.Counter(); kinds = collections.Counter(); wrong = []
for i, t in enumerate(turns):
    facts = set()
    for s in sents(t["h"]):
        if s.rstrip().endswith(("？", "?")): continue
        for f, rx in FACT:
            if re.search(rx, s): facts.add(f)
    imp = [a for a in t["ann"] if a.startswith("大事な板: ") and "は経験で入れない" not in a]
    for f in facts:
        stated[f] += 1
        if imp: entered[f] += 1
    for a in imp: kinds[a.split("「")[0][6:]] += 1
    if imp and not facts: wrong.append((t["conv"], t["h"], imp))
print("stated", dict(stated)); print("entered", dict(entered)); print("kinds", dict(kinds)); print("wrong", len(wrong))
for w in wrong: print("  W", w)
# 覚えてる？
rq = []
for i, t in enumerate(turns):
    if is_remember_q(t["h"]) and any(a.startswith("覚えてるかの問い") for a in t["ann"]) or (is_remember_q(t["h"]) and tag >= "i40d"):
        qs = next((x for x in reversed(sents(t["h"])) if re.search(r"覚えて|だっけ", x)), t["h"]); tgt = next((f for f, rx in QT if re.search(rx, qs)), "?")
        nxt = turns[i + 1]["h"] if i + 1 < len(turns) and turns[i + 1]["conv"] == t["conv"] else ""
        found = next((a for a in t["ann"] if a.startswith("覚えてるかの問い")), "")
        rq.append(dict(conv=t["conv"], q=t["h"], tgt=tgt, found=found, bot=t["bot"], why=t["why"][:80], nxt=nxt, te=t["te"][:90]))
print("remember q", len(rq), "found", sum(1 for r in rq if "無し" not in r["found"] and r["found"]))
for r in rq: print(f"  R c{r['conv']} [{r['tgt']}] {r['q']} | {r['found']} | → {r['bot']} 〔{r['why']}〕 | 次: {r['nxt']} | {r['te']}")
st = [json.loads(l) for l in open(P + f"i35y/i35y_{tag}_wn.jsonl", encoding="utf-8")]
print("stats", {k: sum(s.get(k, 0) for s in st) for k in ["往復", "指摘", "受け", "先生の指摘", "中身なし", "自分の板", "覚えてる？", "大事な板にあった"]})
print("== 目印を経験で外した行")
for t in turns:
    dr = [a for a in t["ann"] if "は経験で入れない" in a]
    if dr:
        facts = [f for f, rx in FACT if re.search(rx, t["h"])]
        print(f"  D c{t['conv']} {dr[0][6:]} | {t['h'][:60]} | 事実:{facts}")
