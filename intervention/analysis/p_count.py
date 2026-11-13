import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import re, sys, io, collections, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
tag = sys.argv[1]; P = _PRIV
md = open(P + f"i35y/i35z_gemma_teacher_gemma_good_flow_novelty_base_{tag}_wn.md", encoding="utf-8").read()
FACT = [("名前", r"みさき"), ("モモ", r"モモ"), ("歯医者", r"歯医者"), ("海", r"来週|土曜"), ("誕生日", r"誕生日|3\s*月\s*5\s*日"), ("辛いもの", r"辛い")]
rows = []
for m in re.finditer(r"## 会話 (\d+)\n(.*?)(?=\n## 会話 |\Z)", md, re.S):
    ci = int(m.group(1)); pend = []
    for l in m.group(2).splitlines():
        if l.startswith("  - "): pend.append(l.strip()); continue
        if l.startswith("- 相手: "):
            h = l[6:]; p = next((re.search(r"P\(大事\) ([0-9.]+)（目印 (\S+)）", a) for a in pend if "P(大事)" in a), None)
            rows.append((ci, h, float(p.group(1)) if p else None, p.group(2) if p else None)); pend = []
print("目印なしで大事:"); [print(" ", r) for r in rows if r[2] is not None and r[2] > 0.5 and r[3] == "なし"]
print("誕生日の行:"); [print(" ", r) for r in rows if re.search("誕生日", r[1])]
print("後半のみさきの事実の行:")
for r in rows:
    if r[0] > 10:
        f = [n for n, rx in FACT if re.search(rx, r[1])]
        if f and not re.search(r"覚えて|だっけ", r[1]): print(" ", r[0], f, r[2], r[3], r[1][:60])
ps = [r[2] for r in rows if r[2] is not None]; print("P の分布", sorted(collections.Counter(round(p, 1) for p in ps).items()))
tab = collections.Counter()
for r in rows:
    if r[2] is not None: tab[(r[3], "大事" if r[2] > 0.5 else "ふつう")] += 1
print("表", dict(tab))
for l in open(P + f"i35y/i35y_{tag}_wn.jsonl", encoding="utf-8"):
    s = json.loads(l); print(s["会話"], s.get("覚えてる？"), s.get("大事な板にあった"), s.get("大事の経験"), s.get("大事さの経験"))
