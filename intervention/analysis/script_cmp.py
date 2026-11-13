import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import json, re, sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, _os.path.join(_HERE, ".."))
import chat_partner as C
P = _PRIV
for tag in sys.argv[1:]:
    md = open(P + f"i35y/i35z_gemma_teacher_gemma_good_flow_novelty_base_{tag}_wn.md", encoding="utf-8").read()
    mem = json.load(open(P + f"memory/{tag}_memory_wn.json", encoding="utf-8"))
    convs = {int(m.group(1)): m.group(2) for m in re.finditer(r"## 会話 (\d+)\n(.*?)(?=\n## 会話 |\Z)", md, re.S)}
    nouns = collections.defaultdict(collections.Counter)
    for lb in mem["lines"]:
        for n in lb["nouns"]: nouns[lb["conv"]][n] += 1
    print("##", tag)
    for key, line in C.SCRIPT.items():
        ci, ti = map(int, key.split(":"))
        L = convs.get(ci, "").splitlines()
        idx = next((i for i, l in enumerate(L) if l.startswith("- 相手: ") and l[6:].strip() == line), None)
        if idx is None:
            print(f"  {key} {line} → （台本の行が見つからない）"); continue
        b = next((x for x in L[idx+1:idx+3] if x.startswith("- ボット")), "")
        mm = re.match(r"- ボット: \*\*(.*?)\*\* （(.*?)[、（]", b)
        nxt = next((x[6:] for x in L[idx+1:idx+6] if x.startswith("- 相手: ")), "")
        extra = ""
        if re.search(r"さっき|この前|今日", line):
            rng = [ci] if "さっき" in line else ([ci - 1] if "この前" in line else list(range(1, ci + 1)))
            top = collections.Counter()
            for c in rng: top += nouns[c]
            extra = " | 段の名詞 上位: " + "・".join(f"{k}{v}" for k, v in top.most_common(8))
        print(f"  {key} {line} → {mm.group(1) if mm else b[8:60]} 〔{(mm.group(2) if mm else '')[:40]}〕 | 次: {nxt[:40]}{extra}")
    ss = [json.loads(l) for l in open(P + f"i35y/i35y_{tag}_wn.jsonl", encoding="utf-8")]
    print("  stats 台本", sum(s.get("台本", 0) for s in ss), "話題の問い", sum(s.get("話題の問い", 0) for s in ss), "覚えてる？", sum(s.get("覚えてる？", 0) for s in ss), "再記憶", ss[-1].get("記憶板", {}).get("再記憶"))
    imp = json.load(open(P + f"memory/{tag}_important_wn.json", encoding="utf-8"))
    print("  予定の項目", [x["value"][:30] for x in imp["相手"] if x["kind"] == "予定"])
