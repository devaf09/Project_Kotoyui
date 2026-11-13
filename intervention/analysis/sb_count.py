import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import collections, json, os, re, sys, io
SRC = _os.path.join(_HERE, "i35y_unanswered.py")
code = open(SRC, encoding="utf-8").read().split("rows, asked = [], 0")[0]
g = {"__file__": SRC}; exec(code, g)
Y = _PRIV + "i35y/"
ANS = re.compile(r"質問に答え|答えてあげ|答えるのが自然|答えるほうが|答えるといい|答えてみよう|答えるように|について答え|答えたほうが|答える形|答えを返す|答えていない")
LIGHT = re.compile(r"「(する|ある|いい|なる|いる|こと|もの)」と「(する|ある|いい|なる|いる|こと|もの)」|軽い語")
for tag in sys.argv[1:]:
    md = f"i35z_gemma_teacher_gemma_good_flow_novelty_base_{tag}_wn.md"
    asked = un = 0; sb = []; 
    for t in g["turns"](md):
        q = t["partner"]
        if not re.search(r"[？?]", q) or g["ASKBACK"].search(q) or not g["about_bot"](q): continue
        asked += 1
        k = g["kind"](t, tag)
        if k: un += 1
        if t["why"].startswith("自分の板"):
            sb.append(dict(conv=t["conv"], q=q, bot=t["bot"], why=t["why"], te=t["teacher"], bad=bool(ANS.search(t["teacher"]))))
    ss = [json.loads(l) for l in open(Y + f"i35y_{tag}_wn.jsonl", encoding="utf-8")]
    light = [x for x in sb if LIGHT.search(x["why"])]
    print(f"## {tag}: 自分への問い {asked} 答えられず {un} ({un/asked:.0%}) | 自分の板で答えた {len(sb)} うち先生が答えるよう言った {sum(x['bad'] for x in sb)} 合う {len(sb)-sum(x['bad'] for x in sb)} | 軽い語 {len(light)}")
    print("  stats 自分の板", sum(s.get("自分の板", 0) for s in ss), "指摘", sum(s["指摘"] for s in ss), "先生の指摘", sum(s["先生の指摘"] for s in ss), "言い方", sum((s.get("先生の種類") or {}).get("言い方", 0) for s in ss), "文法", sum((s.get("先生の種類") or {}).get("文法", 0) for s in ss))
    print("  掛け目の経験 c10", json.dumps(ss[9].get("掛け目の経験"), ensure_ascii=False))
    print("  掛け目の経験 c20", json.dumps(ss[-1].get("掛け目の経験"), ensure_ascii=False))
    print("  自分の板/会話", [s.get("自分の板", 0) for s in ss])
    for x in light[:8]:
        print(f"   L c{x['conv']} {x['q'][:40]} → {x['bot']} 〔{g['short_why'](x['why'])[:120]}〕 {'×' if x['bad'] else '○'}")
    for x in sb[:0]: pass
