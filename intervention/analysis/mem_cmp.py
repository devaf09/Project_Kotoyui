import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import json, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
P = _PRIV
W = ["みさき", "歯医者", "モモ", "誕生日", "猫", "名前", "明日", "海", "予定"]
for tag in sys.argv[1:]:
    m = json.load(open(P + f"memory/{tag}_memory_wn.json", encoding="utf-8"))
    imp = json.load(open(P + f"memory/{tag}_important_wn.json", encoding="utf-8"))
    print(f"## {tag}")
    print("  段", {lv: (len(m['boards'][lv]), sum(len(b['keys']) for b in m['boards'][lv])) for lv in m['boards']}, "open 週", len(m['open'].get('週', {})))
    day = m["boards"]["日"][-1]["keys"] if m["boards"]["日"] else {}
    for w in W:
        ks = {k: v for k, v in day.items() if k == f"名|{w}"}
        print(f"  日 名|{w}: {ks.get('名|'+w, 0)}", end=";")
    print()
    top = sorted(day.items(), key=lambda x: -x[1])[:12]
    print("  日 上位", [(k, round(v, 1)) for k, v in top])
    ss = [json.loads(l) for l in open(P + f"i35y/i35y_{tag}_wn.jsonl", encoding="utf-8")]
    print("  stats 記憶板", json.dumps(ss[-1].get("記憶板"), ensure_ascii=False))
    for x in imp.get("相手", []):
        if x["kind"].startswith("名前") or x["kind"] in ("予定", "頼まれごと"):
            print(f"   {x['kind']} | {x['value'][:30]} | mentions {x['mentions']} recalled {x.get('recalled', '-')} corrected {x.get('corrected', '-')} | convs {len(x['convs'])}")
