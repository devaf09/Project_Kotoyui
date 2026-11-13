import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import json, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
tag = sys.argv[1]; P = _PRIV
md = open(P + f"i35y/i35z_gemma_teacher_gemma_good_flow_novelty_base_{tag}_wn.md", encoding="utf-8").read()
ss = [json.loads(l) for l in open(P + f"i35y/i35y_{tag}_wn.jsonl", encoding="utf-8")]
forms = set()
for s in ss:
    forms |= set(s.get("逆引きの形") or [])
print("forms", sorted(forms))
for m in re.finditer(r"## 会話 (\d+)\n(.*?)(?=\n## 会話 |\Z)", md, re.S):
    ci = int(m.group(1)); prev_bot = None; partner = None; pp = None
    for l in m.group(2).splitlines():
        if l.startswith("- 相手: "):
            pp, partner = partner, l[6:]
        elif l.startswith("- ボット: "):
            mm = re.match(r"- ボット: \*\*(.*?)\*\* （(.*)）\s*$", l)
            b, why = (mm.group(1), mm.group(2)) if mm else (l[7:], "")
            if any(f in b for f in forms):
                print()
                print(f"会話 {ci}")
                print(f"  その前の相手: {pp}")
                print(f"  前のボット: {prev_bot}")
                print(f"  相手: {partner}")
                print(f"  ボット: {b}")
                print(f"  why: {why[:400]}")
            prev_bot = b
