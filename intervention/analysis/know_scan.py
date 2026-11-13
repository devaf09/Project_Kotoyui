import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for tag in sys.argv[1:]:
    md = open(_PRIV + f"i35y/i35z_gemma_teacher_gemma_good_flow_novelty_base_{tag}_wn.md", encoding="utf-8").read()
    L = md.splitlines(); print("##", tag)
    for i, l in enumerate(L):
        if l.startswith("- 相手: ") and re.search(r"(知ってる|何だった)[？?]", l):
            b = next((x for x in L[i+1:i+3] if x.startswith("- ボット")), "")
            m = re.match(r"- ボット: \*\*(.*?)\*\* （(.{0,40})", b)
            own = bool(re.search(r"私の|うちの", l))
            print(f"  {'私の' if own else 'なし'} | {l[6:70]} → {m.group(1) if m else b[:40]} 〔{m.group(2) if m else ''}〕")
