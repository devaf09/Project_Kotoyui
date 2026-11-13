import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import collections, json, os, re, sys
SRC = _os.path.join(_HERE, "i35y_unanswered.py")
code = open(SRC, encoding="utf-8").read().split("rows, asked = [], 0")[0]
g = {"__file__": SRC}; exec(code, g)
Y = _PRIV + "i35y/"
RUNS = {"l": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i37f_wn.md", "i35y_i37f_wn.jsonl"),
        "m": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i39a_wn.md", "i35y_i39a_wn.jsonl"),
        "m2": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i39b_wn.md", "i35y_i39b_wn.jsonl"),
        "m3": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i39c_wn.md", "i35y_i39c_wn.jsonl"),
        "m4": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i39d_wn.md", "i35y_i39d_wn.jsonl"),
        "i40a": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i40a_wn.md", "i35y_i40a_wn.jsonl"),
        "i40c": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i40c_wn.md", "i35y_i40c_wn.jsonl"),
        "i40d": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i40d_wn.md", "i35y_i40d_wn.jsonl"),
        "i40e": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i40e_wn.md", "i35y_i40e_wn.jsonl"),
        "i40f": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i40f_wn.md", "i35y_i40f_wn.jsonl"),
        "i40g": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i40g_wn.md", "i35y_i40g_wn.jsonl"),
        "i40h": ("i35z_gemma_teacher_gemma_good_flow_novelty_base_i40h_wn.md", "i35y_i40h_wn.jsonl")}
ANS = re.compile(r"質問に答え|答えてあげ|答えるのが自然|答えるほうが|答えるといい|答えてみよう|答えるように|について答え|答えたほうが|答える形|答えを返す|答えていない")
def band(x): return "<0.15" if x < 0.15 else "0.15〜0.3" if x < 0.3 else "0.3〜0.6" if x < 0.6 else "0.6〜"
out = {}
for run, (md, st) in RUNS.items():
    if not os.path.exists(Y + md): continue
    asked, rows, allq = 0, [], []
    for t in g["turns"](md):
        q = t["partner"]
        if not re.search(r"[？?]", q) or g["ASKBACK"].search(q) or not g["about_bot"](q): continue
        asked += 1
        k = g["kind"](t, run)
        r = dict(conv=t["conv"], q=q, bot=t["bot"], kind=k, topic=g["topic"](q), why=t["why"], te=t["teacher"])
        allq.append(r)
        if k: rows.append(r)
    sb = [r for r in allq if r["why"].startswith("自分の板: ")]
    bands = collections.defaultdict(lambda: [0, 0])
    for r in sb:
        m = re.search(r"近さ ([0-9.]+)", r["why"]); b = band(float(m.group(1)))
        bands[b][0] += 1; bands[b][1] += bool(ANS.search(r["te"]))
    ss = [json.loads(l) for l in open(Y + st, encoding="utf-8")]
    K = ["往復", "指摘", "受け", "先生の指摘", "中身なし", "繰り返し", "語の繰り返し", "先生の見立てを捨てた", "相手の空", "自分の板"]
    o = dict(asked=asked, un=len(rows), rate=round(len(rows) / asked, 3),
             kinds=dict(collections.Counter(r["kind"] for r in rows).most_common()),
             topics={tp: [sum(1 for r in allq if r["topic"] == tp), sum(1 for r in rows if r["topic"] == tp)] for tp in [k for k, _ in g["TOPICS"]] + ["その他"]},
             sb_about=len(sb), sb_un=sum(1 for r in sb if r["kind"]), sb_ans_teacher=sum(1 for r in sb if ANS.search(r["te"])),
             ans_teacher_all=sum(1 for r in allq if ANS.search(r["te"])),
             bands={k: bands[k] for k in sorted(bands)},
             stats={k: sum(s.get(k, 0) for s in ss) for k in K}, sb_per_conv=[s.get("自分の板", 0) for s in ss],
             novelty=round(sum(s["新規性"] for s in ss) / len(ss), 3),
             sb_ans_half=[sum(1 for r in sb if ANS.search(r["te"]) and r["conv"] <= 10), sum(1 for r in sb if ANS.search(r["te"]) and r["conv"] > 10)],
             sb_half=[sum(1 for r in sb if r["conv"] <= 10), sum(1 for r in sb if r["conv"] > 10)])
    out[run] = o
    out[run + "_ex"] = [f"[{r['topic']}|{r['kind'] or '答えた'}] {run}-{r['conv']} {r['q']} → {r['bot']} 〔{g['short_why'](r['why'])[:150]}〕 先生:{g['short_teacher'](r['te'])[:70]}" for r in allq if r['why'].startswith('自分の板: ')]
for k, v in out.items():
    if k.endswith("_ex"):
        print("== ", k); print("\n".join(v))
    else:
        print(k, json.dumps(v, ensure_ascii=False))
