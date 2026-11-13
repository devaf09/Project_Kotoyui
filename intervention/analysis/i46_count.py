import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import re, sys, json
D = _PRIV + "i35y/"
LOG = D + "i35z_gemma_teacher_gemma_good_flow_novelty_base_{}_wn.md"
ST = D + "i35y_{}_wn.jsonl"


def lines(run):
    out, conv = [], 0
    for s in open(LOG.format(run), encoding="utf-8"):
        m = re.match(r"## 会話 (\d+)", s)
        if m:
            conv = int(m.group(1)); continue
        m = re.match(r"- 相手: (.*)", s)
        if m:
            out.append((conv, "相手", m.group(1).strip(), "")); continue
        m = re.match(r"- ボット: \*\*(.*?)\*\* （(.*)", s)
        if m:
            out.append((conv, "ボット", m.group(1), m.group(2)))
    return out


def katte(run):
    L = lines(run)
    res = []
    for i, (c, w, t, _) in enumerate(L):
        if w == "相手" and re.search(r"って[？?]", t):
            ctx = L[max(0, i - 1):i + 3]
            res.append((c, [(w2, t2) for _, w2, t2, _ in ctx]))
    return res


def stats(run):
    return [json.loads(x) for x in open(ST.format(run), encoding="utf-8")]


if __name__ == "__main__":
    cmd, run = sys.argv[1], sys.argv[2]
    if cmd == "katte":
        r = katte(run)
        print(f"### {run}（{len(r)} 件）")
        for c, ctx in r:
            print(f"- 会話 {c}: " + " ／ ".join(f"{w[0]}「{t}」" for w, t in ctx))
    elif cmd == "mem":
        for s in stats(run):
            m = s["記憶板"]
            print(s["会話"], "薄れ", m.get("薄れ"), m.get("薄れの経験"), "時", m["時"], "日", m["日"], "再記憶", m["再記憶"],
                  "覚えてる？", s.get("覚えてる？"), "台本", s.get("台本"), "話題", s.get("話題の問い"))
    elif cmd == "keys":
        print(sorted(stats(run)[-1].keys()))


def events(run):
    ev, conv, turn, ann, cur = [], 0, 0, [], None
    for s in open(LOG.format(run), encoding="utf-8"):
        m = re.match(r"## 会話 (\d+)", s)
        if m:
            conv, turn, ann = int(m.group(1)), 0, []; continue
        m = re.match(r"  - (覚えてるかの問い|話題の問い): (.*)", s)
        if m:
            ann.append((m.group(1), m.group(2)[:80])); continue
        m = re.match(r"- 相手: (.*)", s)
        if m:
            turn += 1
            if cur is not None:
                cur["next"] = m.group(1).strip(); ev.append(cur); cur = None
            if ann:
                cur = dict(conv=conv, turn=turn, q=m.group(1).strip(), ann=ann)
            ann = []; continue
        m = re.match(r"- ボット: \*\*(.*?)\*\* （(.*)", s)
        if m and cur is not None and "bot" not in cur:
            cur["bot"] = m.group(1); cur["why"] = m.group(2)[:70]
    if cur: ev.append(cur)
    return ev


if __name__ == "__main__" and sys.argv[1] == "ev":
    for e in events(sys.argv[2]):
        kinds = "/".join(sorted({a[0] for a in e["ann"]}))
        print(f'{e["conv"]}:{e["turn"]} [{kinds}] 相「{e["q"]}」→ボ「{e.get("bot")}」→相「{e.get("next","")}」')


def katte2(run):
    """「〜って？」の行ごとに: 前のボの行、相の行、ボの返事と why、相の次の行、分類。"""
    L = lines(run)
    res = []
    for i, (c, w, t, _) in enumerate(L):
        if w != "相手" or not re.search(r"って[？?]", t):
            continue
        prev = L[i - 1][2] if i > 0 and L[i - 1][1] == "ボット" and L[i - 1][0] == c else ""
        b = L[i + 1] if i + 1 < len(L) and L[i + 1][1] == "ボット" else None
        nx = L[i + 2][2] if i + 2 < len(L) and L[i + 2][1] == "相手" and L[i + 2][0] == c else ""
        bt, why = (b[2], b[3]) if b else ("", "")
        if "語板の属性" in why and "説明する" in why:
            k = "語板の説明"
        elif "言い直す" in why and re.search(r"(何のことかなって|のかなって|ってこと)。$", bt):
            k = "言い直し"
        elif "介入" in why:
            k = "作り直し（介入）"
        elif re.match(r"(はい|いいえ|ううん|いや|ああ)[、。]?", bt) or re.search(r"(ない|ではない|じゃない|ん)。$", bt):
            k = "はい・いいえ・否定"
        elif re.match(r"^[^、。]{1,8}[？?]$", bt):
            k = "聞き返し"
        else:
            k = "その他"
        res.append(dict(conv=c, prev=prev, q=t, bot=bt, why=why, next=nx, kind=k,
                        again=bool(re.search(r"って[？?]", nx)), new_conv=not nx))
    return res


def subkind(k):
    if k["kind"] != "言い直し" and not re.search(r"(何のことかなって|のかなって|ってこと)。$", k["bot"]):
        return None
    if "言い直す" not in k["why"]:
        return None
    if "何のことかなって" in k["bot"]:
        return "何のことかなって"
    if "を理由にして言い直す" in k["why"] or "近い所" in k["why"]:
        return "自分の板の理由つき"
    if k["bot"].endswith("のかなって。"):
        return "〜のかなって"
    return "〜ってこと"


if __name__ == "__main__" and sys.argv[1] in ("katte2", "restates"):
    import collections
    run = sys.argv[2]
    K = katte2(run)
    if sys.argv[1] == "katte2":
        C = collections.Counter(k["kind"] for k in K)
        S = collections.Counter(subkind(k) for k in K if k["kind"] == "言い直し")
        f = lambda L: (sum(1 for k in L if k["next"] and not k["again"]), sum(1 for k in L if k["again"]), sum(1 for k in L if not k["next"]))
        R = [k for k in K if k["kind"] == "言い直し"]; G = [k for k in K if k["kind"] == "語板の説明"]
        rep = sum(1 for k in K if re.match(r"「[^」]*(ってこと|のかなって)。」（点[^）]*）は自分の直近の行の繰り返し", k["why"]))
        lost = sum(1 for k in K if re.match(r"「[^」]*(ってこと|のかなって)。」（点[^）]*）より", k["why"]))
        print(run, "「〜って？」", len(K), dict(C), "言い直しの種類", dict(S))
        print("直後の相手（受けた/また聞き返した/会話の終わり）: 言い直し", f(R), "語板の説明", f(G),
              "それ以外", f([k for k in K if k["kind"] not in ("言い直し", "語板の説明")]))
        print("言い直しが繰り返しで捨てられた", rep, "行づくりに点で負けた", lost, "「はい」で始まる返事", sum(1 for k in K if k["bot"].startswith("はい")))
    else:
        for k in K:
            if k["kind"] in ("言い直し", "語板の説明"):
                print(k["conv"], subkind(k) or k["kind"], "|", k["prev"], "|", k["q"], "|", k["bot"], "|", k["next"])


def after_recall(run):
    """思い出しの答え（「〜、だよね。」「ごめん、覚えてない。」）の直後: 相手の行（答え合わせ）と、それへのボットの行、相手の次の行。"""
    L = lines(run)
    res = []
    for i, (c, w, t, why) in enumerate(L):
        if w == "ボット" and ("大事な板" in why or "記憶板の「" in why or "問いの語が無い" in why) and i + 2 < len(L) \
                and L[i + 1][1] == "相手" and L[i + 1][0] == c and L[i + 2][1] == "ボット":
            nx = L[i + 3][2] if i + 3 < len(L) and L[i + 3][1] == "相手" and L[i + 3][0] == c else ""
            b = L[i + 2][2]
            res.append(dict(conv=c, ans=t, check=L[i + 1][2], bot=b, why=L[i + 2][3], next=nx,
                            pick=("正解" in b or "覚え" in b), again=bool(re.search(r"って[？?]", nx))))
    return res


if __name__ == "__main__" and sys.argv[1] == "after":
    R = after_recall(sys.argv[2])
    print(sys.argv[2], "思い出しの直後", len(R), "「正解・覚え」を拾った", sum(r["pick"] for r in R), "相手がまた聞き返した", sum(r["again"] for r in R))
    for r in R:
        print(r["conv"], "|", r["ans"], "|", r["check"], "|", r["bot"], "|", r["next"], "| why:", r["why"][:50])
