"""I48x: 型を、働きの並びと語尾の流れとして、人の短い返事から浮かび上がらせる（ロードマップ 52 の土台。測る所まで）。

型の印（手の語リストを増やさない。今ある見分けだけで作る）:
  問い   : I48e の is_question
  受け   : I48l の受けの一言で始まる、または感動詞がある
  返す   : 相手の前の行の中身の語と同じ原形がある（穴 {単語}）
  足す   : 相手の前の行に無い中身の語がある
  近い   : 足した語のうち、相手の行の語に意味の波で近い語がある（I48m の Near。連想のパス）
  時     : 行の状況の過去（I48o の line_sit）
  語尾   : 行の最後の終助詞（I48p の終わりの助詞。無ければ ∅）
丁寧さは型から外す（話し方はくだけた側から始める。丁寧さは語尾のスライドの側。I48n と同じ切り離し）。
型 = 印の組。資料で多い組から順に型の候補にし、持ち主の例文（i48x_type_examples.txt、答え合わせだけ）がどの型に入るかを見る。

実行: python intervention/i48x_type_board.py --cache <rpc.json> [--extra_cache <青空文庫>] --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48e_relation_split as E  # noqa: E402
import i48l_uke_flow as U  # noqa: E402
import i48o_slide_board as O  # noqa: E402
from i48w_reply_function import plain_len  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i48x_type_examples.txt")


def tail(toks):
    T = [t for t in toks if t["pos"] != "PUNCT"]
    k = len(T)
    while k > 0 and T[k - 1]["pos"] == "PART":
        k -= 1
    return "+".join(t["lemma"] for t in T[k:]) or "∅"


def close_any(near, new, P):
    """足した語のどれかが、相手の語のどれかに意味の波で近いか。比べる組の数 m が多いほど偶然に越えやすいので、
    床を σ√(2 ln(V·m)) に上げる（I48m の床 σ√(2 ln V) を、組の数で直した物。手の値は足さない）。"""
    import math
    import numpy as np
    pairs = [(w, p) for w in new for p in P if w in near.wv.ix and p in near.wv.ix]
    if not pairs:
        return False
    fl = math.sqrt(2 * math.log(max(2, len(near.wv.vocab) * len(pairs)))) / math.sqrt(near.D)
    W = near.wv.W
    for w, p in pairs:
        a, b = W[near.wv.ix[w]], W[near.wv.ix[p]]
        if abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9) > fl:
            return True
    return False


def signature(toks, text, prev_toks, near=None, tails=None):
    P = {t["lemma"] for t in prev_toks if t["pos"] in E.CONTENT} if prev_toks else set()
    C = [t["lemma"] for t in toks if t["pos"] in E.CONTENT]
    new = [w for w in C if w not in P]
    q = E.is_question(dict(text=text, toks=toks))
    o = U.uke(toks) is not None or any(t["pos"] == "INTJ" for t in toks)
    close = bool(near is not None and P and close_any(near, new, P))
    tl = tail(toks)
    if tails is not None and tl not in tails:
        tl = "ほか"
    return ("問い" if q else "-", "受け" if o else "-", "返す" if any(w in P for w in C) else "-", "足す" if new else "-",
            "近い" if close else "-", "時" if (O.line_sit(toks) or (0, 0, 0))[0] else "-", tl)


def inventory(works, near=None, lo=9, hi=20, min_share=0.005):
    rows = []
    for w in works:
        L = w["lines"]
        for i in range(1, len(L)):
            n = plain_len(L[i].get("text", ""))
            if lo <= n <= hi and L[i]["toks"]:
                rows.append((L[i]["toks"], L[i].get("text", ""), L[i - 1]["toks"], L[i - 2]["toks"] if i >= 2 else None))
    tails = {t for t, c in Counter(tail(r[0]) for r in rows).items() if c >= max(20, min_share * len(rows))}
    sig = [signature(t, x, p, near, tails) for t, x, p, _ in rows]
    cnt = Counter(sig)
    return rows, sig, cnt, tails


def flow(works, near, tails, lo=9, hi=20):
    """同じ話し手の前の返事（2 行前）の型 → 今の型。前の型を知ると、今の型がどれだけ当てやすくなるか（多い方を言う当たりと比べる）。"""
    import math
    pairs = []
    for w in works:
        L = w["lines"]
        sig = [signature(L[i]["toks"], L[i].get("text", ""), L[i - 1]["toks"], near, tails) if i >= 1 and L[i]["toks"] else None
               for i in range(len(L))]
        for i in range(2, len(L)):
            if sig[i] and sig[i - 2] and lo <= plain_len(L[i].get("text", "")) <= hi:
                pairs.append((sig[i - 2], sig[i]))
    cur = Counter(b for _, b in pairs)
    by = defaultdict(Counter)
    for a, b in pairs:
        by[a][b] += 1
    n = len(pairs)
    H = -sum(c / n * math.log2(c / n) for c in cur.values())
    Hc = -sum(sum(c.values()) / n * sum(x / sum(c.values()) * math.log2(x / sum(c.values())) for x in c.values()) for c in by.values())
    acc0 = cur.most_common(1)[0][1] / n
    acc1 = sum(c.most_common(1)[0][1] for c in by.values()) / n
    qq = [(a[0] == "問い", b[0] == "問い") for a, b in pairs]
    pq = sum(1 for a, b in qq if a and b) / max(1, sum(1 for a, _ in qq if a))
    return dict(組=n, 今の型の情報量=round(H, 3), 前の型を知った時=round(Hc, 3), 多い方の当たり=round(acc0, 4), 前の型から当てた当たり=round(acc1, 4),
                問いの後に問い=round(pq, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra_cache")
    ap.add_argument("--near", action="store_true", help="意味の波（RPC、D 512）で「近い」を見る")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {}
    near = None
    if a.near:
        from i48_wave_board import Waves
        from i48m_echo_phrase import Near
        W = json.load(open(a.cache, encoding="utf-8"))
        random.Random(0).shuffle(W)
        tr = W[500:]
        vocab = sorted({t["lemma"] for w in tr for ln in w["lines"] for t in ln["toks"] if t["pos"] in E.CUE})
        seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in E.CUE] for w in tr for ln in w["lines"] + [dict(toks=[])]]
        wv = Waves(vocab, 512, 0)
        wv.learn(seq)
        near = Near(wv, 512)
    for name, path in (("RPC", a.cache), ("青空文庫", a.extra_cache)):
        if not path:
            continue
        works = json.load(open(path, encoding="utf-8"))
        works = works["works"] if isinstance(works, dict) else works
        rows, sig, cnt, tails = inventory(works, near)
        fl = flow(works, near, tails)
        n = len(sig)
        top = cnt.most_common(40)
        cover = []
        acc = 0
        for k, (s, c) in enumerate(top):
            acc += c
            if k + 1 in (5, 10, 20, 40):
                cover.append((k + 1, round(acc / n, 3)))
        res[name] = dict(流れ=fl, 行=n, 型の数=len(cnt), 多い型=[(" ".join(s), round(c / n, 4)) for s, c in top[:25]], 上位でおおう割合=cover, 語尾の候補=sorted(tails))
    # 持ち主の例文（答え合わせ）: {単語} は仮の名詞に替え、相手の行はその名詞だけの行とする
    try:
        import spacy
        nlp = spacy.load("ja_ginza")
        sigs = defaultdict(list)
        top_rpc = {tuple(s.split(" ")) for s, _ in res["RPC"]["多い型"]} if "RPC" in res else set()
        for ln in open(EXAMPLES, encoding="utf-8"):
            if not ln.strip() or ln.startswith("#"):
                continue
            kind, text = ln.rstrip("\n").split("\t")
            text = text.replace("{単語}", "海")
            toks = [dict(i=t.i, text=t.text, lemma=t.lemma_, pos=t.pos_, dep=t.dep_, head=t.head.i) for t in nlp(text)]
            prev = [dict(i=0, text="海", lemma="海", pos="NOUN", dep="ROOT", head=0)]
            s = signature(toks, text, prev, near, set(res.get("RPC", {}).get("語尾の候補", [])))
            sigs[kind].append((" ".join(s), s in top_rpc))
        res["例文の型"] = {k: dict(型=Counter(s for s, _ in v).most_common(), 多い型に入る割合=round(sum(b for _, b in v) / len(v), 3)) for k, v in sigs.items()}
    except Exception as e:                                   # GiNZA の無い所では飛ばす
        res["例文の型"] = f"飛ばした: {e}"
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
