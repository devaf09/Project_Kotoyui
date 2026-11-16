"""I48ae: 意味の印（過去・否定・丁寧・進行・問い）を文の全体に効かせる。詰め物（要点の間）の数え上げの中心のグレイン（i48ac）で比べる。

印の入れ方（置き方）:
  なし       : 印を入れない（線形 ±1 だけ）
  帯         : 印を 1 本の帯として周りに足す（線形）
  積         : 左の語 × 印 の積の帯を足す（I48ac の −1×印）
  シフト     : 文（または節）の語すべての帯を、印ごとの決まったシフトの和だけずらす（I48j・I48o のスライドと同じ考え）。印ごとに
               違う所に帯が立つので、同じ語の並びでも時が違えば別の帯で学ぶ
  シフト＋元 : シフトした帯と、ずらさない帯の両方（重み 0.5 ずつ。時の違いを越えて分け合う分を残す）
印の単位:
  文   : 文で 1 組（今まで）
  節   : 間ごとに、その間の右のいちばん近い述語（VERB・ADJ。無ければ文の最後の内容語）と、その後ろの詰め物の印
測る物: 間・文末の当たり、過去（と否定・丁寧）を替えた時に途中の詰め物が変わる割合（モデルと人の資料）、時の語のある文の文末の当たり、例。
手で置く物: 印ごとのシフトの歩幅（MARK_STEP）、時の語の一覧（TIME_WORDS、測るためだけ）。
python intervention/i48ae_marks.py --conllu ... --content ... --func ... --out out.json [--train N] [--test N] [--D 8192]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ac_center_grain as CG  # noqa: E402

MARK_NAMES = ("過去", "否定", "丁寧", "進行", "問い")
TIME_WORDS = ("昨日", "明日", "今日", "今", "もう", "まだ", "去年", "来年", "昔", "将来", "先週", "来週", "さっき", "あとで", "すでに", "これから")
MODES = ("なし", "帯", "積", "シフト", "シフト＋元")


def mark_step(S, m):
    return (S.mix * (3 + 2 * MARK_NAMES.index(m))) % S.D


def mshift(S, mk):
    return sum(mark_step(S, m) for m in mk) % S.D


def clause_marks(s, func):
    """間 i（内容語 i の後ろ）ごとの節の印: 内容語 i 以降でいちばん近い述語（VERB・ADJ）の、後ろの詰め物の印。"""
    T = s["toks"]
    ci = [i for i, t in enumerate(T) if G.is_content(t)]
    tails = []
    for k, i in enumerate(ci):
        j = ci[k + 1] if k + 1 < len(ci) else len(T)
        tails.append(T[i + 1:j])
    out = []
    for k in range(len(ci)):
        p = next((q for q in range(k, len(ci)) if T[ci[q]]["upos"] in ("VERB", "ADJ")), len(ci) - 1)
        out.append(tuple(sorted(func.marks(tails[p]))))
    return out


def items(S, s, unit):
    cont, gaps, _ = G.split(s)
    if not cont:
        return []
    mk_s = tuple(sorted(S.func.marks(s["toks"])))
    mk_c = clause_marks(s, S.func) if unit == "節" else [mk_s] * len(cont)
    A = [S.attrs(l, u) for l, _, u in cont]
    n = len(cont)
    has_time = any(l in TIME_WORDS for l, _, _ in cont)
    return [dict(A=A, i=i, y=gaps[i], l=cont[i][0], r=cont[i + 1][0] if i + 1 < n else None, mk=mk_c[i], mk_s=mk_s, last=i + 1 == n, time=has_time)
            for i in range(n)]


def feats(S, x, mode, mk=None):
    mk = x["mk"] if mk is None else mk
    A, i = x["A"], x["i"]
    ctx = [(-1, A[i])] + ([(1, A[i + 1])] if i + 1 < len(A) else [])
    marks = [S.attrs("印:" + m, "印") for m in mk] or [S.attrs("印:なし", "印")]
    M = (np.concatenate([f for f, _ in marks]), np.concatenate([v for _, v in marks]))
    ctx_m = ctx + [("印", M)]
    if x["last"]:
        ctx_m = ctx_m + [("文末", S.attrs("印:文末", "印"))]
    base = CG.spec_features(S, ctx_m, dict(W=1, lin_W=1, lin=("文末",), pairs=[], triples=[]))
    if mode == "なし":
        g = base
    elif mode == "帯":
        g = CG.spec_features(S, ctx_m, dict(W=1, lin_W=1, lin=("印", "文末"), pairs=[], triples=[]))
    elif mode == "積":
        g = dict(base)
        g.update(CG.spec_features(S, ctx_m, dict(W=1, lin_W=0, lin=(), pairs=[(-1, "印")], triples=[])))
    else:
        sh = mshift(S, mk)
        g = {("シフト", k): ((f + sh) % S.D, v) for k, (f, v) in base.items()}
        if mode == "シフト＋元":
            g = {k: (f, v * 0.5) for k, (f, v) in g.items()}
            g.update({k: (f, v * 0.5) for k, (f, v) in base.items()})
    return CG.join(g)


def toggle(mk, m):
    s = set(mk)
    s ^= {m}
    return tuple(sorted(s))


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--units", nargs="*", default=["文", "節"])
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    a = ap.parse_args()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0][:-2000][: a.train]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    res = dict(学びの文=len(tr), 試しの文=len(te), D=a.D, 手で置いた物=dict(印のシフトの歩幅={m: mark_step(S, m) for m in MARK_NAMES}, 時の語=TIME_WORDS), 構成={})
    for unit in a.units:
        TR = [x for s in tr for x in items(S, s, unit)]
        TE = [x for s in te for x in items(S, s, unit)]
        cnt = Counter(x["y"] for x in TR)
        # 人の資料: 同じ (左, 右) で、その印の有り無しでいちばん多い詰め物が変わる割合
        hum = {}
        for m in ("過去", "否定", "丁寧"):
            t = defaultdict(Counter)
            for x in TR:
                if not x["last"]:
                    t[(x["l"], x["r"], m in x["mk"])][x["y"]] += 1
            n = ch = 0
            for (l, r, has), c in t.items():
                if has and sum(c.values()) >= 3:
                    c2 = t.get((l, r, False))
                    if c2 and sum(c2.values()) >= 3:
                        n += 1
                        ch += c.most_common(1)[0][0] != c2.most_common(1)[0][0]
            hum[m] = [n, round(ch / max(1, n), 3)]
        res.setdefault("人の資料_印で途中の詰め物が変わる割合", {})[unit] = hum
        for mode in a.modes:
            t0 = time.time()
            pg = G.PairGrain(a.D).fit(((*feats(S, x, mode), x["y"]) for x in TR), cnt)
            c = Counter()
            for x in TE:
                p = pg.predict(*feats(S, x, mode))[0]
                key = "文末" if x["last"] else "間"
                c[key + ":n"] += 1
                c[key] += p == x["y"]
                if x["last"]:
                    k2 = "時の語あり" if x["time"] else "時の語なし"
                    c[k2 + ":n"] += 1
                    c[k2] += p == x["y"]
                else:
                    for m in ("過去", "否定", "丁寧"):
                        c[m + ":n"] += 1
                        c[m] += pg.predict(*feats(S, x, mode, toggle(x["mk"], m)))[0] != p
            o = dict(秒=round(time.time() - t0), 間=round(c["間"] / max(1, c["間:n"]), 3), 文末=round(c["文末"] / max(1, c["文末:n"]), 3),
                     文末_時の語あり=[c["時の語あり:n"], round(c["時の語あり"] / max(1, c["時の語あり:n"]), 3)],
                     文末_時の語なし=[c["時の語なし:n"], round(c["時の語なし"] / max(1, c["時の語なし:n"]), 3)],
                     印を替えて途中の詰め物が変わる={m: round(c[m] / max(1, c[m + ":n"]), 3) for m in ("過去", "否定", "丁寧")})
            res["構成"][f"印の単位 {unit}・{mode}"] = o
            print(unit, mode, json.dumps(o, ensure_ascii=False), flush=True)
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(unit, "人の資料", hum, flush=True)
    # 例: 「大きくなったら王様になりたい」などの節ごとの印
    ex = []
    for s in te[:2000]:
        if any(t["form"] in ("たら", "ても") for t in s["toks"]) and len(ex) < 5:
            ex.append(dict(文=s["text"], 文で1組=list(func.marks(s["toks"])), 節ごと=[list(m) for m in clause_marks(s, func)]))
    res["節ごとの印の例"] = ex
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(ex, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
