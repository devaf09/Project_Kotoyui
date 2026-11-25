"""I48ac の組み立ての比べ（数え上げの版、詰め物＝要点の間）。帯の組を 1 つずつ足し、帯の種類ごとの重みをそろえる／そろえない、prior の重みを比べる。
部分一致と全体一致の差（述語・印を替えた時の切り替わり）と、持ち主の例の点の内訳も出す。

まわり: 内容語の相対の位置（−3〜+3）、"述"（文の最後の内容語）、"印"（意味の印の帯の和。無ければ「印:なし」）。
python intervention/past/i48ac_ablate.py --conllu ... --content ... --func ... --out out.json [--train N] [--test N] [--D 8192]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ac_center_grain as CG  # noqa: E402

STEPS = [("線形 ±1", dict(W=3, lin_W=1, lin=(), pairs=[], triples=[])),
         ("＋隣の積 −1×+1", dict(pairs=[(-1, 1)])),
         ("＋離れた積 −1×+2", dict(pairs=[(-1, 2)])),
         ("＋離れた積 −1×+3", dict(pairs=[(-1, 3)])),
         ("＋−1×文末の述語", dict(pairs=[(-1, "述")])),
         ("＋−1×印", dict(pairs=[(-1, "印")])),
         ("＋3 つの積 −1×+1×述語", dict(triples=[(-1, 1, "述")])),
         ("＋3 つの積 −1×+1×印", dict(triples=[(-1, 1, "印")]))]
CANDS = ("の", "と", "に", "で", "を", "が", "は")


def marks_attr(S, mk):
    fs = [S.attrs("印:" + m, "印") for m in sorted(mk)] or [S.attrs("印:なし", "印")]
    return np.concatenate([f for f, _ in fs]), np.concatenate([v for _, v in fs])


def items(S, s):
    cont, gaps, _ = G.split(s)
    if not cont:
        return []
    mk = S.func.marks(s["toks"])
    A = [S.attrs(l, u) for l, _, u in cont]
    M = marks_attr(S, mk)
    n = len(cont)
    out = []
    for i in range(n):
        ctx = [(-(i - j + 1), A[j]) for j in range(max(0, i - 2), i + 1)]
        ctx += [(j - i, A[j]) for j in range(i + 1, min(n, i + 4))]
        ctx += [("述", A[-1]), ("印", M)]
        out.append(dict(ctx=ctx, y=gaps[i], l=cont[i][0], r=cont[i + 1][0] if i + 1 < n else None, p=cont[-1][0], mk=tuple(sorted(mk)),
                        last=i + 1 == n, n=n))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--D", type=int, default=8192)
    a = ap.parse_args()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0][:-2000]
    if a.train:
        tr = tr[: a.train]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    TR = [x for s in tr for x in items(S, s)]
    TE = [x for s in te for x in items(S, s)]
    cnt = Counter(x["y"] for x in TR)
    true_share = Counter(x["y"] for x in TE)
    res = dict(学びの所=len(TR), 試しの所=len(TE), D=a.D, 上位の属性=G.TOPK, 構成={}, prior={})
    # 人の資料: (左, 右) ごと・(左, 右, 述語) ごと・(左, 右, 過去か) ごとのいちばん多い詰め物
    t2, t3, tp = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for x in TR:
        if x["last"]:
            continue
        t2[(x["l"], x["r"])][x["y"]] += 1
        t3[(x["l"], x["r"], x["p"])][x["y"]] += 1
        tp[(x["l"], x["r"], "過去" in x["mk"])][x["y"]] += 1
    spec = {}
    best = None
    verbs = [w["name"] for w in sorted(cb, key=lambda z: -z.get("freq", 0)) if w["pos"] == "動詞"][:50]
    rng = np.random.default_rng(0)
    for name, add in STEPS:
        for k, v in add.items():
            spec[k] = list(spec.get(k, [])) + list(v) if isinstance(v, list) else v
        for norm in (False, True):
            t0 = time.time()
            pg = G.PairGrain(a.D).fit(((*CG.join(CG.spec_features(S, x["ctx"], spec), norm), x["y"]) for x in TR), cnt)
            hit = Counter()
            for x in TE:
                p = pg.predict(*CG.join(CG.spec_features(S, x["ctx"], spec), norm))[0]
                hit["n"] += 1
                hit["当たり"] += p == x["y"]
                hit["間:n" if not x["last"] else "文末:n"] += 1
                hit["間" if not x["last"] else "文末"] += p == x["y"]
            o = dict(秒=round(time.time() - t0), 当たり=round(hit["当たり"] / hit["n"], 3), 間=round(hit["間"] / max(1, hit["間:n"]), 3),
                     文末=round(hit["文末"] / max(1, hit["文末:n"]), 3))
            res["構成"][f"{name}・{'帯の種類ごとにそろえる' if norm else 'そろえない'}"] = o
            print(name, norm, o, flush=True)
            if best is None or o["当たり"] > best[0]:
                best = (o["当たり"], name, norm, {k: (list(v) if isinstance(v, list) else v) for k, v in spec.items()}, pg)
            last = (o["当たり"], name, norm, {k: (list(v) if isinstance(v, list) else v) for k, v in spec.items()}, pg)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["いちばん良い構成"] = [best[1], best[2]]
    tw = Counter()
    for x in TE:                                            # 基準: 隣の語の表（同じ学び）
        t = t2.get((x["l"], x["r"])) if not x["last"] else None
        tw["n"] += 1
        tw["当たり"] += bool(t) and t.most_common(1)[0][0] == x["y"]
    res["基準_隣の語の表（間だけ、文末は外れ扱い）"] = round(tw["当たり"] / tw["n"], 3)
    for tag, (_, bname, bnorm, bspec, pg) in (("いちばん良い構成", best), ("全部入り（帯の種類ごとにそろえる）", last)):
        res[tag] = analyse(S, pg, bname, bnorm, bspec, TE, true_share, t3, tp, verbs, rng)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def analyse(S, pg, bname, bnorm, bspec, TE, true_share, t3, tp, verbs, rng):
    res = dict(構成=[bname, bnorm], prior={})
    feat = lambda ctx: CG.join(CG.spec_features(S, ctx, bspec), bnorm)
    # prior の重みと、選んだ詰め物の分布
    for pw in (0.0, 0.5, 1.0):
        sel = Counter()
        hit = 0
        for x in TE:
            p = pg.predict(*feat(x["ctx"]), prior_w=pw)[0]
            sel[p] += 1
            hit += p == x["y"]
        diff = sorted(((sel[k] - true_share[k]) / len(TE), k) for k in set(sel) | set(true_share))
        res["prior"][str(pw)] = dict(当たり=round(hit / len(TE), 3), の_選んだ割合=round(sel["の"] / len(TE), 3), の_実際=round(true_share["の"] / len(TE), 3),
                                    選びすぎ=[[k, round(d, 3)] for d, k in diff[::-1][:5]], 選ばなさすぎ=[[k, round(d, 3)] for d, k in diff[:5]])
        print("prior", pw, res["prior"][str(pw)], flush=True)
    # 部分一致と全体一致: 述語・印を替えた時に詰め物が変わる割合（モデル）と、人の資料で変わる割合
    ch = Counter()
    for x in TE:
        if x["last"]:
            continue
        p0 = pg.predict(*feat(x["ctx"]))[0]
        nv = verbs[int(rng.integers(len(verbs)))]
        ctx_v = [(o, S.attrs(nv, "VERB") if o == "述" else fv) for o, fv in x["ctx"]]
        pv = pg.predict(*feat(ctx_v))[0]
        mk2 = set(x["mk"]) ^ {"過去"}
        ctx_m = [(o, marks_attr(S, mk2) if o == "印" else fv) for o, fv in x["ctx"]]
        pm = pg.predict(*feat(ctx_m))[0]
        ch["n"] += 1
        ch["述語を替えて変わる"] += pv != p0
        ch["過去を替えて変わる"] += pm != p0
        a3, b3 = t3.get((x["l"], x["r"], x["p"])), t3.get((x["l"], x["r"], nv))
        if a3 and b3 and sum(a3.values()) >= 2 and sum(b3.values()) >= 2:
            ch["人_述語:n"] += 1
            ch["人_述語:変わる"] += a3.most_common(1)[0][0] != b3.most_common(1)[0][0]
            ch["人_述語:モデルも変わる"] += (a3.most_common(1)[0][0] != b3.most_common(1)[0][0]) and pv != p0
        past = "過去" in x["mk"]
        a4, b4 = tp.get((x["l"], x["r"], past)), tp.get((x["l"], x["r"], not past))
        if a4 and b4 and sum(a4.values()) >= 3 and sum(b4.values()) >= 3:
            ch["人_過去:n"] += 1
            ch["人_過去:変わる"] += a4.most_common(1)[0][0] != b4.most_common(1)[0][0]
    res["切り替わり"] = dict(件数=ch["n"], モデル_述語を替えて変わる=round(ch["述語を替えて変わる"] / max(1, ch["n"]), 3),
                         モデル_過去を替えて変わる=round(ch["過去を替えて変わる"] / max(1, ch["n"]), 3),
                         人_述語で答え合わせできた=ch["人_述語:n"], 人_述語を替えて変わる=round(ch["人_述語:変わる"] / max(1, ch["人_述語:n"]), 3),
                         人_過去で答え合わせできた=ch["人_過去:n"], 人_過去を替えて変わる=round(ch["人_過去:変わる"] / max(1, ch["人_過去:n"]), 3))
    print("切り替わり", res["切り替わり"], flush=True)
    # 持ち主の例: 候補ごとの点の内訳
    ex = []
    for keys, mk in (((("昨日", "NOUN"), ("友達", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")), {"過去"}),
                     ((("昨日", "NOUN"), ("公園", "NOUN"), ("犬", "NOUN"), ("散歩", "NOUN"), ("する", "VERB")), {"過去"})):
        s_ = dict(toks=[])
        A = [S.attrs(l, u) for l, u in keys]
        M = marks_attr(S, mk)
        n = len(keys)
        rows = []
        for i in range(n - 1):
            ctx = [(-(i - j + 1), A[j]) for j in range(max(0, i - 2), i + 1)] + [(j - i, A[j]) for j in range(i + 1, min(n, i + 4))]
            ctx += [("述", A[-1]), ("印", M)]
            groups = CG.spec_features(S, ctx, bspec)
            if bnorm:
                groups = {k: (f, v / max(1e-9, float(v.sum()))) for k, (f, v) in groups.items()}
            part = {}
            for c in CANDS:
                if c not in pg.cls:
                    continue
                ci = pg.cls.index(c)
                d = {"prior": round(float(pg.prior[ci]), 2)}
                for g, (f, v) in groups.items():
                    d[g] = round(float(pg.logp[ci, f] @ v), 2)
                d["計"] = round(sum(d.values()), 2)
                part[c] = d
            choose = pg.predict(*feat(ctx))[0]
            rows.append(dict(間=f"{keys[i][0]} _ {keys[i + 1][0]}", 選んだ=choose, 候補ごと=part))
        ex.append(dict(要点=[k for k, _ in keys] + sorted(mk), 間ごと=rows))
    res["持ち主の例"] = ex
    print(json.dumps(ex, ensure_ascii=False)[:2000])
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
