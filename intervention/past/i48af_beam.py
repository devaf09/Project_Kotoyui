"""I48af: 先読みの幅を、文のグレインの「頭からの途中の文の整合」で絞る（I48ad の続き）。節ごとの印（I48ae の 節・帯）と合わせる。

- 候補のグレイン: 詰め物の数え上げの中心のグレイン（内容語 ±1 ＋ 節ごとの印の帯。I48ae の「節・帯」）。prior は帯の確かさで弱める（I48ad）。
- 途中の点（幅を絞る所）:
    元の点   : 置いた候補の、候補のグレインでの割合の平均（I48ad と同じ）
    左の整合 : 左だけの文のグレイン（間の詰め物を中心に、内容語 ±1・節の印・左に置いた詰め物）での、置いた詰め物それぞれの割合の平均。
               置くたびに、頭からの途中の文の流れとの重なりで点を付け直す
- 最後の点: 文のグレイン（左右の詰め物）での、置いた詰め物それぞれの割合の平均（I48ad の a 平均）と、弱い所で下位 3 割を落として平均（a 両方）。
- 測る物: 幅の中に元の並びが残る割合、間・文末・文が元と同じ、語の表と並べる。持ち主の例。
手で置く物: 幅、k、DROP（I48ad と同じ）。
python intervention/past/i48af_beam.py --conllu ... --content ... --func ... --out out.json [--train N] [--test N] [--k 5 12] [--beam 10 20]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ac_center_grain as CG  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402

DROP = AD.DROP
LEFT_SPEC = dict(W=1, lin_W=1, lin=("F-", "印"), pairs=[(-1, 1), ("F-", -1)], triples=[])
FULL_SPEC = dict(W=1, lin_W=1, lin=("F-", "F+", "印"), pairs=[(-1, 1), ("F-", -1), (1, "F+")], triples=[])


def ctx_of(S, x):
    A, i = x["A"], x["i"]
    marks = [S.attrs("印:" + m, "印") for m in x["mk"]] or [S.attrs("印:なし", "印")]
    M = (np.concatenate([f for f, _ in marks]), np.concatenate([v for _, v in marks]))
    ctx = [(-1, A[i]), ("印", M)] + ([(1, A[i + 1])] if i + 1 < len(A) else [])
    return ctx


def sfeat(S, x, fill, left, right, spec):
    ctx = ctx_of(S, x)
    if left is not None:
        ctx.append(("F-", fill.wave(left)))
    if right is not None:
        ctx.append(("F+", fill.wave(right)))
    return CG.join(CG.spec_features(S, ctx, spec), False)


def share(pg, f, v, y):
    if y not in pg.ci:
        return 0.0
    p = AD.posterior(pg, f, v)
    return float(p[pg.ci[y]])


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--k", type=int, nargs="*", default=[5, 12])
    ap.add_argument("--beam", type=int, nargs="*", default=[10, 20])
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold = tr_all[-2000:]
    tr = tr_all[:-2000][: a.train] if a.train else tr_all[:-2000]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    TRs = [AE.items(S, s, "節") for s in tr]
    TR = [x for xs in TRs for x in xs]
    cnt = Counter(x["y"] for x in TR)
    pair_n = Counter((x["l"], x["r"]) for x in TR if not x["last"])
    print(f"読み込み {time.time() - t0:.0f} 秒、学び {len(tr)} 文、間 {len(TR)}", flush=True)
    t1 = time.time()
    feat = lambda x: AE.feats(S, x, "帯")
    pg = G.PairGrain(a.D).fit(((*feat(x), x["y"]) for x in TR), cnt)
    lg = G.PairGrain(a.D).fit(((*sfeat(S, x, fill, xs[x["i"] - 1]["y"] if x["i"] > 0 else None, None, LEFT_SPEC), x["y"]) for xs in TRs for x in xs), cnt)
    fg = G.PairGrain(a.D).fit(((*sfeat(S, x, fill, xs[x["i"] - 1]["y"] if x["i"] > 0 else None,
                                       xs[x["i"] + 1]["y"] if x["i"] + 1 < len(xs) else None, FULL_SPEC), x["y"]) for xs in TRs for x in xs), cnt)
    for g_ in (pg, lg, fg):
        g_.ci = {c: i for i, c in enumerate(g_.cls)}
    print(f"グレイン 3 枚 {time.time() - t1:.0f} 秒", flush=True)
    # 帯の確かさで prior を弱める（I48ad と同じ選び方）
    HO = [x for s in hold for x in AE.items(S, s, "節")]
    best_w = {}
    for b in range(len(AD.EV_BINS)):
        hx = [x for x in HO if AD.ev_bin(pair_n[(x["l"], x["r"])] if not x["last"] else 10) == b][:1500]
        F = [feat(x) for x in hx]
        best_w[b] = max((sum(pg.cls[int(np.argmax(w * pg.prior + pg.logp[:, f] @ v))] == x["y"] for (f, v), x in zip(F, hx)), w)
                        for w in (0.0, 0.25, 0.5, 1.0))[1] if hx else 1.0
    nt = CG.NeighborTable()
    for x in TR:
        nt.add(x["l"], x["r"] if not x["last"] else ("文末",) + x["mk"], x["y"])
    TE = [(s, AE.items(S, s, "節")) for s in te]
    res = dict(学びの文=len(tr), 試しの文=len(te), D=a.D, 確かさの段ごとの_prior=best_w, 構成={})

    def run(xs, k, bw, prune):
        cands = []
        for x in xs:
            pw = best_w[AD.ev_bin(pair_n[(x["l"], x["r"])] if not x["last"] else 10)]
            p = AD.posterior(pg, *feat(x), pw)
            p[-1] = 0
            cands.append([(pg.cls[i], float(p[i])) for i in np.argsort(-p)[:k]])
        beam = [((), 0.0, [])]
        for i, cs in enumerate(cands):
            nb = []
            for seq, sc, ls in beam:
                for y, q in cs:
                    if prune == "左の整合":
                        s_ = share(lg, *sfeat(S, xs[i], fill, seq[-1] if seq else None, None, LEFT_SPEC), y)
                        nb.append((seq + (y,), sc, ls + [s_]))
                    else:
                        nb.append((seq + (y,), sc + q, ls))
            key = (lambda z: -float(np.mean(z[2]))) if prune == "左の整合" else (lambda z: -z[1])
            nb.sort(key=key)
            beam = nb[:bw]
        seqs = [seq for seq, _, _ in beam]
        scored = []
        for seq in seqs:
            sp = [share(fg, *sfeat(S, x, fill, seq[i - 1] if i > 0 else None, seq[i + 1] if i + 1 < len(seq) else None, FULL_SPEC), seq[i])
                  for i, x in enumerate(xs)]
            scored.append(dict(seq=seq, 弱い所=min(sp), 平均=float(np.mean(sp))))
        thr = sorted(z["弱い所"] for z in scored)[int(len(scored) * DROP)] if len(scored) > 1 else -1
        return cands, seqs, dict(a_平均=max(scored, key=lambda z: z["平均"])["seq"],
                                 a_両方=max([z for z in scored if z["弱い所"] >= thr] or scored, key=lambda z: z["平均"])["seq"]), scored

    for prune in ("元の点", "左の整合"):
        for k in a.k:
            for bw in a.beam:
                t2 = time.time()
                c = Counter()
                for s, xs in TE:
                    if not xs:
                        continue
                    cont, gaps, head = G.split(s)
                    cands, seqs, picks, _ = run(xs, k, bw, prune)
                    c["文"] += 1
                    c["間n"] += len(gaps) - 1
                    c["元の並び"] += tuple(gaps) in seqs
                    picks = dict(picks, 先読みなし=tuple(cs[0][0] for cs in cands),
                                 語の表=tuple(nt.predict(x["l"], x["r"] if not x["last"] else ("文末",) + x["mk"]) for x in xs))
                    for nm, seq in picks.items():
                        for i, (g, y) in enumerate(zip(seq, gaps)):
                            c[nm + (":間" if i + 1 < len(gaps) else ":文末")] += g == y
                        sent = "".join(surf.form(l, u, g) + g for (l, f_, u), g in zip(cont, seq))
                        c[nm + ":文"] += G.norm(head + sent) == G.norm(s["text"])
                o = dict(秒=round(time.time() - t2), 残った中に元の並び=round(c["元の並び"] / max(1, c["文"]), 3))
                for nm in ("先読みなし", "a_平均", "a_両方", "語の表"):
                    o[nm] = dict(間=round(c[nm + ":間"] / max(1, c["間n"]), 3), 文末=round(c[nm + ":文末"] / max(1, c["文"]), 3), 文=round(c[nm + ":文"] / max(1, c["文"]), 3))
                res["構成"][f"{prune}・k {k}・幅 {bw}"] = o
                print(prune, k, bw, json.dumps(o, ensure_ascii=False), flush=True)
                json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    exs = []
    for keys, mk in (((("昨日", "NOUN"), ("友達", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")), ("過去",)),
                     ((("昨日", "NOUN"), ("公園", "NOUN"), ("犬", "NOUN"), ("散歩", "NOUN"), ("する", "VERB")), ("過去",))):
        A = [S.attrs(l, u) for l, u in keys]
        n = len(keys)
        xs = [dict(A=A, i=i, l=keys[i][0], r=keys[i + 1][0] if i + 1 < n else None, mk=mk, last=i + 1 == n) for i in range(n)]
        cands, seqs, picks, scored = run(xs, max(a.k), max(a.beam), "左の整合")
        rows = sorted(scored, key=lambda z: -z["平均"])[:10]
        exs.append(dict(要点=[k_ for k_, _ in keys] + list(mk),
                        残った上位10=[dict(文="".join(surf.form(l, u, g) + g for (l, u), g in zip(keys, z["seq"])), 弱い所=round(z["弱い所"], 3), 平均=round(z["平均"], 3))
                                    for z in rows]))
    res["持ち主の例"] = exs
    print(json.dumps(exs, ensure_ascii=False)[:2500], flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
