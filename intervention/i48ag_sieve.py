"""I48ag: 正誤の判断を、本の数えのふるい落としだけに使う（I48ad・I48af の続き）。文のグレインで選び直すと下がったので、選ぶのは候補のグレインの点のまま。

- 候補のグレイン: 節ごとの印の 帯 または シフト＋元（I48ae）。prior は帯の確かさで弱める（I48ad）。
- 先読み: 候補の上位 k を、置いた候補の割合の平均で幅 bw に絞る（I48af の「元の点」）。
- 選び方:
    先読みなし : 間ごとにいちばん強い候補
    幅の最上   : 幅に残った中で、候補の割合の平均がいちばん大きい並び
    ふるい     : 幅に残った中から、本の数えの自然さ（(助詞, 右の述語) の PMI の平均、I48ad の (b)）の下位 DROP を落とし、残りの最上
    ふるい・弱 : 自然さが「学びの文の自然さ」の下位 DROP の線より下の並びだけを落とし（線は学びから決める）、残りの最上
手で置く物: k、幅、DROP（I48ad と同じ 0.3）。
python intervention/i48ag_sieve.py --conllu ... --content ... --func ... --out out.json [--train N] [--test N] [--modes 帯 シフト＋元]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ac_center_grain as CG  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402

DROP = AD.DROP


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--k", type=int, nargs="*", default=[5, 12])
    ap.add_argument("--beam", type=int, nargs="*", default=[10, 20])
    ap.add_argument("--modes", nargs="*", default=["帯", "シフト＋元"])
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold = tr_all[-2000:]
    tr = tr_all[:-2000][: a.train]
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
    TRs = [(s, AE.items(S, s, "節")) for s in tr]
    TR = [x for _, xs in TRs for x in xs]
    cnt = Counter(x["y"] for x in TR)
    pair_n = Counter((x["l"], x["r"]) for x in TR if not x["last"])
    pv, pp, vv = Counter(), Counter(), Counter()
    for s, xs in TRs:
        cont, gaps, _ = G.split(s)
        for i, g in enumerate(gaps):
            v = next((c[0] for c in cont[i + 1:] if c[2] in ("VERB", "ADJ")), None)
            if v is None:
                continue
            for t in fill.toks.get(g, []):
                if t["upos"] == "ADP":
                    pv[(t["lemma"], v)] += 1
                    pp[t["lemma"]] += 1
                    vv[v] += 1
    NP = sum(pp.values())

    def pmi(cont, fills):
        vals = []
        for i, g in enumerate(fills):
            v = next((c[0] for c in cont[i + 1:] if c[2] in ("VERB", "ADJ")), None)
            if v is None:
                continue
            for t in fill.toks.get(g, []):
                if t["upos"] == "ADP":
                    vals.append(math.log((pv[(t["lemma"], v)] + 0.5) * NP / ((pp[t["lemma"]] + 0.5) * (vv[v] + 0.5))))
        return float(np.mean(vals)) if vals else 0.0
    # 学びの文の自然さの分布（取りおきの文で）から、下位 DROP の線を決める
    hv = sorted(pmi(G.split(s)[0], G.split(s)[1]) for s in hold if G.split(s)[0])
    line = hv[int(len(hv) * DROP)] if hv else -1e9
    print(f"読み込み {time.time() - t0:.0f} 秒、学び {len(tr)} 文、自然さの線 {line:.3f}", flush=True)
    nt = CG.NeighborTable()
    for x in TR:
        nt.add(x["l"], x["r"] if not x["last"] else ("文末",) + x["mk"], x["y"])
    TE = [(s, AE.items(S, s, "節")) for s in te]
    HO = [x for s in hold for x in AE.items(S, s, "節")]
    res = dict(学びの文=len(tr), 試しの文=len(te), D=a.D, 自然さの線=line, 構成={})
    for mode in a.modes:
        t1 = time.time()
        feat = lambda x, mode=mode: AE.feats(S, x, mode)
        pg = G.PairGrain(a.D).fit(((*feat(x), x["y"]) for x in TR), cnt)
        best_w = {}
        for b in range(len(AD.EV_BINS)):
            hx = [x for x in HO if AD.ev_bin(pair_n[(x["l"], x["r"])] if not x["last"] else 10) == b][:1500]
            F = [feat(x) for x in hx]
            best_w[b] = max((sum(pg.cls[int(np.argmax(w * pg.prior + pg.logp[:, f] @ v))] == x["y"] for (f, v), x in zip(F, hx)), w)
                            for w in (0.0, 0.25, 0.5, 1.0))[1] if hx else 1.0
        print(mode, "学び", round(time.time() - t1), "秒、prior", best_w, flush=True)
        for k in a.k:
            for bw in a.beam:
                t2 = time.time()
                c = Counter()
                for s, xs in TE:
                    if not xs:
                        continue
                    cont, gaps, head = G.split(s)
                    cands = []
                    for x in xs:
                        pw = best_w[AD.ev_bin(pair_n[(x["l"], x["r"])] if not x["last"] else 10)]
                        p = AD.posterior(pg, *feat(x), pw)
                        p[-1] = 0
                        cands.append([(pg.cls[i], float(p[i])) for i in np.argsort(-p)[:k]])
                    beam = [((), 0.0)]
                    for i, cs in enumerate(cands):
                        beam = sorted([(seq + (y,), sc + q) for seq, sc in beam for y, q in cs], key=lambda z: -z[1])[:bw]
                    sc = [dict(seq=seq, 点=v / len(seq), 自然さ=pmi(cont, seq)) for seq, v in beam]
                    thr = sorted(z["自然さ"] for z in sc)[int(len(sc) * DROP)] if len(sc) > 1 else -1e9
                    picks = dict(先読みなし=tuple(cs[0][0] for cs in cands), 幅の最上=max(sc, key=lambda z: z["点"])["seq"],
                                 ふるい=max([z for z in sc if z["自然さ"] >= thr] or sc, key=lambda z: z["点"])["seq"],
                                 ふるい_弱=max([z for z in sc if z["自然さ"] >= line] or sc, key=lambda z: z["点"])["seq"],
                                 語の表=tuple(nt.predict(x["l"], x["r"] if not x["last"] else ("文末",) + x["mk"]) for x in xs))
                    c["文"] += 1
                    c["間n"] += len(gaps) - 1
                    c["元の並び"] += tuple(gaps) in [z["seq"] for z in sc]
                    for nm, seq in picks.items():
                        for i, (g, y) in enumerate(zip(seq, gaps)):
                            c[nm + (":間" if i + 1 < len(gaps) else ":文末")] += g == y
                        sent = "".join(surf.form(l, u, g) + g for (l, f_, u), g in zip(cont, seq))
                        c[nm + ":文"] += G.norm(head + sent) == G.norm(s["text"])
                o = dict(秒=round(time.time() - t2), 残った中に元の並び=round(c["元の並び"] / max(1, c["文"]), 3))
                for nm in ("先読みなし", "幅の最上", "ふるい", "ふるい_弱", "語の表"):
                    o[nm] = dict(間=round(c[nm + ":間"] / max(1, c["間n"]), 3), 文末=round(c[nm + ":文末"] / max(1, c["文"]), 3), 文=round(c[nm + ":文"] / max(1, c["文"]), 3))
                res["構成"][f"{mode}・k {k}・幅 {bw}"] = o
                print(mode, k, bw, json.dumps(o, ensure_ascii=False), flush=True)
                json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
