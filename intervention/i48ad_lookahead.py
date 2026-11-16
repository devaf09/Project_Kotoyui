"""I48ad: 先読みと正誤の判断（要点 → 文）。持ち主の方針: 確率（学習量）ではなく流れに合う共鳴を追い、語板をずらしつつ 10 通りほど先まで読み、
正誤の介入で判断する。

1. 候補: 間ごとに、中心のグレイン（i48ac の数え上げの版）の上位 k。prior の重みは 0・0.5・1、または「帯の確かさで弱める」
   （左右の語の組の学びの件数の段ごとに、prior の重みを取りおきの文で選ぶ）。
2. 先読み: 文の頭から間を順に埋め、幅 BEAM を残す（途中の点は、置いた候補の中心のグレインでの割合の平均）。
3. 文全体の点（確率の積にしない）:
   (a) 文のグレイン: 間の詰め物を中心に、内容語 ±1・選んだ隣の詰め物（左・右）・意味の印を周りに置いた数え上げのグレインで、
       置いた詰め物それぞれの割合（その間の候補の中での事後の割合）の、いちばん弱い所 と 平均。
   (b) 本の数えの自然さ: 詰め物の中の助詞と、その右のいちばん近い述語（VERB・ADJ）の組の PMI の平均（学びの数え）。
   (c) (b) の下位 DROP の割合の候補を落とし、残りを (a) の平均で選ぶ。
手で置く物: BEAM、k、DROP、詰め物の周りの波に詰め物の表記の印を入れること（機能語の板だけでは の・に・で が同じ「格」になるため）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ac_ablate as AB  # noqa: E402
import i48ac_center_grain as CG  # noqa: E402

BEAM = 10
DROP = 0.3
SPECS = {"線形 ±1・そろえない": (dict(W=3, lin_W=1, lin=(), pairs=[], triples=[]), False),
         "全部入り・そろえる": (dict(W=3, lin_W=1, lin=(), pairs=[(-1, 1), (-1, 2), (-1, 3), (-1, "述"), (-1, "印")],
                               triples=[(-1, 1, "述"), (-1, 1, "印")]), True)}
SENT_SPEC = dict(W=1, lin_W=1, lin=("F-", "F+", "印"), pairs=[(-1, 1), ("F-", -1), (1, "F+")], triples=[])
EV_BINS = (0, 1, 3, 10)


def ev_bin(n):
    return max(i for i, b in enumerate(EV_BINS) if n >= b)


class Fill:
    """詰め物の表記 → 周りに置く波（詰め物の中の機能語の板の属性 ＋ 表記の印）。"""

    def __init__(self, S):
        self.S, self.toks, self.c = S, {}, {}

    def see(self, s):
        T = s["toks"]
        ci = [i for i, t in enumerate(T) if G.is_content(t)]
        for k, i in enumerate(ci):
            j = ci[k + 1] if k + 1 < len(ci) else len(T)
            g = "".join(t["form"] for t in T[i + 1:j])
            if g not in self.toks:
                self.toks[g] = T[i + 1:j]

    def wave(self, g):
        if g not in self.c:
            fs, vs = [self.S.attrs("詰:" + g, "詰")[0]], [self.S.attrs("詰:" + g, "詰")[1]]
            for t in self.toks.get(g, []):
                f, v = self.S.tok(t)
                fs.append(f)
                vs.append(v * 0.5)
            self.c[g] = (np.concatenate(fs), np.concatenate(vs))
        return self.c[g]


def sent_ctx(x, fill, left, right):
    ctx = [(o, fv) for o, fv in x["ctx"] if o in (-1, 1, "印")]
    if left is not None:
        ctx.append(("F-", fill.wave(left)))
    if right is not None:
        ctx.append(("F+", fill.wave(right)))
    return ctx


def posterior(pg, f, v, prior_w=1.0):
    s = prior_w * pg.prior + pg.logp[:, f] @ v
    s = s - s.max()
    p = np.exp(s)
    return p / p.sum()


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--k", type=int, nargs="*", default=[3, 5])
    ap.add_argument("--beam", type=int, nargs="*", default=[BEAM])
    ap.add_argument("--specs", nargs="*", default=list(SPECS))
    ap.add_argument("--pmodes", nargs="*", default=["0", "0.5", "1", "確かさ"])
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
    fill = Fill(S)
    for s in tr:
        fill.see(s)
    surf = G.Surface()
    for s in tr:
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    TRs = [(s, AB.items(S, s)) for s in tr]
    TR = [x for _, xs in TRs for x in xs]
    cnt = Counter(x["y"] for x in TR)
    pair_n = Counter((x["l"], x["r"]) for x in TR if not x["last"])
    # (b) の数え: 詰め物の中の助詞と、右のいちばん近い述語
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

    print(f"読み込みと数え {time.time() - t0:.0f} 秒、学び {len(tr)} 文、間 {len(TR)}", flush=True)
    # 文のグレイン（周りに選んだ隣の詰め物を置く）
    t1 = time.time()

    def sent_rows():
        for _, xs in TRs:
            for i, x in enumerate(xs):
                left = xs[i - 1]["y"] if i > 0 else None
                right = xs[i + 1]["y"] if i + 1 < len(xs) else None
                yield (*CG.join(CG.spec_features(S, sent_ctx(x, fill, left, right), SENT_SPEC), False), x["y"])
    sg = G.PairGrain(a.D).fit(sent_rows(), cnt)
    print(f"文のグレイン {time.time() - t1:.0f} 秒", flush=True)
    TE = [(s, AB.items(S, s)) for s in te]
    HO = [(s, AB.items(S, s)) for s in hold]
    res = dict(学びの文=len(tr), 試しの文=len(te), D=a.D, BEAM=BEAM, DROP=DROP, 構成={})
    # 基準: 語の表（隣の語）と I48ab の混ぜに当たる物（語の表に 3 件以上なら表、無ければ 1 か所ずつ）
    nt = CG.NeighborTable()
    for x in TR:
        nt.add(x["l"], x["r"] if not x["last"] else ("文末",) + x["mk"], x["y"])
    for gname in a.specs:
        spec, norm = SPECS[gname]
        t1 = time.time()
        pg = G.PairGrain(a.D).fit(((*CG.join(CG.spec_features(S, x["ctx"], spec), norm), x["y"]) for x in TR), cnt)
        feat = lambda x: CG.join(CG.spec_features(S, x["ctx"], spec), norm)
        # 帯の確かさで prior を弱める: 段ごとの重みを取りおきで選ぶ
        best_w = {}
        for b in range(len(EV_BINS)):
            hx = [x for _, xs in HO for x in xs if ev_bin(pair_n[(x["l"], x["r"])] if not x["last"] else 10) == b][:1500]
            if not hx:
                best_w[b] = 1.0
                continue
            F = [feat(x) for x in hx]
            best_w[b] = max((sum(pg.cls[int(np.argmax(w * pg.prior + pg.logp[:, f] @ v))] == x["y"] for (f, v), x in zip(F, hx)), w)
                            for w in (0.0, 0.25, 0.5, 1.0))[1]
        print(gname, "学び", round(time.time() - t1), "秒、確かさの段ごとの prior", best_w, flush=True)
        for pmode, k, bw in [(p_, k_, b_) for p_ in a.pmodes for k_ in a.k for b_ in a.beam]:
            if True:
                t2 = time.time()
                c = Counter()
                ex = []
                for s, xs in TE:
                    if not xs:
                        continue
                    cont, gaps, head = G.split(s)
                    key = [(l, f_, u) for l, f_, u in cont]
                    cands = []
                    for x in xs:
                        pw = best_w[ev_bin(pair_n[(x["l"], x["r"])] if not x["last"] else 10)] if pmode == "確かさ" else float(pmode)
                        f, v = feat(x)
                        p = posterior(pg, f, v, pw)
                        p[-1] = 0                           # 「その他」は言わない
                        top = np.argsort(-p)[:k]
                        cands.append([(pg.cls[i], float(p[i])) for i in top])
                        c["上位kに正解"] += x["y"] in [y for y, _ in cands[-1]]
                        c["所"] += 1
                    # 先読み（幅 BEAM）
                    beam = [((), 0.0)]
                    for i, cs in enumerate(cands):
                        nb = [(seq + (y,), sc + q) for seq, sc in beam for y, q in cs]
                        nb.sort(key=lambda z: -z[1] / (i + 1))
                        beam = nb[:bw]
                    seqs = [seq for seq, _ in beam]
                    c["残った中に元の詰め物の並び"] += tuple(gaps) in seqs
                    one = tuple(cs[0][0] for cs in cands)   # 1 か所ずつ（先読みなし）
                    scored = []
                    for seq in seqs:
                        sp = []
                        for i, x in enumerate(xs):
                            f, v = CG.join(CG.spec_features(S, sent_ctx(x, fill, seq[i - 1] if i > 0 else None, seq[i + 1] if i + 1 < len(seq) else None), SENT_SPEC), False)
                            post = posterior(sg, f, v)
                            sp.append(float(post[sg.cls.index(seq[i])]) if seq[i] in sg.cls else 0.0)
                        scored.append(dict(seq=seq, 弱い所=min(sp), 平均=float(np.mean(sp)), 自然さ=pmi(key, seq)))
                    thr = sorted(z["自然さ"] for z in scored)[int(len(scored) * DROP)] if len(scored) > 1 else -1e9
                    picks = dict(先読みなし=one, a_弱い所=max(scored, key=lambda z: z["弱い所"])["seq"], a_平均=max(scored, key=lambda z: z["平均"])["seq"],
                                 b_自然さ=max(scored, key=lambda z: z["自然さ"])["seq"],
                                 c_両方=max([z for z in scored if z["自然さ"] >= thr] or scored, key=lambda z: z["平均"])["seq"],
                                 a_両方=max([z for z in scored if z["弱い所"] >= sorted(y["弱い所"] for y in scored)[int(len(scored) * DROP)]] or scored,
                                           key=lambda z: z["平均"])["seq"],
                                 語の表=tuple(nt.predict(x["l"], x["r"] if not x["last"] else ("文末",) + x["mk"]) for x in xs),
                                 混ぜ=tuple(nt.predict(x["l"], x["r"] if not x["last"] else ("文末",) + x["mk"]) if sum(nt.t2.get((x["l"], x["r"] if not x["last"] else ("文末",) + x["mk"]), {}).values()) >= 3
                                           else one[i] for i, x in enumerate(xs)))
                    c["文"] += 1
                    for nm, seq in picks.items():
                        for i, (g, y) in enumerate(zip(seq, gaps)):
                            c[nm + (":間" if i + 1 < len(gaps) else ":文末")] += g == y
                        sent = "".join(surf.form(l, u, g) + g for (l, f_, u), g in zip(key, seq))
                        c[nm + ":文"] += G.norm(head + sent) == G.norm(s["text"])
                    c["間n"] += len(gaps) - 1
                o = dict(秒=round(time.time() - t2), 上位kに正解=round(c["上位kに正解"] / max(1, c["所"]), 3),
                         残った中に元の並び=round(c["残った中に元の詰め物の並び"] / max(1, c["文"]), 3))
                for nm in ("先読みなし", "a_弱い所", "a_平均", "a_両方", "b_自然さ", "c_両方", "語の表", "混ぜ"):
                    o[nm] = dict(間=round(c[nm + ":間"] / max(1, c["間n"]), 3), 文末=round(c[nm + ":文末"] / max(1, c["文"]), 3), 文=round(c[nm + ":文"] / max(1, c["文"]), 3))
                res["構成"][f"{gname}・prior {pmode}・k {k}・幅 {bw}"] = o
                print(gname, pmode, k, bw, json.dumps(o, ensure_ascii=False), flush=True)
                json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        # 持ち主の例（k 5、確かさの prior）
        exs = []
        for keys, mk in (((("昨日", "NOUN"), ("友達", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")), {"過去"}),
                         ((("昨日", "NOUN"), ("公園", "NOUN"), ("犬", "NOUN"), ("散歩", "NOUN"), ("する", "VERB")), {"過去"}),
                         ((("昨日", "NOUN"), ("犬", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")), {"過去"})):
            A = [S.attrs(l, u) for l, u in keys]
            M = AB.marks_attr(S, mk)
            n = len(keys)
            xs = []
            for i in range(n):
                ctx = [(-(i - j + 1), A[j]) for j in range(max(0, i - 2), i + 1)] + [(j - i, A[j]) for j in range(i + 1, min(n, i + 4))]
                ctx += [("述", A[-1]), ("印", M)]
                xs.append(dict(ctx=ctx, l=keys[i][0], r=keys[i + 1][0] if i + 1 < n else None, last=i + 1 == n, mk=tuple(sorted(mk))))
            cands = []
            for x in xs:
                pw = best_w[ev_bin(pair_n[(x["l"], x["r"])] if not x["last"] else 10)]
                p = posterior(pg, *feat(x), pw)
                p[-1] = 0
                cands.append([(pg.cls[i], float(p[i])) for i in np.argsort(-p)[:max(a.k)]])
            beam = [((), 0.0)]
            for i, cs in enumerate(cands):
                beam = sorted([(seq + (y,), sc + q) for seq, sc in beam for y, q in cs], key=lambda z: -z[1] / (i + 1))[:max(a.beam)]
            rows = []
            for seq, _ in beam:
                sp = []
                for i, x in enumerate(xs):
                    f, v = CG.join(CG.spec_features(S, sent_ctx(x, fill, seq[i - 1] if i > 0 else None, seq[i + 1] if i + 1 < len(seq) else None), SENT_SPEC), False)
                    post = posterior(sg, f, v)
                    sp.append(float(post[sg.cls.index(seq[i])]) if seq[i] in sg.cls else 0.0)
                key3 = [(l, l, u) for l, u in keys]
                rows.append(dict(文="".join(surf.form(l, u, g) + g for (l, u), g in zip(keys, seq)), 弱い所=round(min(sp), 3), 平均=round(float(np.mean(sp)), 3),
                                 自然さ=round(pmi(key3, seq), 2)))
            exs.append(dict(要点=[k_ for k_, _ in keys] + sorted(mk), 上位5=[[y for y, _ in cs] for cs in cands], 残った10通り=rows))
        res.setdefault("持ち主の例", {})[gname] = exs
        print(json.dumps(exs, ensure_ascii=False)[:1500], flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
