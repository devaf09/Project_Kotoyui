"""I48at (三): 二回り目の違和感チェック（持ち主の方針: 判断に介入があると良い。一度回した後、もう一度一つずつ、自覚と介入で違和感をチェック）。

- 一回り学んだ後（学び 5 万、同じ重さ）、学んだ文をもう一度一つずつ照らす（左から順）。各間の違和感を数えで出す:
    (a) 自覚: 選びの 1 位と 2 位の差（小さいほど迷い）、正解に与えた確からしさ（小さいほど違和感）
    (b) 論理の筋（R2 ＋ R3）と全体の波（R1）の食い違い: 選んだ候補の、筋の順位 ＋ 波の順位
    (c) 介入の板の向きとの食い違い・頼った単位の深さは、この試しでは介入の板が無いので数えない。
    (d) 筋の無い共振で選ばれたか（選んだ候補の筋の点がその間の候補の中央値以下で、波の点がいちばん）
    (e) 前後との並びの自然さ: 選んだ詰め物の中の助詞と、右のいちばん近い述語の PMI（学びの文の数え、I48ag と同じ形）。低いほど違和感
  違和感 = 各材料の順位の和（どれも違和感が強いほど上。手の重みなし）。I48au の案（怪しさの順位）と同じ物。
- 直し方:
    (i) 機械だけ: 学んだ間をもう一度、重さ = 1 − 正解の確からしさ で本体に足し直す（違和感の強い間ほど深く）。
    (ii) 人が見る代わり: 違和感の上位 N 件（N = 10・30・100・300）だけ正解を介入の単位（I48ap、型 C）として教える。
- 測る物: 二回り目の時間（1 文あたり・合計）、違和感の上位 N 件のうち外れ（選び ≠ 正解）だった割合（N = 10〜1,000）、直した後の間・文末・文・害・広がり。
  上位の例は Tatoeba の文の要点だけ（数件）。
手で置く物: N の並び。
- 絞り（--known-only）: 正解が候補の組の中にある間（「その他」でない間）だけで違和感の順を作り直し、(ii) をやり直す。
python intervention/past/i48at_recheck.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
    [--known-only] [--ns 10 30 100 300 1000] [--no-machine] [--rows rows.json]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
import i48ap_units as AP  # noqa: E402

NS = (10, 30, 100, 300)


class Natural:
    """(助詞, 右の述語) の PMI（学びの文で数える）。"""

    def __init__(self, fill, tr):
        from collections import Counter
        self.fill, self.pv, self.pp, self.vv = fill, Counter(), Counter(), Counter()
        for s in tr:
            cont, gaps, _ = G.split(s)
            for i, g in enumerate(gaps):
                v = next((c[0] for c in cont[i + 1:] if c[2] in AJ.PRED), None)
                if v is None:
                    continue
                for t in fill.toks.get(g, []):
                    if t["upos"] == "ADP":
                        self.pv[(t["lemma"], v)] += 1
                        self.pp[t["lemma"]] += 1
                        self.vv[v] += 1
        self.N = max(1, sum(self.pp.values()))

    def score(self, g, pred):
        if pred is None:
            return 0.0
        vals = [np.log((self.pv[(t["lemma"], pred)] + 0.5) * self.N / ((self.pp[t["lemma"]] + 0.5) * (self.vv[pred] + 0.5)))
                for t in self.fill.toks.get(g, []) if t["upos"] == "ADP"]
        return float(np.mean(vals)) if vals else 0.0


def recheck(m, tr, nat):
    """学んだ文を一つずつ照らし直し、間ごとの違和感の材料を集める。"""
    rows = []
    for j, s in enumerate(tr):
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        keys = AK.keys_of(s)
        gaps = G.split(s)[1]
        fic = m.who.fiction(keys)
        ups = {u for _, u in keys}
        placed = []
        for x in xs:
            g = m.gap_scores(keys, x, fic)
            r3 = m.gram.score(m.gram.state(ups, placed, x["last"]))
            logic = g["R2"] + r3
            wave = g["R1"]
            three = wave + logic
            both = g["今"] + three
            three[-1] = both[-1] = -1e9
            cand = (set(np.argsort(-g["今"])[:AK.K]) | set(np.argsort(-three)[:AK.K])) - {m.nc - 1}
            pick = max(cand, key=lambda c: both[c])
            top2 = np.sort(both)[-2:]
            p = np.exp(both - both.max())
            p /= p.sum()
            gold = m.cid(x["y"])
            lg, wv = logic.copy(), wave.copy()
            lg[-1] = wv[-1] = -1e9
            pred = AJ.next_pred(keys, x["i"])
            rows.append(dict(j=j, i=int(x["i"]), gold=int(gold), pick=int(pick), 差=float(top2[1] - top2[0]), 確から=float(p[gold]),
                             食い違い=int(AJ.ranks(lg)[pick] + AJ.ranks(wv)[pick]),
                             筋無し=int(lg[pick] <= np.median(lg[:-1]) and int(np.argmax(wv)) == pick),
                             自然さ=nat.score(m.pg.cls[pick], pred[0] if pred else None)))
            placed.append(m.pg.cls[pick])
    return rows


def oddness(rows):
    """違和感の順位（0 がいちばん違和感が強い）: 差が小さい・確からしさが小さい・食い違いが大きい の順位の和。"""
    a = np.array([r["差"] for r in rows])
    b = np.array([r["確から"] for r in rows])
    c = -np.array([r["食い違い"] for r in rows], dtype=float)
    d = -np.array([r["筋無し"] for r in rows], dtype=float)
    e = np.array([r["自然さ"] for r in rows])
    rs = sum(np.argsort(np.argsort(v, kind="stable"), kind="stable") for v in (a, b, c, d, e))
    return np.argsort(rs, kind="stable")


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    ap.add_argument("--known-only", action="store_true")
    ap.add_argument("--ns", nargs="*", type=int, default=list(NS))
    ap.add_argument("--no-machine", action="store_true")
    ap.add_argument("--rows", default=None, help="照らし直しの記録（あれば読み、無ければ書く）")
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
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
    m = AP.UModel(S, fill, a.r1).fit(tr, hold).init_units("C")
    m.refresh()
    res = dict(学びの文=len(tr), 試しの文=len(te), 土台=AO.measure(m, te, surf))

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"学び {time.time() - t0:.0f} 秒", json.dumps(res["土台"], ensure_ascii=False), flush=True)
    save()
    t1 = time.time()
    if a.rows and os.path.exists(a.rows):
        rows = json.load(open(a.rows, encoding="utf-8"))
    else:
        rows = recheck(m, tr, Natural(fill, tr))
        if a.rows:
            json.dump(rows, open(a.rows, "w", encoding="utf-8"))
    dt = time.time() - t1
    n_all = len(rows)
    if a.known_only:
        rows = [r for r in rows if r["gold"] != m.nc - 1]
    res["絞り"] = dict(正解が候補にある間だけ=bool(a.known_only), 全部の間=n_all, 使った間=len(rows))
    order = oddness(rows)
    wrong = np.array([r["pick"] != r["gold"] for r in rows])
    res["二回り目"] = dict(秒=round(dt), 一文の秒=round(dt / max(1, len(tr)), 4), 間=len(rows), 外れの割合_全体=round(float(wrong.mean()), 4),
                        上位の外れの割合={str(n): round(float(wrong[order[:n]].mean()), 3) for n in (10, 30, 100, 300, 1000, 3000)},
                        外れのうち上位1割に入る割合=round(float(wrong[order[:len(order) // 10]].sum() / max(1, wrong.sum())), 4),
                        上位1割の外れ=round(float(wrong[order[:len(order) // 10]].mean()), 4), 上位2割の外れ=round(float(wrong[order[:len(order) // 5]].mean()), 4),
                        外れのうち上位2割に入る割合=round(float(wrong[order[:len(order) // 5]].sum() / max(1, wrong.sum())), 4),
                        筋の無い共振=int(sum(r["筋無し"] for r in rows)),
                        上位の例=[dict(要点="・".join(k[0] for k in AK.keys_of(tr[rows[o]["j"]])), 間=rows[o]["i"], 正解=m.pg.cls[rows[o]["gold"]],
                                     選び=m.pg.cls[rows[o]["pick"]]) for o in order[:5]])
    print("二回り目", json.dumps(res["二回り目"], ensure_ascii=False), flush=True)
    save()
    # (ii) 上位 N 件だけ正解を介入の単位として教える（型 C）
    res["人が見る代わり"] = {}
    for n in a.ns:
        t1 = time.time()
        mm = m.clone()
        mm.kind = "C"
        for o in order[:n]:
            r = rows[o]
            s = tr[r["j"]]
            keys = AK.keys_of(s)
            x = AE.items(S, s, "節")[r["i"]]
            mm.teach_unit(("違和感", f"上位 {n}", keys, G.split(s)[1]), xs=[x])
        mm.refresh()
        res["人が見る代わり"][str(n)] = rr = AO.measure(mm, te, surf)
        print("上位", n, round(time.time() - t1), "秒", json.dumps(rr, ensure_ascii=False), flush=True)
        save()
    if a.no_machine:
        res["秒"] = round(time.time() - t0)
        save()
        return
    # (i) 機械だけ: 重さ 1 − 確からしさ で本体に足し直す
    t1 = time.time()
    mm = m.clone()
    mm.units = []
    mm.init_units("C")
    for r in rows:
        s = tr[r["j"]]
        keys = AK.keys_of(s)
        gaps = G.split(s)[1]
        x = AE.items(S, s, "節")[r["i"]]
        mm.add_boards(keys, x, r["gold"], 1.0 - r["確から"], mm.who.fiction(keys), gaps[:r["i"]], base=True)
    tot = mm.pg.W.sum(1, keepdims=True)
    mm.pg.logp = np.log((mm.pg.W + G.ALPHA) / (tot + G.ALPHA * S.D))
    mm.pg.logp[-1] = -1e9
    mm.refresh()
    res["機械だけ"] = AO.measure(mm, te, surf)
    print("機械だけ", round(time.time() - t1), "秒", json.dumps(res["機械だけ"], ensure_ascii=False), flush=True)
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
