"""I48ak: 介入で選んで覚える（持ち主の方針: ある所まで出力を上げたら、後半は介入で学ぶ）。I48aj の続き。

- 候補: 今の点（I48ag の先読みなし、節・シフト＋元）の上位 K と、三つの共振（I48aj の R1＋R2＋R3、prior なし）の上位 K の和集合。
  選ぶ物: 候補の中で 今×三つ がいちばん大きい物（頭から順に、R3 は置いた物で）。
- 介入: 練習の文（学びに使っていない文）の間ごとに、先生役（試しの場では元の文）が候補の中の元の詰め物を指す。
    指された詰め物: その間の特徴を、全部の板（今の点のグレイン・R1・R2 の 3 つ・R3）に足す（明るい側）。
    選んでいた物が違った時: 選んでいた物の今の点の板の暗い側に DARK_W で足す（I48ai と同じ）。
    元が候補に無い時: 何もしない版（候補の中だけ）と、候補の外でも教える版（候補の外も）の 2 通り。
  足す重さ: 1 回分（数え 1）、または「その板の平均の 1 例分」（板の総量 ÷ 回数 を、この例の量で割った重さ）。
- 比べ: 最初の学び（5,000・2 万・5 万文）ごとに、介入 5,000 文の前後と、同じ 5,000 文をただ学びに足した場合（普通の数え）。
- 教えた数: 自分で作った「〜と〜に行った」の文を 1・3・10 文だけ介入で教えた時の、持ち主の例の「と」の順位。
- 別の板（sep、I48al）: 教えた分を本体に混ぜず介入だけの板に持ち、今の点に sep_ratio の点を足す（教えた帯だけが上がる）。
- 練習の間、板の対数の表は REFRESH 文ごとに作り直す（速さのため。教えた文の効きが少し遅れる）。
手で置く物: K（5）、DARK_W（1/1,000）、REFRESH（100 文）、自作の教える文（TEACH）。ほかは I48aj と同じ。
python intervention/past/i48ak_teach.py --conllu ... --content ... --func ... --out out.json [--inits 5000 20000 50000] [--practice 5000] [--test 1000] [--r1 型]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import copy
import json
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
from i48core.model import DARK_W, EXAMPLES, K, keys_of, made_items, made_sentence, MODE, Model, REFRESH, sep_ratio, TEACH  # noqa: E402,F401  （実装は i48core.model へ移した）


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--inits", type=int, nargs="*", default=[5000, 20000, 50000])
    ap.add_argument("--practice", type=int, default=5000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="基準", choices=AJ.R1_VARS)
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold = tr_all[-2000:]
    tr_all = tr_all[:-2000]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    top = max(a.inits)
    pr = tr_all[top: top + a.practice]                     # 練習の文: どの最初の学びにも入っていない文
    for s in tr_all[:top] + pr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    res = dict(R1の型=a.r1, 練習の文=len(pr), 試しの文=len(te), K=K, DARK_W=DARK_W, REFRESH=REFRESH, 学び={})
    print(f"読み込み {time.time() - t0:.0f} 秒", flush=True)

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for n0 in a.inits:
        t1 = time.time()
        base = Model(S, fill, a.r1).fit(tr_all[:n0], hold)
        o = dict(β=base.beta, prior=base.best_w, 前=base.evaluate(te, surf))
        res["学び"][str(n0)] = o
        print(n0, "前", round(time.time() - t1), "秒", json.dumps(o["前"], ensure_ascii=False), flush=True)
        save()
        # 同じ練習の文を、ただ学びに足す（普通の数え）
        t2 = time.time()
        plain = Model(S, fill, a.r1).fit(tr_all[:n0] + pr, hold)
        o["ただ足す"] = plain.evaluate(te, surf)
        print(n0, "ただ足す", round(time.time() - t2), "秒", json.dumps(o["ただ足す"], ensure_ascii=False), flush=True)
        save()
        del plain
        for outside in (False, True):
            for w in (1.0, "平均"):
                t2 = time.time()
                m = base.clone()
                n = Counter()
                for j, s in enumerate(pr):
                    xs = AE.items(S, s, "節")
                    if xs:
                        n.update(m.teach(keys_of(s), xs, G.split(s)[1], outside, w))
                    if (j + 1) % REFRESH == 0:
                        m.refresh()
                m.refresh()
                nm = f"介入・{'候補の外も' if outside else '候補の中だけ'}・{'1 回分' if w == 1.0 else '平均の 1 例分'}"
                o[nm] = dict(数=dict(n), 後=m.evaluate(te, surf))
                print(n0, nm, round(time.time() - t2), "秒", json.dumps(o[nm], ensure_ascii=False), flush=True)
                save()
                del m
        if n0 == top:                                       # 自作の文を 1・3・10 文だけ教える
            o["自作を教える"] = {}
            for outside in (False, True):
                for k in (1, 3, 10):
                    m = base.clone()
                    n = Counter()
                    for a_, b_ in TEACH[:k]:
                        keys, gaps = made_sentence(a_, b_)
                        n.update(m.teach(keys, made_items(S, keys, gaps), gaps, outside, 1.0))
                        m.refresh()
                    nm = f"{'候補の外も' if outside else '候補の中だけ'}・{k} 文"
                    o["自作を教える"][nm] = dict(数=dict(n), 例=m.examples())
                    print(n0, "自作", nm, json.dumps(o["自作を教える"][nm], ensure_ascii=False), flush=True)
                    del m
            save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
