"""I48at (二): 本体の学びの重さ = 驚き。学びの文を足す前に、今の板で各間の答えを出し、当たった間は浅く、外した間は深く足す。

- 学びの文を BLOCK 文ずつ足す。束を足す前に、それまでに足した板で、束の各間の正解の確からしさ p（今の点の分布、prior の強さ 1）を出す。
  最初の束は、まだ板が無いので重さ 1。
- 重さの型: 同じ重さ（今まで）、1 − p、−log p（手で置くしきい値なし）。重さは、その間を全部の板（今の点のグレイン・prior・R1・R2・R3）に足す時の数え。
- 学びの文の順に依るので、順を 2 通り（元の並び、乱数 seed 1 で並べ替え）。
- 測る物: 間・文末・文、助詞ごとの当たりと害（I48ao の数え）、筋の無い共振（介入の単位なし: 論理の筋 = R2 ＋ R3、全体の波 = R1）。
  重さの分布（平均・中央値・0.1 未満の割合）も。
手で置く物: BLOCK（1,000 文。板の表を作り直す間隔、速さのためだけ）。
python intervention/past/i48at_surprise.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000] [--weights 同じ 1-p -logp] [--orders 0 1]
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
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
from i48core.learn import BLOCK, WModel, monitor  # noqa: E402,F401  （実装は i48core.learn へ移した）


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    ap.add_argument("--weights", nargs="*", default=["同じ", "1-p", "-logp"], help="-logp は引数では logp と書く")
    ap.add_argument("--orders", nargs="*", type=int, default=[0, 1])
    a = ap.parse_args()
    a.weights = ["-logp" if w == "logp" else w for w in a.weights]
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr0 = tr_all[-2000:], tr_all[:-2000][: a.train]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr0:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    res = dict(学びの文=len(tr0), 試しの文=len(te), BLOCK=BLOCK, 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for order in a.orders:
        tr = tr0 if order == 0 else [tr0[k] for k in np.random.default_rng(order).permutation(len(tr0))]
        for how in a.weights:
            if how == "同じ" and order != a.orders[0]:
                continue                                         # 同じ重さは順に依らない
            t1 = time.time()
            m = WModel(S, fill, a.r1)
            info = m.fit_weighted(tr, hold, how)
            r = AO.measure(m, te, surf)
            r["筋の無い共振"] = monitor(m, te)
            r["重さ"] = info
            res["型"][f"順 {order}・{how}"] = r
            print(order, how, round(time.time() - t1), "秒", json.dumps(r, ensure_ascii=False), flush=True)
            save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
