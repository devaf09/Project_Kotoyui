"""I48av: 驚きを場所ごとに効かせる（持ち主: 全体に一様だと、傾きが一枚の板のように全体に効く。ポイントを強調する・共振部分を強調する）。

I48at (二) の −log p（間ごとの 1 つの重さを、その間の全部の帯に同じだけ掛ける）を土台に、次を比べる。
  (0) logp: 今の −log p。
  (1) 帯: 今の点の板（PairGrain）に足す時、その間の重さを全部の帯に同じだけ掛けるのではなく、
      「正解の点を押し下げていた帯」に寄せて足す。押し下げの量は、帯ごとの寄与の差
          d_k = v_k (logp[相手, f_k] − logp[正解, f_k])、相手 = 正解を除いていちばん点の高い詰め物
      の正の部分。足す量は 重さ × Σv × max(d_k, 0) / Σ max(d, 0)（和は (0) と同じ）。正の d が無い間は (0) と同じ形。
      ほかの板（R1・R2・R3・prior）は (0) と同じ重さのまま。
  (2) ならし: 束（BLOCK 文）ごとに、詰め物ごとの重さの和を、その束での詰め物の数（同じ重さの学びの時の和）に合わせる。
      数の少ない詰め物（と など）が、多い詰め物の大きな驚きに押されないようにする。
  (3) 帯ならし: (1) ＋ (2)。
I48aw（I48av の続き、寄せ替えが強すぎたので）:
  (4) 足し増し: 重さのうち 1 までの分（min(w, 1)）は (0) と同じく全部の帯に v の形で足し、驚きの分 max(w − 1, 0) だけを
      押し下げていた帯に寄せる（足す量は max(w − 1, 0) × Σv × 寄せ方）。和は (0) と同じ。正の d が無い間は (0) と同じ。
  (5) 全体ならし: 学びを全部足した後に、詰め物ごとの重さの和を、学び全体でのその詰め物の数に合わせる
      （その詰め物の全部の板の行・prior・R3 の重さを、数 ÷ 重さの和 倍する）。束ごとではないので、1 回への集まりが起きない。
  (6) 足し増し全体ならし: (4) ＋ (5)。比べる相手は (0) logp と 同じ重さ。
測る物は I48at (二) と同じ（間・文末・文、助詞ごとの当たりと害、「と」の順位、筋の無い共振）。
手で置く物: なし（BLOCK は I48at と同じ 1,000 文）。
python intervention/past/i48av_local_surprise.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
    [--weights logp 帯 ならし 帯ならし 同じ 足し増し 全体ならし 足し増し全体ならし]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
import i48at_surprise as AW  # noqa: E402

HOWS = ("logp", "帯", "ならし", "帯ならし")
HOWS_AW = ("同じ", "logp", "足し増し", "全体ならし", "足し増し全体ならし")


def band_share(f, v, logp, sc, c):
    """正解 c の点を押し下げていた帯への寄せ方（和 1）。無ければ v の形のまま。"""
    s = sc.copy()
    s[c] = -np.inf
    s[-1] = -np.inf
    k = int(np.argmax(s))
    d = v * (logp[k, f] - logp[c, f])
    pos = np.maximum(d, 0.0)
    tot = float(pos.sum())
    if tot <= 0:
        return v / max(1e-9, float(v.sum())), False
    return pos / tot, True


class VModel(AW.WModel):
    def weight(self, how, p):
        return -np.log(max(p, 1e-12))

    def adjust_block(self, items, how):
        if how not in ("ならし", "帯ならし"):                  # 束ごとのならし（I48av）。全体ならしは finish_weights
            return
        tot, n = defaultdict(float), defaultdict(int)
        for it in items:
            tot[it[2]] += it[3]
            n[it[2]] += 1
        for it in items:
            if tot[it[2]] > 0:
                it[3] = it[3] * n[it[2]] / tot[it[2]]

    def add_gap(self, keys, x, c, w, fic, prev, d, how):
        if ("帯" not in how and "足し増し" not in how) or d is None:
            return AW.WModel.add_gap(self, keys, x, c, w, fic, prev, d, how)
        f, v, logp, sc = d
        share, moved = band_share(f, v, logp, sc, c)
        self.n_moved += moved
        self.n_gap += 1
        if "足し増し" in how:
            np.add.at(self.pg.W[c], f, min(w, 1.0) * v + max(w - 1.0, 0.0) * float(v.sum()) * share)
        else:
            np.add.at(self.pg.W[c], f, w * float(v.sum()) * share)
        self.pr[c] += w
        self.add_boards(keys, x, c, w, fic, prev, base=False)

    def finish_weights(self, ws, cs, how):
        """(5) 全体ならし: 詰め物ごとの重さの和を、学び全体でのその詰め物の数に合わせる。"""
        if "全体ならし" not in how:
            return
        tot, n = defaultdict(float), defaultdict(int)
        for w, c in zip(ws, cs):
            tot[c] += w
            n[c] += 1
        a = {c: n[c] / tot[c] for c in tot if tot[c] > 0}
        for c, k in a.items():
            self.pg.W[c] *= k
            self.pr[c] *= k
            self.who.B[c] *= k
            self.who.n[c] *= k
            for R in (self.rR, self.rL, self.rP):
                R.W[c] *= k
        for i, c in enumerate(cs):
            ws[i] *= a.get(c, 1.0)
        self.scale_max = round(max(a.values()), 3) if a else None
        self.scale_min = round(min(a.values()), 3) if a else None

    def fit_weighted(self, tr, hold, how):
        self.n_moved = self.n_gap = 0
        self.scale_max = self.scale_min = None
        info = AW.WModel.fit_weighted(self, tr, hold, how)
        if self.n_gap:
            info["帯を寄せた間の割合"] = round(self.n_moved / self.n_gap, 4)
        if self.scale_max is not None:
            info["ならしの倍率の最小"], info["ならしの倍率の最大"] = self.scale_min, self.scale_max
        return info


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    ap.add_argument("--weights", nargs="*", default=list(HOWS))
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
    res = dict(学びの文=len(tr), 試しの文=len(te), BLOCK=AW.BLOCK, 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for how in a.weights:
        t1 = time.time()
        m = VModel(S, fill, a.r1)
        info = m.fit_weighted(tr, hold, how)
        r = AO.measure(m, te, surf)
        r["筋の無い共振"] = AW.monitor(m, te)
        r["重さ"] = info
        res["型"][how] = r
        print(how, round(time.time() - t1), "秒", json.dumps(r, ensure_ascii=False), flush=True)
        save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
