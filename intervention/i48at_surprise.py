"""I48at (二): 本体の学びの重さ = 驚き。学びの文を足す前に、今の板で各間の答えを出し、当たった間は浅く、外した間は深く足す。

- 学びの文を BLOCK 文ずつ足す。束を足す前に、それまでに足した板で、束の各間の正解の確からしさ p（今の点の分布、prior の強さ 1）を出す。
  最初の束は、まだ板が無いので重さ 1。
- 重さの型: 同じ重さ（今まで）、1 − p、−log p（手で置くしきい値なし）。重さは、その間を全部の板（今の点のグレイン・prior・R1・R2・R3）に足す時の数え。
- 学びの文の順に依るので、順を 2 通り（元の並び、乱数 seed 1 で並べ替え）。
- 測る物: 間・文末・文、助詞ごとの当たりと害（I48ao の数え）、筋の無い共振（介入の単位なし: 論理の筋 = R2 ＋ R3、全体の波 = R1）。
  重さの分布（平均・中央値・0.1 未満の割合）も。
手で置く物: BLOCK（1,000 文。板の表を作り直す間隔、速さのためだけ）。
python intervention/i48at_surprise.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000] [--weights 同じ 1-p -logp] [--orders 0 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402

BLOCK = 1000


class WModel(AK.Model):
    def weight(self, how, p):
        """正解の確からしさ p から、その間の重さ。"""
        return 1.0 - p if how == "1-p" else -np.log(max(p, 1e-12))

    def adjust_block(self, items, how):
        """束の中の重さを足す前に直す所（ここでは何もしない。I48av で使う）。"""

    def finish_weights(self, ws, cs, how):
        """全部を足し終えた後に、板と重さ（ws、R3 の作り直しに使う）を直す所（ここでは何もしない。I48aw で使う）。"""

    def add_gap(self, keys, x, c, w, fic, prev, d, how):
        """1 つの間を重さ w で全部の板に足す。d = (f, v, logp, sc) は足す前の点（I48av で使う）。"""
        self.add_boards(keys, x, c, w, fic, prev, base=True)

    def fit_weighted(self, tr, hold, how):
        """学びの文を BLOCK 文ずつ、驚きの重さで足す。AK.Model.fit と同じ板を作る。"""
        S = self.S
        TRs = [(s, AE.items(S, s, "節")) for s in tr]
        TR = [x for _, xs in TRs for x in xs]
        cnt = Counter(x["y"] for x in TR)
        self.pair_n = Counter((x["l"], x["r"]) for x in TR if not x["last"])
        self.pg = G.PairGrain(S.D)
        self.pg.cls = [y for y, _ in cnt.most_common(self.pg.n_cls)] + [None]
        nc = self.nc = len(self.pg.cls)
        self.pg.W = np.zeros((nc, S.D), dtype=np.float64)
        self.ci = {y: i for i, y in enumerate(self.pg.cls) if y is not None}
        self.pr = np.zeros(nc)
        self.Wd = np.zeros_like(self.pg.W)
        self.sep, self.Wi, self.ni, self.Li = False, np.zeros_like(self.pg.W, dtype=np.float32), np.zeros(nc), None
        self.log, self.cur_text = [], ""
        self.who = AJ.Whole(S, nc, self.r1)
        self.rR, self.rL, self.rP = AJ.Ratio(nc, S.D), AJ.Ratio(nc, S.D), AJ.Ratio(nc, S.D)
        self.gram = AJ.Gram(nc, self.fill)
        TRK = [AK.keys_of(s) for s, _ in TRs]
        self.who.index(TRK)
        ws, cs = [], []
        for b in range(0, len(TRs), BLOCK):
            if b > 0 and how != "同じ":                             # それまでの板で、今の点の分布を出す
                tot = self.pg.W.sum(1, keepdims=True)
                logp = np.log((self.pg.W + G.ALPHA) / (tot + G.ALPHA * S.D))
                logp[-1] = -1e9
                prior = np.log((self.pr + 1) / (self.pr.sum() + len(self.pr)))
            items = []
            for j in range(b, min(b + BLOCK, len(TRs))):
                s, xs = TRs[j]
                keys = TRK[j]
                if not keys:
                    continue
                fic = self.who.fiction(keys, j)
                gaps = G.split(s)[1]
                for x in xs:
                    c = self.cid(x["y"])
                    if b == 0 or how == "同じ":
                        w, d = 1.0, None
                    else:
                        f, v = self.feat(x)
                        sc = prior + logp[:, f] @ v
                        sc = sc - sc.max()
                        p = np.exp(sc)
                        p = float(p[c] / p.sum())
                        w, d = self.weight(how, p), (f, v, logp, sc)
                    items.append([keys, x, c, w, fic, gaps[:x["i"]], d])
            self.adjust_block(items, how)
            for keys, x, c, w, fic, prev, d in items:
                ws.append(w)
                cs.append(c)
                self.add_gap(keys, x, c, w, fic, prev, d, how)
        self.finish_weights(ws, cs, how)
        # R3 の行は Gram.add（重さ 1）で入るので、重さつきで作り直す
        self.gram = AJ.Gram(nc, self.fill)
        k = 0
        for j, (s, xs) in enumerate(TRs):
            keys = TRK[j]
            if not keys:
                continue
            gaps = G.split(s)[1]
            ups = {u for _, u in keys}
            for x in xs:
                names = self.gram.state(ups, gaps[:x["i"]], x["last"])
                self.gram.rows.append((self.cid(x["y"]), self.gram.f(names, grow=True), ws[k]))
                k += 1
        self.gram.R = AJ.Ratio(nc, max(1, len(self.gram.idx)))
        for c, f, w in self.gram.rows:
            if len(f):
                np.add.at(self.gram.R.W[c], f, np.ones(len(f)) / len(f) * w)
        self.gram.rows = []
        self.gram.R.build()
        tot = self.pg.W.sum(1, keepdims=True)
        self.pg.logp = np.log((self.pg.W + G.ALPHA) / (tot + G.ALPHA * S.D))
        self.pg.logp[-1] = -1e9
        self.pg.prior = np.log((self.pr + 1) / (self.pr.sum() + len(self.pr)))
        HO = [x for s in hold for x in AE.items(S, s, "節")]
        self.best_w = {}
        for bn in range(len(AD.EV_BINS)):
            hx = [x for x in HO if AD.ev_bin(self.pair_n[(x["l"], x["r"])] if not x["last"] else 10) == bn][:1500]
            F = [self.feat(x) for x in hx]
            self.best_w[bn] = max((sum(self.pg.cls[int(np.argmax(w * self.pg.prior + self.pg.logp[:, f] @ v))] == x["y"] for (f, v), x in zip(F, hx)), w)
                                  for w in (0.0, 0.25, 0.5, 1.0))[1] if hx else 1.0
        self.beta = self.choose_beta(hold[:400])
        self.refresh()
        ws = np.array(ws)
        return dict(平均=round(float(ws.mean()), 4), 中央値=round(float(np.median(ws)), 4), 一割未満=round(float((ws < 0.1).mean()), 4), 最大=round(float(ws.max()), 3))


def monitor(m, te):
    """筋の無い共振（介入なし）: 選んだ候補の 論理の筋（R2 ＋ R3）がその間の候補の中央値以下で、全体の波（R1）がいちばん高い物。"""
    n_pick = hit = miss = 0
    pairs = Counter()
    for s in te:
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        keys = AK.keys_of(s)
        gaps = G.split(s)[1]
        rs, fic = m.walk(keys, xs)
        ups = {u for _, u in keys}
        placed = []
        for x, r in zip(xs, rs):
            g = m.gap_scores(keys, x, fic)
            logic = g["R2"] + m.gram.score(m.gram.state(ups, placed, x["last"]))
            wave = g["R1"].copy()
            logic[-1] = wave[-1] = -1e9
            c = r["選び"]
            n_pick += 1
            if logic[c] <= np.median(logic[:-1]) and int(np.argmax(wave)) == c:
                ok = m.pg.cls[c] == gaps[x["i"]]
                hit += ok
                miss += not ok
                pairs[(x["l"], m.pg.cls[c])] += 1
            placed.append(m.pg.cls[c])
    rep = [(k, v) for k, v in pairs.most_common() if v >= 2]
    return dict(選んだ数=n_pick, 筋の無い共振=hit + miss, 当たり=hit, 外れ=miss, くり返す組の数=len(rep),
                くり返す組の例=[dict(左の語=k[0], 詰め物=k[1], 回数=v) for k, v in rep[:5]])


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
