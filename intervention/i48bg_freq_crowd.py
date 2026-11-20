"""I48bg: 周波数で位置を持つ語板（I48bf）の、混む箱の手当て（案、未確定）。5 万文・1,000 文、W 1。

I48bf では、よく出る語の箱が混みすぎて（「相手 × 距離」の種類が値の数 800 を越える箱 1.2%、最大 134 倍）、遠くからの読みが雑音になった。
比べる形（どれも本体に差し込み、読みの範囲以外は I48bf と同じ）:
  (i)   隣だけ（I48bf と同じ。比べの土台）
  (ii)  隣と 2 つ隣まで読む（間の左右の語に加えて、その外の内容語 1 つずつ。距離 ±1・±3）
  (iii) 覚える組を距離 4 以内に限り、隣と 2 つ隣まで読む（遠い組を覚えないので箱が混みにくい）
  (iv)  混む箱（種類 ÷ 値の数 が 1 を越える箱）だけ、値の数を 3 倍（帯ごとに 300 個の周波数）にした別の箱で持ち、隣だけ読む
測る物: 間・文末・文・と の当たり（AP.measure）、箱の混み具合、秒。
手で置く物: 覚える組の距離 4、2 つ隣（距離 ±3）、混む箱の境 1、値の数の倍率 3、ほかは I48bf と同じ。
python intervention/i48bg_freq_crowd.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48ax_sayable as AX  # noqa: E402
import i48bf_freq_position as BF  # noqa: E402
from i48core import learn as LN  # noqa: E402
from i48core import model as MD  # noqa: E402

W = 1.0
WIN = 4
BIG = 3


class FreqCrowd(BF.FreqPos):
    def __init__(self, m, S, W, win=None, big=False):
        super().__init__(m, S, W)
        self.win, self.big = win, big

    def fit(self, m, tr):
        if self.win is None:
            super().fit(m, tr)
        else:
            E0 = self.E
            win = self.win
            self.E = lambda d: E0(d) * (abs(d) <= win)          # 学びの時だけ、遠い組を覚えない
            super().fit(m, tr)
            self.E = E0
            self.stats["覚える組の距離"] = win
        if self.big:
            self._fit_big(m, tr)
        return self

    def _fit_big(self, m, tr):
        t0 = time.time()
        U, F = self.U, self.V.shape[1]
        # 混み具合を数え直す（箱ごとの相手 × 距離の種類）
        span = 2 * self.Lmax + 1
        seen = [set() for _ in range(U)]
        for s in tr:
            u = self.line(m, s)
            for p, b in enumerate(u):
                for q, c in enumerate(u):
                    if q != p:
                        seen[b].add(c * span + (q - p))
        load = np.array([len(x) for x in seen]) / F
        self.crowd = np.where(load > 1)[0]
        ci = {int(b): i for i, b in enumerate(self.crowd)}
        nc2 = BF.NC * BIG
        F2 = len(BF.BANDS) * nc2
        self.nu2 = np.tile(np.arange(nc2) * self.W / nc2, len(BF.BANDS))

        def shape2(key):
            r = np.random.default_rng(zlib.crc32(("大|" + key).encode()))
            return np.exp(2j * np.pi * r.random(nc2)).astype(np.complex64) / np.sqrt(nc2)
        # 大きい値は、元の値と同じ作り（帯ごとの属性の乱数の位相・名前）を 300 個の周波数で作り直す
        V2 = np.zeros((U, F2), dtype=np.complex64)
        for i in range(U):
            if i < self.K:
                l, up = self.keys[i]
                names, nm, eps = BF.content_names(self.S, l, up), l + "|" + up, BF.EPS
            else:
                y = m.pg.cls[i - self.K]
                names, nm, eps = [], "詰め物|" + ("その他" if y is None else (y or "空")), BF.EPS_F
            v = np.zeros(F2, dtype=np.complex64)
            for bi, b in enumerate(BF.BANDS[:-1]):
                x = np.zeros(nc2, dtype=np.complex64)
                for bb, a, wt in names:
                    if bb == b:
                        x += wt * shape2(b + "|" + a)
                n = np.linalg.norm(x)
                if n > 0:
                    v[bi * nc2:(bi + 1) * nc2] = x / n
            n = np.linalg.norm(v)
            v = (1 - eps) * (v / n if n > 0 else v)
            v[(len(BF.BANDS) - 1) * nc2:] = eps * shape2("名前|" + nm)
            V2[i] = v / np.linalg.norm(v)
        R2 = np.zeros((len(self.crowd), F2), dtype=np.complex64)
        for s in tr:
            u = self.line(m, s)
            for p, b in enumerate(u):
                if b in ci:
                    for q, c in enumerate(u):
                        if q != p:
                            R2[ci[b]] += V2[c] * self.E2(q - p)
        self.V2, self.R2, self.ci2 = V2, R2, ci
        r = np.random.default_rng(7920)
        pr = np.exp(2j * np.pi * r.random((BF.P_PROBE, F2))).astype(np.complex64)
        self.probe2 = pr / np.linalg.norm(pr, axis=1, keepdims=True)
        self.stats["混む箱"] = dict(数=len(self.crowd), 内容語=int((self.crowd < self.K).sum()), 詰め物=int((self.crowd >= self.K).sum()),
                                 大きい箱_MB=round(R2.nbytes / 1e6, 1), 大きい値_MB=round(V2.nbytes / 1e6, 1), 秒=round(time.time() - t0))

    def E2(self, d):
        return np.exp(-2j * np.pi * self.nu2 * d).astype(np.complex64)

    def read(self, b, d, cand):
        if not self.big or b not in self.ci2:
            return super().read(b, d, cand)
        Q = np.conj(self.R2[self.ci2[b]]) * self.E2(d)
        nsub = len(BF.BANDS) * BF.NSUB
        Z = (self.V2[cand] * Q[None, :]).reshape(len(cand), nsub, -1).sum(2)
        tot = Z.sum(1)
        nhat = np.maximum(0.0, np.real(tot))
        fm = np.abs(tot) / np.maximum(1e-9, np.abs(Z).sum(1))
        Zp = (self.probe2 * Q[None, :]).reshape(BF.P_PROBE, nsub, -1).sum(2)
        fp = np.abs(Zp.sum(1)) / np.maximum(1e-9, np.abs(Zp).sum(1))
        return nhat, fm, float(fp.mean() + fp.std() * np.sqrt(2 * np.log(len(cand))))

    def gap_pts(self, keys, i):
        if self.ctx != "2つ隣":
            return super().gap_pts(keys, i)
        n = len(keys)
        pg = 2 * i + 2
        out = np.zeros(self.nc, dtype=np.float32)
        for k in (i - 1, i, i + 1, i + 2):
            if 0 <= k < n:
                v = self.pts(self.kid.get(tuple(keys[k]), -1), pg - (2 * k + 1))
                if v is not None:
                    out += v
        return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    ref = tr_all[:-2000][a.train:]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill, surf = AD.Fill(S), G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    base = MD.Model(S, fill, a.r1).fit(tr, hold)
    print(f"学び {time.time() - t0:.0f} 秒", flush=True)
    for s in te:
        AE.items(S, s, "節")
    import i48core.measure as ME
    for _, _, keys, _ in ME.WANT:
        MD.made_items(S, keys, None)
    fill_ref = AD.Fill(S)
    for s in ref:
        fill_ref.see(s)
    say, nat = AX.Sayable(S, ref), RC.Natural(fill_ref, ref)
    res = dict(学びの文=len(tr), 試しの文=len(te), W=W, 覚える組の距離=WIN, 値の数の倍率=BIG, 箱={}, 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def report(nm, m):
        t1 = time.time()
        r = AP.measure(m, te, surf)
        r["言えるか"] = AX.score(m, te, say, nat)
        r["筋の無い共振"] = LN.monitor(m, te)
        r["秒"] = round(time.time() - t1)
        res["型"][nm] = r
        print(nm, r["秒"], "秒", json.dumps({k: r[k] for k in ("間", "文末", "文", "と_人", "と_人以外")}, ensure_ascii=False), flush=True)
        save()
    report("本体", base)
    for nm, win, big, ctx in (("(i) 隣だけ", None, False, "隣"), ("(ii) 隣と 2 つ隣", None, False, "2つ隣"),
                              ("(iii) 覚える組を距離 4 以内・隣と 2 つ隣", WIN, False, "2つ隣"), ("(iv) 混む箱だけ値の数 3 倍・隣だけ", None, True, "隣")):
        if nm.startswith("(ii)"):
            fp.ctx = ctx
        else:
            fp = FreqCrowd(base, S, W, win=win, big=big).fit(base, tr)
            fp.ctx = ctx
            res["箱"][nm] = fp.stats
            print(nm, json.dumps(fp.stats, ensure_ascii=False), flush=True)
        fp.cache = {}
        m = base.clone()
        m.extras.append(fp)
        m.refresh()
        report(nm, m)
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
