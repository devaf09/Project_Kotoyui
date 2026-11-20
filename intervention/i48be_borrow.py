"""I48be: 数えの少ない語が、使われ方の近い語から借りる（新しい語板 1 版目の続き。案、未確定）。

数えの少ない語（後ろ・前の数え N が床 2 ln V に届かない、1 回以上は出た語）の行を、使われ方の近い語の行で補う。
  近さ: 下調べ G の (b) と同じ。語ごとの使われ方の数え（前・後ろの詰め物、右のいちばん近い述語、位置）を長さ 1 にした物のコサイン。
    貸す語は、床を越えた語（その種類の N が床以上）。上位 TOPN 語。
  借りる量: m = 床 − N（数えが増えるほど自分の行に寄り、床を越えたら借りない）。
  (a) 表: 行 = 自分の数え ＋ m × 近い語の行の平均（各行を割合にしてから平均）。N は N ＋ m として点を出す（点の式・ならしは I48bc と同じ）。
  (b) 波: 語の波 = 自分の波 ＋ m × 近い語の波の平均（各波を N で割ってから平均）。読み方（AM・FM→AM）は I48bc と同じ。
  (c) 借りない（I48bd の表・波と同じ）。
比べる形: 本体、表（借りない・借りる）、波 AM（借りない・借りる）、波 FM→AM（借りない・借りる）。波は DK 2,048 だけ。
測り: いつもの物 ＋ 左の語の数えの帯ごとの当たり（本体も）。帯: 0（学びに出ない）・床未満（1〜床）・〜50・〜500・500 超。
  床未満の帯で、借りる形が借りない形から選びを替えた数と、その当たり。
手で置く物: TOPN（20）、DK（2,048）、B（16）、P（32）、種、帯の境（50・500）。
python intervention/i48be_borrow.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
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
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48ax_sayable as AX  # noqa: E402
import i48bc_usage_board as BC  # noqa: E402
import i48bd_table_vs_wave as BD  # noqa: E402
from i48core import learn as LN  # noqa: E402
from i48core import model as MD  # noqa: E402

TOPN = 20
DK = 2048
BANDS = ("0", "床未満", "〜50", "〜500", "500超")


def band_of(n, floor):
    if n == 0:
        return "0"
    if n < floor:
        return "床未満"
    return "〜50" if n <= BD.N_EDGES[0] else ("〜500" if n <= BD.N_EDGES[1] else "500超")


def usage_vectors(ub):
    """語ごとの使われ方の数え（前・後ろ・述語・位置）を、長さ 1 にした疎な行（特徴の番号 → 値）。"""
    feats, rows = {}, {}
    keys = set(ub.cnt_all["後ろ"]) | set(ub.cnt_all["前"])
    for key in keys:
        r = Counter()
        for k, tag in (("後ろ", "後"), ("前", "前")):
            for c, n in ub.cnt_all[k].get(key, {}).items():
                r[(tag, c)] += n
        for k in ("述語", "位置"):
            for c, n in ub.other_kinds[k].get(key, {}).items():
                r[(k, c)] += n
        nr = float(np.sqrt(sum(v * v for v in r.values())))
        if nr > 0:
            rows[key] = {feats.setdefault(f, len(feats)): v / nr for f, v in r.items()}
    return rows, len(feats)


def neighbours(ub, rows, nf):
    """種類ごとに、床未満の語 → 使われ方の近い貸す語（上位 TOPN）。"""
    out = {}
    for k in ub.KINDS:
        donors = [key for key, c in ub.cnt_all[k].items() if sum(c.values()) >= ub.floor_n and key in rows]
        low = [key for key, c in ub.cnt_all[k].items() if 0 < sum(c.values()) < ub.floor_n and key in rows]
        Dm = np.zeros((len(donors), nf), dtype=np.float32)
        for i, key in enumerate(donors):
            for f, v in rows[key].items():
                Dm[i, f] = v
        nb = {}
        for j0 in range(0, len(low), 500):
            part = low[j0:j0 + 500]
            Q = np.zeros((len(part), nf), dtype=np.float32)
            for i, key in enumerate(part):
                for f, v in rows[key].items():
                    Q[i, f] = v
            S = Q @ Dm.T
            top = np.argpartition(-S, min(TOPN, len(donors) - 1), axis=1)[:, :TOPN]
            for i, key in enumerate(part):
                nb[key] = [donors[t] for t in top[i]]
        out[k] = nb
    return out


class BorrowTable(BD.CountTable):
    """差し込み: 数えの表 ＋ 床未満の語は近い語の行を m = 床 − N 回分借りる。"""

    def fit(self, m, tr, ub, nbrs):
        super().fit(m, tr, ub)
        self.n_borrowed = {}
        for k in self.KINDS:
            prof = {key: v / max(1.0, float(v.sum())) for key, v in self.T[k].items()}
            for key, donors in nbrs[k].items():
                c = ub.cnt_all[k][key]
                N = sum(c.values())
                mm = ub.floor_n - N
                v = np.zeros(self.nc, dtype=np.float32)
                v[np.fromiter(c.keys(), dtype=np.int64)] = np.fromiter(c.values(), dtype=np.float32)
                v += mm * np.mean([prof[d] for d in donors], axis=0)
                self.T[k][key] = v
                self.N[k][key] = N + mm
            self.n_borrowed[k] = len(nbrs[k])
        return self


def borrow_waves(ub, nbrs):
    """波を写して、床未満の語に、近い語の波（N で割った物）の平均を m 回分足す。読みは同じ物を使う。"""
    import copy
    wb = copy.copy(ub)
    wb.W = {k: dict(v) for k, v in ub.W.items()}
    wb.N = {k: Counter(v) for k, v in ub.N.items()}
    for k in ub.KINDS:
        C = ub.code[k]
        for key, donors in nbrs[k].items():
            c = ub.cnt_all[k][key]
            N = sum(c.values())
            mm = ub.floor_n - N
            idx = np.fromiter(c.keys(), dtype=np.int64)
            val = np.fromiter(c.values(), dtype=np.float32)
            own = (val[:, None] * C[idx]).sum(0)
            lent = np.mean([ub.W[k][d] / ub.N[k][d] for d in donors], axis=0)
            wb.W[k][key] = (own + mm * lent).astype(np.complex64)
            wb.N[k][key] = N + mm
    return wb


def picks(m, te):
    return BD.picks(m, te)


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
    base = MD.Model(S, fill, a.r1).fit(tr, hold)           # 本体を先に学ぶ（属性の割り当ての順を I48bc・bd と同じにする）
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
    te_x = [s for s in te if AE.items(S, s, "節")]
    res = dict(学びの文=len(tr), 試しの文=len(te), TOPN=TOPN, DK=DK, 帯=BC.B, 床の符号=BC.P_PROBE, 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    t1 = time.time()
    ub = BC.UsageBoard(base, dk=DK).fit(base, tr)
    rows, nf = usage_vectors(ub)
    nbrs = neighbours(ub, rows, nf)
    Nl = {key: sum(c.values()) for key, c in ub.cnt_all["後ろ"].items()}
    floor = float(ub.floor_n)
    res["借りる"] = dict(床=round(floor, 2), 特徴の数=nf, 借りる語={k: len(v) for k, v in nbrs.items()},
                       貸す語={k: sum(1 for c in ub.cnt_all[k].values() if sum(c.values()) >= floor) for k in ub.KINDS}, 秒=round(time.time() - t1))
    print("借りる", json.dumps(res["借りる"], ensure_ascii=False), flush=True)

    def bands(m, q):
        c = Counter()
        for s, qq in zip(te_x, q):
            keys, gaps = MD.keys_of(s), G.split(s)[1]
            for i, ci in enumerate(qq):
                b = band_of(Nl.get(keys[i], 0), floor)
                c[b] += 1
                c[b + "当たり"] += m.pg.cls[ci] == gaps[i]
        return {b: dict(数=c[b], 当たり=round(c[b + "当たり"] / max(1, c[b]), 4)) for b in BANDS}

    def low_moves(m, q_no, q_yes):
        """床未満の帯で、借りる形が借りない形から替えた選びの数と、替えた後・前の当たり。"""
        ch = hit_new = hit_old = 0
        for s, a0, a1 in zip(te_x, q_no, q_yes):
            keys, gaps = MD.keys_of(s), G.split(s)[1]
            for i, (c0, c1) in enumerate(zip(a0, a1)):
                if band_of(Nl.get(keys[i], 0), floor) != "床未満" or c0 == c1:
                    continue
                ch += 1
                hit_new += m.pg.cls[c1] == gaps[i]
                hit_old += m.pg.cls[c0] == gaps[i]
        return dict(替えた選び=ch, 替えた後の当たり=hit_new, 替える前の当たり=hit_old)

    def changed(m, q0, q1):
        ch = un = hit = 0
        for s, a0, a1 in zip(te_x, q0, q1):
            keys, gaps = MD.keys_of(s), G.split(s)[1]
            xs = AE.items(S, s, "節")
            fic = m.who.fiction(keys)
            ups = {u for _, u in keys}
            for x, c0, c1 in zip(xs, a0, a1):
                if c0 == c1:
                    continue
                ch += 1
                g = m.gap_scores(keys, x, fic)
                logic = g["R2"] + m.gram.score(m.gram.state(ups, [m.pg.cls[c] for c in a1[:x["i"]]], x["last"]))
                if logic[c1] <= np.median(logic[:-1]):
                    un += 1
                    hit += m.pg.cls[c1] == gaps[x["i"]]
        return ch, un, hit

    def report(nm, m, q0=None, q_no=None):
        t2 = time.time()
        r = AP.measure(m, te, surf)
        r["言えるか"] = AX.score(m, te, say, nat)
        r["筋の無い共振"] = LN.monitor(m, te)
        q = picks(m, te)
        r["左の語の数えの帯ごと"] = bands(m, q)
        if q0 is not None:
            ch, un, hit = changed(m, q0, q)
            r.update(新しい語板を抜くと替わる選び=ch, 選びの数=sum(len(x) for x in q0), 想定外の反応=dict(想定外=un, 当たり=hit, 外れ=un - hit))
        if q_no is not None:
            r["床未満で借りない形から替えた選び"] = low_moves(m, q_no, q)
            ch, un, hit = changed(m, q_no, q)
            r["借りない形から替わった選び"] = dict(数=ch, 想定外=un, 当たり=hit, 外れ=un - hit)
        res["型"][nm] = r
        print(nm, round(time.time() - t2), "秒", json.dumps({k: r[k] for k in ("間", "文末", "文", "と_人", "左の語の数えの帯ごと")}, ensure_ascii=False), flush=True)
        save()
        return q

    def with_(x):
        m = base.clone()
        m.extras.append(x)
        m.refresh()
        return m

    q0 = report("本体", base)
    tab = BD.CountTable(base).fit(base, tr, ub)
    q_t = report("表・借りない", with_(tab), q0)
    tb = BorrowTable(base).fit(base, tr, ub, nbrs)
    report("表・借りる", with_(tb), q0, q_t)
    del tab, tb
    wb = borrow_waves(ub, nbrs)
    for mode in ("AM", "FM→AM"):
        ub.mode = wb.mode = mode
        q_w = report(f"波 {mode}・借りない", with_(ub), q0)
        report(f"波 {mode}・借りる", with_(wb), q0, q_w)
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
