"""I48as: 前処理 → 絞って走査の二段（I48ar の続き）。

- 一段目（前処理）: 位相で重ねた板（I48ar の (イ)、J = 2・3・8）を照らし、詰め物ごとに復号 d の上位 n 枚（n = 1・3・5）に絞る。
- 二段目（走査）: 絞った単位だけを 1 枚ずつ照らして正確な点 s を出し、I48ap の (B)（最大）または (C)（重なりで重みづけした平均）で合わせる。
- 比べ: 全部を 1 枚ずつ照らす (B)・(C)。当たり（間・文末・文、助詞ごとの当たりと害、「と」の順位）と、1 間あたりの照らす時間。
- 枚数: 自作 70 文の単位 159、それに学びに使っていない文から機械的に作った単位を足して 1,159・10,161。
  当たりは 159 で全部の組、1,159・10,161 では (B)・(C) と、159 で間がいちばん高かった組だけ（時間のため）。
- 想定外の反応（持ち主の頼み）: 1 枚ずつ照らすと点が 0（試す間と帯が 1 つも重ならない＝設計上は鳴らない）のに、重ねた板の復号で上位 n に入った単位。
  （d が 0 より大きい物だけ。単に鳴る単位が n より少なくて入った 0 の単位は数えない）。強さは、点が 0 なのに d が 0 でない単位の |d| の中央値（にじみの床）の何倍か。同じ単位が別の入力でくり返し入るか（くり返す単位の数と最大の回数）。しきい値は置かない。
- 筋の無い共振（持ち主の頼み、決まり）: 選んだ候補の点を、論理の筋（R2 語板の成分 ＋ R3 文法グレイン ＋ 介入の単位の点）と全体の波（R1）に分け、
  筋の点がその間の全部の候補の筋の点の中央値以下（いつもより鳴っていない）なのに、波の点がその間でいちばん高くて選ばれた物を記録する。
  当たり／外れ、左の語と選んだ詰め物の組のくり返しも数える。線は手で置かず、その間の候補の分布（中央値・いちばん）で決める。
手で置く物: J・n の並び、枚数（1,000・1 万を足す）、試しの間の数（想定外と時間の数え、1,000）。
python intervention/past/i48as_two_stage.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
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
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48ar_superpose as AR  # noqa: E402

JS = (2, 3, 8)
NS = (1, 3, 5)
EXTRA = tuple(int(x) for x in os.environ.get("AS_EXTRA", "1000,10000").split(","))
GAPS = int(os.environ.get("AS_GAPS", "1000"))


def overlap(U, idx, xP, xQ):
    o = np.abs(U["P"][idx]) @ xP
    for r, x in xQ.items():
        o = o + np.abs(U[r][idx]) @ x
    return o


class SModel(AP.UModel):
    """二段で選ぶ。mode = ("全部", B/C) または ("二段", B/C, J, n)。"""

    def set_mode(self, mode):
        self.mode = mode
        self.full = AR.Stack(self, self.units, 1)               # 1 枚ずつの板（U）
        self.stack = AR.Stack(self, self.units, mode[2]) if mode[0] == "二段" else None

    def class_scores(self, xP, xQ):
        out = {}
        X = self.full.exact(xP, xQ)
        L = self.stack.light(xP, xQ) if self.stack is not None else None
        for c, s in X.items():
            idx = np.arange(len(s))
            if L is not None:
                d = L[c][1]
                idx = np.argsort(-d)[: self.mode[3]]
            sc = s[idx]
            if self.mode[1] == "B":
                out[c] = float(sc.max())
            else:
                o = overlap(self.full.U[c], idx, xP, xQ)
                out[c] = float((o / o.sum()) @ sc) if o.sum() > 0 else 0.0
        return out

    def unit_term(self, keys, x, f, v):
        xP, xQ = AR.gap_inputs(self, keys, x)
        add = np.zeros(self.nc, dtype=np.float32)
        for c, s in self.class_scores(xP, xQ).items():
            add[c] = s
        return add


def unexpected(m, gaps_in, J, n):
    """想定外の反応と、照らす時間（二段・全部）。"""
    full, st = AR.Stack(m, m.units, 1), AR.Stack(m, m.units, J)
    cnt, ratios, n_short = Counter(), [], 0
    t2 = tf = 0.0
    for xP, xQ in gaps_in:
        t0 = time.perf_counter()
        L = st.light(xP, xQ)
        Xs = {}
        for c, (z, d, _) in L.items():
            idx = np.argsort(-d)[:n]
            U = full.U[c]
            sP = U["P"][idx] @ xP
            Xs[c] = (idx, sP)
        t2 += time.perf_counter() - t0
        t0 = time.perf_counter()
        X = full.exact(xP, xQ)
        tf += time.perf_counter() - t0
        for c, s in X.items():
            d = L[c][1]
            zero = s == 0
            nz = zero & (d != 0)                              # 点が 0 なのに重ねた板では鳴る単位（にじみ）
            floor = float(np.median(np.abs(d[nz]))) if nz.any() else 0.0
            for k in np.argsort(-d)[:n]:
                n_short += 1
                if s[k] == 0 and d[k] > 0:
                    u = full.by_c[c][k]
                    cnt[(c, u["番号"])] += 1
                    ratios.append(abs(float(d[k])) / max(1e-12, floor))
    rep = [v for v in cnt.values() if v >= 2]
    top = []
    for (c, k), v in cnt.most_common(5):
        u = m.units[k]
        top.append(dict(詰め物=m.pg.cls[c], 単位=k, 文=u["文"], 回数=v))
    nn = max(1, len(gaps_in))
    return dict(絞った単位=n_short, 想定外=sum(cnt.values()), 想定外の割合=round(sum(cnt.values()) / max(1, n_short), 4),
                床の何倍_中央値=round(float(np.median(ratios)), 2) if ratios else None, 床の何倍_最大=round(float(np.max(ratios)), 2) if ratios else None,
                くり返す単位=len(rep), くり返しの最大=max(cnt.values()) if cnt else 0, 多い単位=top,
                照らす秒_二段の一段目=round(t2 / nn, 5), 照らす秒_全部=round(tf / nn, 5))


def monitor(m, te):
    """筋の無い共振を数える。"""
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
            r3 = m.gram.score(m.gram.state(ups, placed, x["last"]))
            xP, xQ = AR.gap_inputs(m, keys, x)
            uni = np.zeros(m.nc, dtype=np.float32)
            for c, v in m.class_scores(xP, xQ).items():
                uni[c] = v
            logic = g["R2"] + r3 + uni
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
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr, extra_src = tr_all[-2000:], tr_all[:-2000][: a.train], tr_all[:-2000][a.train:]
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
    m = SModel(S, fill, a.r1).fit(tr, hold).init_units("B")
    for it in AO.TEACH_ALL:
        m.teach_unit(it)
    m.refresh()
    res = dict(学びの文=len(tr), 試しの文=len(te), JS=JS, NS=NS, 枚数={})
    gaps_in = []
    for s in te:
        keys = AK.keys_of(s)
        for x in AE.items(S, s, "節"):
            gaps_in.append(AR.gap_inputs(m, keys, x))
        if len(gaps_in) >= GAPS:
            break
    gaps_in = gaps_in[:GAPS]

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"学び {time.time() - t0:.0f} 秒、単位 {len(m.units)}", flush=True)
    src = iter(extra_src)
    n_made = len(m.units)
    best = None
    for target in (0,) + EXTRA:
        while len(m.units) < n_made + target:
            s = next(src)
            keys = AK.keys_of(s)
            if keys:
                m.teach_unit(("機械", "Tatoeba", keys, G.split(s)[1]))
        m.refresh()
        nu = len(m.units)
        o = res["枚数"][str(nu)] = dict(選び={}, 想定外と時間={})
        modes = [("全部", "B"), ("全部", "C")]
        if target == 0:
            modes += [("二段", k, J, n) for k in ("B", "C") for J in JS for n in NS]
        elif best is not None:
            modes += [best]
        for mode in modes:
            t1 = time.time()
            m.set_mode(mode)
            nm = "・".join(str(v) for v in mode)
            o["選び"][nm] = AP.measure(m, te, surf)
            print(nu, nm, round(time.time() - t1), "秒", json.dumps(o["選び"][nm], ensure_ascii=False), flush=True)
            save()
        if target == 0:
            best = max((md for md in modes if md[0] == "二段"), key=lambda md: o["選び"]["・".join(str(v) for v in md)]["間"])
            o["いちばん良い二段"] = "・".join(str(v) for v in best)
        if target == 0:
            for mode in (("全部", "B"), best):
                m.set_mode(mode)
                o.setdefault("筋の無い共振", {})["・".join(str(v) for v in mode)] = r = monitor(m, te)
                print(nu, "筋の無い共振", mode, json.dumps(r, ensure_ascii=False), flush=True)
                save()
        for J in JS:
            for n in NS:
                t1 = time.time()
                o["想定外と時間"][f"J {J}・n {n}"] = r = unexpected(m, gaps_in, J, n)
                print(nu, "想定外", J, n, round(time.time() - t1), "秒", json.dumps(r, ensure_ascii=False), flush=True)
                save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
