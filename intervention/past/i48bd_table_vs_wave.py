"""I48bd: 新しい語板 1 版目の確かめ。効いたのは「語ごとの使われ方の数え」か「波（と FM の読み）」か。

比べる形（同じ 5 万文・1,000 文、同じ測り）:
  (i)   本体（古い版）
  (ii)  本体 ＋ 普通の数えの表: 語の名前ごとの前後の詰め物の数え（波なし）。点・ならし・床 2 ln V は I48bc と同じ。
  (iii) 本体 ＋ 波の AM（DK = 512・2,048・8,192）
  (iv)  本体 ＋ 波の FM→AM（DK = 512・2,048・8,192）
あわせて出す物:
  - 少ない語: 間の左の語の、学びの数え N の帯（床未満・〜50・〜500・500 超）ごとの当たり
  - FM の床で落ちた候補: 表で数えが 1 以上なのに床を越えない候補の割合、数え 0 なのに床を越える候補の割合
  - 見積もりのずれ: 波の n̂ と表の数えの差（数えの有る候補で、平均の絶対の差）
  - 大きさ: 表（語 × 候補、数えの有る所だけ持つ時と、全部持つ時）、波（DK で決まる）
手で置く物: DK の並び（512・2,048・8,192）、B（16）、P（32）、種、N の帯の境（50・500）。
python intervention/past/i48bd_table_vs_wave.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
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
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48ax_sayable as AX  # noqa: E402
import i48bc_usage_board as BC  # noqa: E402
from i48core import learn as LN  # noqa: E402
from i48core import model as MD  # noqa: E402

DKS = (512, 2048, 8192)
N_EDGES = (50, 500)


class CountTable(BC.UsageBoard):
    """差し込み: 波の代わりに、語の名前ごとの前後の詰め物の数えをそのまま持つ（点の式・ならし・床は I48bc と同じ）。"""

    def __init__(self, m):
        self.mode = "AM"
        self.N = {k: Counter() for k in self.KINDS}
        self.T = {k: {} for k in self.KINDS}
        self.nc = m.nc

    def fit(self, m, tr, ub):
        """数えは ub（同じ学びの文で数えた I48bc の板）から写す。"""
        self.p, self.floor_n = ub.p, ub.floor_n
        for k in self.KINDS:
            for key, c in ub.cnt[k].items():
                v = np.zeros(self.nc, dtype=np.float32)
                v[np.fromiter(c.keys(), dtype=np.int64)] = np.fromiter(c.values(), dtype=np.float32)
                self.T[k][key] = v
                self.N[k][key] = ub.N[k][key]
        return self

    def read(self, k, key):
        v = self.T[k].get(key)
        if v is None:
            return None
        return v, np.ones(self.nc), 0.0


def picks(m, te):
    out = []
    for s in te:
        xs = AE.items(m.S, s, "節")
        if xs:
            out.append([r["選び"] for r in m.walk(MD.keys_of(s), xs)[0]])
    return out


def band_of(n):
    return "床未満" if n == 0 else ("〜50" if n <= N_EDGES[0] else ("〜500" if n <= N_EDGES[1] else "500超"))


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    ap.add_argument("--dks", nargs="*", type=int, default=list(DKS))
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
    base = MD.Model(S, fill, a.r1).fit(tr, hold)           # 本体を先に学ぶ（属性の割り当ての順を I48bc と同じにする）
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
    res = dict(学びの文=len(tr), 試しの文=len(te), 帯=BC.B, 床の符号=BC.P_PROBE, DKの並び=a.dks, 型={}, 大きさ={}, 波の読み={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def by_band(m, q, Nl):
        c = Counter()
        for s, qq in zip(te_x, q):
            keys, gaps = MD.keys_of(s), G.split(s)[1]
            for i, ci in enumerate(qq):
                b = band_of(Nl.get(keys[i], 0))
                c[b] += 1
                c[b + "当たり"] += m.pg.cls[ci] == gaps[i]
        return {b: dict(数=c[b], 当たり=round(c[b + "当たり"] / max(1, c[b]), 4)) for b in ("床未満", "〜50", "〜500", "500超")}

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

    def report(nm, m, Nl=None, q0=None, q_ref=None):
        t1 = time.time()
        r = AP.measure(m, te, surf)
        r["言えるか"] = AX.score(m, te, say, nat)
        r["筋の無い共振"] = LN.monitor(m, te)
        q = picks(m, te)
        if Nl is not None:
            r["左の語の数えの帯ごと"] = by_band(m, q, Nl)
        if q0 is not None:
            ch, un, hit = changed(m, q0, q)
            r.update(新しい語板を抜くと替わる選び=ch, 選びの数=sum(len(x) for x in q0), 想定外の反応=dict(想定外=un, 当たり=hit, 外れ=un - hit))
        if q_ref is not None:
            r["表と選びが違う間"] = sum(x != y for a_, b_ in zip(q_ref, q) for x, y in zip(a_, b_))
        res["型"][nm] = r
        print(nm, round(time.time() - t1), "秒", json.dumps({k: r[k] for k in ("間", "文末", "文", "と_人", "と_人以外")}, ensure_ascii=False), flush=True)
        save()
        return q

    q0 = report("(i) 本体", base)
    ub0 = BC.UsageBoard(base, dk=a.dks[0]).fit(base, tr)    # 数えはどの DK でも同じ。表はここから写す
    tab = CountTable(base).fit(base, tr, ub0)
    Nl = dict(tab.N["後ろ"])
    nz = sum(int((v > 0).sum()) for d in tab.T.values() for v in d.values())
    res["大きさ"]["表"] = dict(語の数={k: len(v) for k, v in tab.T.items()},
                             全部持つ_MB=round(sum(len(v) for v in tab.T.values()) * base.nc * 4 / 1e6, 2),
                             数えの有る所だけ_MB=round(nz * (4 + 4) / 1e6, 2), 数えの有る所=nz)
    m = base.clone()
    m.extras.append(tab)
    m.refresh()
    q_tab = report("(ii) 本体 ＋ 数えの表", m, Nl, q0)
    del ub0
    for dk in a.dks:
        t1 = time.time()
        ub = BC.UsageBoard(base, dk=dk).fit(base, tr)
        res["大きさ"][f"波 DK {dk}"] = dict(学び秒=round(time.time() - t1), 語の波_MB=round(sum(w.nbytes for v in ub.W.values() for w in v.values()) / 1e6, 1),
                                         符号_MB=round(sum(c.nbytes for c in ub.code.values()) / 1e6, 1))
        # 波の読みと表のずれ、FM の床で落ちる候補
        err, n_err, drop, n_pos, false_up, n_zero = 0.0, 0, 0, 0, 0, 0
        for k in ub.KINDS:
            for key, v in tab.T[k].items():
                nhat, fm, floor = ub.read(k, key)
                pos = v > 0
                err += float(np.abs(nhat[pos] - v[pos]).sum())
                n_err += int(pos.sum())
                up = fm > floor
                drop += int((pos & ~up).sum())
                n_pos += int(pos.sum())
                false_up += int((~pos & up).sum())
                n_zero += int((~pos).sum())
        res["波の読み"][f"DK {dk}"] = dict(数えの有る候補の平均の絶対の差=round(err / max(1, n_err), 3),
                                        数え有りで床を越えない割合=round(drop / max(1, n_pos), 4),
                                        数え0で床を越える割合=round(false_up / max(1, n_zero), 4))
        print("DK", dk, json.dumps(res["大きさ"][f"波 DK {dk}"], ensure_ascii=False), json.dumps(res["波の読み"][f"DK {dk}"], ensure_ascii=False), flush=True)
        save()
        for mode in ("AM", "FM→AM"):
            ub.mode = mode
            m = base.clone()
            m.extras.append(ub)
            m.refresh()
            report(f"(iii/iv) 本体 ＋ 波 {mode} DK {dk}", m, Nl, q0, q_tab)
        del ub, m
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
