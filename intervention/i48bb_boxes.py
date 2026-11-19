"""I48bb: 箱（詰め物の候補）の数を増やす（持ち主の案、未確定: 大体合う選びができて負荷がそれほど増えないなら増やして良い。
裾の箱は板だけ持って選びに出さず、数えが溜まった時に起こす）。

比べ: 箱の数 n_cls = 300・1,000・3,000（学びの上位 n_cls 個 ＋ その他）。それぞれ 2 つの形:
  全部起こす: 全部の箱を選びに出す。
  床で起こす: 学びの回数 n_c が床 2 ln V（V = 箱の数）を越えた箱だけ選びに出し、ほかは板だけ持つ（差し込み Dormant で選びから外す。
    本体の数えは変えず、差し込みを抜けば元に戻る）。床は「回数のゆらぎ √n_c の √(2 ln V) 倍を越える」と同じ式で、手の線なし。
測る物: 間・文末・文、助詞ごとの当たりと害、と の人・人以外、言える（物差しは学んでいない文）、選んだ「その他」の割合と、元がその他の間の割合、
  1 間あたりの選びの秒（試しの文の walk を測った秒）、学びの秒、板の大きさ（MB）、筋の無い共振、想定外の反応（300 の本体と比べて選びが
  替わり、論理の筋が中央値以下の物）。
手で置く物: 箱の数の並び（300・1,000・3,000）。
python intervention/i48bb_boxes.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000] [--boxes 300 1000 3000]
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
from i48core import learn as LN  # noqa: E402
from i48core import model as MD  # noqa: E402


class Dormant:
    """差し込み: 床を越えていない箱を選びから外す（点に −1e9 を足す）。板の数えは変えない。"""

    def __init__(self, mask):
        self.mask = mask

    def score(self, m, keys, x, f, v):
        out = np.zeros(m.nc, dtype=np.float32)
        out[self.mask] = -1e9
        return out

    def refresh(self, m):
        pass

    def clone(self):
        return self


def board_mb(m):
    arrs = [m.pg.W, m.pg.logp, m.Wd, m.Wi, m.who.B, m.rR.W, m.rL.W, m.rP.W, m.gram.R.W, m.rR.L, m.rL.L, m.rP.L, m.gram.R.L]
    return round(sum(a.nbytes for a in arrs if a is not None) / 1e6, 1)


def picks(m, te):
    out, t0 = [], time.time()
    n = 0
    for s in te:
        xs = AE.items(m.S, s, "節")
        if xs:
            rs, _ = m.walk(MD.keys_of(s), xs)
            out.append([m.pg.cls[r["選び"]] for r in rs])
            n += len(xs)
    return out, (time.time() - t0) / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    ap.add_argument("--boxes", nargs="*", type=int, default=[300, 1000, 3000])
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    ref = tr_all[:-2000][a.train:]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    res = dict(学びの文=len(tr), 試しの文=len(te), 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    base_seq = None
    for nb in a.boxes:
        S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
        fill, surf = AD.Fill(S), G.Surface()
        for s in tr:
            fill.see(s)
            cont, gaps, _ = G.split(s)
            for (l, f_, u), g in zip(cont, gaps):
                surf.add(l, f_, u, g)
        t1 = time.time()
        m = MD.Model(S, fill, a.r1)
        m.n_cls = nb
        m.fit(tr, hold)
        fit_s = round(time.time() - t1)
        for s in te:
            AE.items(S, s, "節")
        fill_ref = AD.Fill(S)
        for s in ref:
            fill_ref.see(s)
        say, nat = AX.Sayable(S, ref), RC.Natural(fill_ref, ref)
        floor = 2 * np.log(m.nc)
        dormant = (m.pr < floor)
        dormant[-1] = False                                     # その他は今まで通り（伏せは OTHER が決める）
        for mode in ("全部起こす", "床で起こす"):
            mm = m.clone()
            if mode == "床で起こす":
                mm.extras.append(Dormant(dormant))
            mm.refresh()
            seqs, per_gap = picks(mm, te)
            r = AP.measure(mm, te, surf)
            r["言えるか"] = AX.score(mm, te, say, nat)
            r["筋の無い共振"] = LN.monitor(mm, te)
            known = set(c for c in mm.pg.cls[:-1])
            n_gap = sum(len(q) for q in seqs)
            r.update(箱の数=nb, 起きている箱=int(m.nc - 1 - dormant[:-1].sum()) if mode == "床で起こす" else m.nc - 1,
                     床=round(float(floor), 2), 学びの秒=fit_s, 一間の選びの秒=round(per_gap, 5), 板の大きさ_MB=board_mb(mm),
                     選んだその他の割合=round(sum(g is None for q in seqs for g in q) / max(1, n_gap), 4),
                     元がその他の間の割合=round(sum(y not in known for s in te for y in G.split(s)[1][:-1]) / max(1, sum(len(G.split(s)[1]) - 1 for s in te)), 4))
            if base_seq is None:
                base_seq = seqs
            else:                                               # 想定外の反応: 300 の本体と比べて選びが替わり、論理の筋が中央値以下
                ch = un = hit = 0
                for s, q0, q1 in zip([s for s in te if AE.items(S, s, "節")], base_seq, seqs):
                    keys, gaps = MD.keys_of(s), G.split(s)[1]
                    xs = AE.items(S, s, "節")
                    fic = mm.who.fiction(keys)
                    ups = {u for _, u in keys}
                    for x, g0, g1 in zip(xs, q0, q1):
                        if g0 == g1:
                            continue
                        ch += 1
                        g = mm.gap_scores(keys, x, fic)
                        logic = g["R2"] + mm.gram.score(mm.gram.state(ups, q1[:x["i"]], x["last"]))
                        c = mm.cid(g1)
                        if logic[c] <= np.median(logic[:-1]):
                            un += 1
                            hit += g1 == gaps[x["i"]]
                r["想定外の反応"] = dict(箱300と比べて替わった選び=ch, 想定外=un, 当たり=hit, 外れ=un - hit)
            res["型"][f"{nb}・{mode}"] = r
            print(nb, mode, json.dumps({k: r[k] for k in ("間", "文末", "文", "起きている箱", "一間の選びの秒", "板の大きさ_MB", "学びの秒")}, ensure_ascii=False), flush=True)
            save()
        del m
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
