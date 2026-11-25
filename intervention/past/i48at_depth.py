"""I48at (一): 介入の単位の記憶深さ（持ち主の方針: 全体に同じ重みの学習は、ある所で誤判定が始まる。学習の重みを決める記憶深さのグレイン）。

- 単位は I48ap と同じ（自作 70 文 × 詰め物）。土台は学び 5 万文。介入の板は本体に融合しない。記録と取り消しは I48ap のまま。
- 深さ（単位ごと）: 練習の文（学びにも試しにも使っていない文）で照らして選ぶたびに、その選び（詰め物）に効いた単位の「使われた回数」と
  「そのうち当たった回数」を数える。効いた単位は、(B) ならいちばん強く鳴った 1 つ、(C) なら重なりの重み（和 1）で分けて数える。
    深さ = (当たった回数 + p0) / (使われた回数 + 1)、p0 は練習の全部の単位の当たりの割合（ならしは 1 回分）。
    単位の点に 深さ / p0 を掛ける（使われていない単位は 1 のまま。いつもより当たる単位は強く、外れを呼ぶ単位は弱く）。記憶は消さない。
- 驚きの重さ（別の型）: 教える順に、新しい単位と既にある同じ詰め物の単位の重なり（P と Q の帯の重さのコサインの最大）を測り、
  重さ = 1 − 最大のコサイン（既に知っている形ほど浅く、新しい形ほど深く）。単位の点に掛ける。
- 比べ: 同じ重さの (B)・(C)（I48ap）と、(B)・(C) × 深さ・驚き。測る物は I48ap と同じ ＋ 筋の無い共振（I48as の見張り）。
手で置く物: 練習の文の数（2,000）。
python intervention/past/i48at_depth.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000] [--practice 2000]
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
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48ar_superpose as AR  # noqa: E402
import i48as_two_stage as AS  # noqa: E402


class DModel(AS.SModel):
    """単位ごとの重さ amp（深さ・驚き）を掛けて照らす。"""

    def set_amp(self, amp):
        self.amp = amp                                          # {単位の番号: 重さ}

    def class_scores(self, xP, xQ, detail=False):
        out, used = {}, {}
        X = self.full.exact(xP, xQ)
        for c, s in X.items():
            nums = np.array([u["番号"] for u in self.full.by_c[c]])
            a = np.array([self.amp.get(int(k), 1.0) for k in nums], dtype=np.float32) if getattr(self, "amp", None) else 1.0
            s = s * a
            if self.mode[1] == "B":
                k = int(np.argmax(s))
                out[c] = float(s[k])
                used[c] = {int(nums[k]): 1.0}
            else:
                o = AS.overlap(self.full.U[c], np.arange(len(s)), xP, xQ)
                w = o / o.sum() if o.sum() > 0 else np.zeros(len(s))
                out[c] = float(w @ s)
                used[c] = {int(nums[k]): float(w[k]) for k in np.nonzero(w)[0]}
        return (out, used) if detail else out


def practice_depth(m, pr):
    """練習の文で、選びに効いた単位の使われた回数と当たった回数を数え、深さの重さを返す。"""
    n_used, n_hit = {}, {}
    for s in pr:
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        keys = AK.keys_of(s)
        gaps = G.split(s)[1]
        rs, fic = m.walk(keys, xs)
        for x, r in zip(xs, rs):
            c = r["選び"]
            xP, xQ = AR.gap_inputs(m, keys, x)
            _, used = m.class_scores(xP, xQ, detail=True)
            ok = m.pg.cls[c] == gaps[x["i"]]
            for k, w in used.get(c, {}).items():
                n_used[k] = n_used.get(k, 0) + w
                n_hit[k] = n_hit.get(k, 0) + w * ok
    tot_u = sum(n_used.values())
    p0 = sum(n_hit.values()) / tot_u if tot_u else 0.5
    amp = {k: ((n_hit[k] + p0) / (n_used[k] + 1)) / max(1e-9, p0) for k in n_used}
    return amp, dict(使われた単位=len(n_used), p0=round(p0, 4), 重さの最小=round(min(amp.values()), 3) if amp else None,
                     重さの最大=round(max(amp.values()), 3) if amp else None, 重さの中央値=round(float(np.median(list(amp.values()))), 3) if amp else None)


def surprise_amp(m):
    """教えた順に、同じ詰め物の既にある単位との重なり（帯の重さのコサインの最大）から、重さ = 1 − 最大のコサイン。"""
    amp, seen = {}, {}
    for u in m.units:
        if u["取り消し"]:
            continue
        D = m.S.D
        v = np.zeros(D * 4, dtype=np.float32)
        pf, pw = u["P"]
        v[pf] += pw
        for j, r in enumerate(AP.ROLES):
            if r in u["Q"]:
                qf, qw = u["Q"][r]
                v[D * (j + 1) + qf] += qw
        v /= max(1e-9, float(np.linalg.norm(v)))
        prev = seen.setdefault(u["c"], [])
        amp[u["番号"]] = 1.0 - (max(float(v @ p) for p in prev) if prev else 0.0)
        prev.append(v)
    return amp


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--practice", type=int, default=2000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    pr = tr_all[:-2000][a.train: a.train + a.practice]          # 練習の文: 学びにも試しにも取りおきにも使っていない文
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
    m = DModel(S, fill, a.r1).fit(tr, hold).init_units("B")
    for it in AO.TEACH_ALL:
        m.teach_unit(it)
    m.refresh()
    res = dict(学びの文=len(tr), 試しの文=len(te), 練習の文=len(pr), 単位=len(m.units), 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"学び {time.time() - t0:.0f} 秒、単位 {len(m.units)}", flush=True)
    sur = surprise_amp(m)
    for k in ("B", "C"):
        m.set_mode(("全部", k))
        for nm in ("同じ重さ", "深さ", "驚き"):
            t1 = time.time()
            info = None
            if nm == "同じ重さ":
                m.set_amp(None)
            elif nm == "深さ":
                m.set_amp(None)
                amp, info = practice_depth(m, pr)
                m.set_amp(amp)
            else:
                m.set_amp(sur)
                info = dict(重さの最小=round(min(sur.values()), 3), 重さの中央値=round(float(np.median(list(sur.values()))), 3))
            r = AP.measure(m, te, surf)
            r["筋の無い共振"] = AS.monitor(m, te)
            r["重さ"] = info
            res["型"][f"{k}・{nm}"] = r
            print(k, nm, round(time.time() - t1), "秒", json.dumps(r, ensure_ascii=False), flush=True)
            save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
