"""I48ap: 教えた文が増えると薄め合う点の直し（I48ao の続き）。学び 5 万、型は 中間（鳴った役だけの点の平均）、教える文は I48ao の自作 70 文。

今（I48ao、型 A）は詰め物ごとに 1 枚の板に足して、教えた回数で割る。似ていない文が互いを薄めている。比べる形:
  (A) 今のまま（回数で割る）。数字は I48ao の 学び 5 万・中間 をそのまま使う（同じ土台・同じ手順）。
  (B) 教えた文ごとに小さな板（単位 = 文 × 詰め物）を別々に持ち、試す間では、その詰め物の単位のうちいちばん強く鳴る 1 つの点を使う（最大）。
  (C) 単位の点を、試す間との重なり（単位に載せた重さ × 試す間の重さ の和、P と Q の役の和）で重みづけして平均する（重みは和 1、しきい値なし）。
  (D) 1 枚の板に足すが、帯ごとに「その帯に実際に載せた単位の数」で割る（回数ではなく）。
単位の点（B・C）: 単位の 1 回分の板として、4.38 の sep_ratio の形（回数 1）で P（並び）と Q（属性、役ごと）の比を取り、
  P の点 ＋ Q の役の点の 中間（鳴った役だけの平均）。教えていない帯は 0。
介入の板は本体に融合しない。単位ごとの記録を残し、単位の番号で取り消せる（undo）。float32、数えと足し算だけ。
測る物: I48ao と同じ（間・文末・文、助詞ごとの当たりと害、「と」の順位）＋ 元が「と」の間を左の語が人のクラス（I48an の HUMAN）か否かで分けた当たり・上位 3。
手で置く物: なし（教える文・段・順の数は I48ao と同じ）。
python intervention/past/i48ap_units.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--kinds B C D]
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
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48am_attr_weight as AM  # noqa: E402
import i48an_attr_board as AN  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402

ROLES = AN.ROLES


def lookup(ub, uv, f, v):
    """単位の帯 ub（並べ済み）と値 uv から、試す帯 f（重さ v）に当たる分の和 Σ v × uv。"""
    if len(ub) == 0:
        return 0.0
    idx = np.searchsorted(ub, f)
    idx = np.minimum(idx, len(ub) - 1)
    hit = ub[idx] == f
    return float((uv[idx[hit]] * v[hit]).sum())


def pack(f, w):
    """帯ごとにまとめて並べる（同じ帯は足す）。"""
    ub, inv = np.unique(f, return_inverse=True)
    uw = np.zeros(len(ub), dtype=np.float32)
    np.add.at(uw, inv, w.astype(np.float32))
    return ub, uw


class UnitScore:
    """差し込み: 介入の単位の点（UModel.unit_term）を今の点に足す。"""

    def score(self, m, keys, x, f, v):
        return m.unit_term(keys, x, f, v)

    def refresh(self, m):
        pass

    def clone(self):
        return self


class UModel(AO.OModel):
    """単位ごとの介入の板（B・C）と、帯ごとの単位の数で割る板（D）。本体と P・Q の今の板（A）は使わない（差し込みは UnitScore だけ）。"""

    def init_units(self, kind):
        self.init_q("中間")
        self.extras = [e for e in self.extras if not isinstance(e, (AN.QScore, UnitScore))] + [UnitScore()]
        self.kind, self.units = kind, []
        self.Pd = np.zeros_like(self.pg.W, dtype=np.float32)          # D: 足した重さ
        self.Pn = np.zeros_like(self.pg.W, dtype=np.float32)          # D: 帯ごとの単位の数
        self.Qd = {r: np.zeros_like(self.rR.W, dtype=np.float32) for r in ROLES}
        self.Qn = {r: np.zeros_like(self.rR.W, dtype=np.float32) for r in ROLES}
        self.LD = None
        return self

    def clone(self):
        m = super().clone()
        m.units = [dict(u) for u in self.units]
        m.Pd, m.Pn = self.Pd.copy(), self.Pn.copy()
        m.Qd = {r: a.copy() for r, a in self.Qd.items()}
        m.Qn = {r: a.copy() for r, a in self.Qn.items()}
        m.LD = None
        return m

    def logpm(self, W):
        return np.log((W + G.ALPHA) / (W.sum(1, keepdims=True) + G.ALPHA * W.shape[1])).astype(np.float32)

    def teach_unit(self, item, xs=None):
        """1 文を教える: 間ごとに単位（文 × 詰め物）を作る。xs を渡せば、その間（実際の文の印つき）だけを教える。"""
        axis, text, keys, gaps = item
        xs = AK.made_items(self.S, keys, gaps, ("過去",)) if xs is None else xs
        for x in xs:
            c = self.cid(x["y"])
            if c == self.nc - 1 and not G.other_ok(self):
                continue
            f, v = self.feat(x)
            v = np.asarray(v, dtype=np.float64)
            pf, pw = pack(f, v / max(1e-9, float(v.sum())))
            u = dict(番号=len(self.units), 文=text, 間=x["i"], 詰め物=self.pg.cls[c], c=c, P=(pf, pw), Q={}, 取り消し=False)
            for r, key in AN.role_keys(keys, x["i"]).items():
                if key is None:
                    continue
                qf, qv = self.S.attrs(*key)
                qv = np.asarray(qv, dtype=np.float64)
                u["Q"][r] = pack(qf, qv / max(1e-9, float(qv.sum())))
            self.units.append(u)
            self.add_d(u, +1)
            self.log.append(dict(番号=u["番号"], 時刻=time.strftime("%F %T"), 文=text, 間=x["i"], 詰め物=u["詰め物"], 板="単位", 型=self.kind, 取り消し=False))

    def add_d(self, u, sign):
        c = u["c"]
        pf, pw = u["P"]
        self.Pd[c, pf] += sign * pw
        self.Pn[c, pf] += sign
        for r, (qf, qw) in u["Q"].items():
            self.Qd[r][c, qf] += sign * qw
            self.Qn[r][c, qf] += sign

    def undo(self, numbers):
        for k in numbers:
            u = self.units[k]
            if not u["取り消し"]:
                self.add_d(u, -1)
                u["取り消し"] = True
        self.refresh()

    def refresh(self):
        AK.Model.refresh(self)                                    # 本体の表だけ（A の板は使わない）
        if not hasattr(self, "units"):
            return
        self.lpm = self.logpm(self.pg.W)
        self.lqm = {r: self.logpm(self.board(r).W) for r in ROLES}
        live = [u for u in self.units if not u["取り消し"]]
        for u in live:                                            # 単位の比（回数 1 の sep_ratio）を帯ごとに持つ
            c = u["c"]
            pf, pw = u["P"]
            pm = np.exp(self.lpm[c, pf])
            u["LP"] = (np.log((pw + pm) / 2) - np.log(pm)).astype(np.float32)
            u["LQ"] = {}
            for r, (qf, qw) in u["Q"].items():
                qm = np.exp(self.lqm[r][c, qf])
                u["LQ"][r] = (np.log((qw + qm) / 2) - np.log(qm)).astype(np.float32)
        self.by_c = {}
        for u in live:
            self.by_c.setdefault(u["c"], []).append(u)
        if self.kind == "D" and live:
            def ratio(Wd, Wn, lpm):
                pm = np.exp(lpm)
                L = np.log((Wd + pm) / (Wn + 1)) - lpm
                L[Wn <= 0] = 0
                L[-1] = 0
                return L.astype(np.float32)
            self.LD = dict(P=ratio(self.Pd, self.Pn, self.lpm), Q={r: ratio(self.Qd[r], self.Qn[r], self.lqm[r]) for r in ROLES})
        else:
            self.LD = None

    def unit_scores(self, keys, i, f, v):
        """詰め物ごとの介入の点（型 B・C・D）。"""
        out = np.zeros(self.nc, dtype=np.float32)
        if not getattr(self, "units", None):
            return out
        rk = AN.role_keys(keys, i)
        rq = {r: self.S.attrs(*key) for r, key in rk.items() if key is not None}
        rq = {r: (qf, np.asarray(qv, dtype=np.float64) / max(1e-9, float(np.sum(qv)))) for r, (qf, qv) in rq.items()}
        if self.kind == "D":
            if self.LD is None:
                return out
            out += self.LD["P"][:, f] @ v
            parts = [self.LD["Q"][r][:, qf] @ qv for r, (qf, qv) in rq.items()]
            if parts:
                P = np.stack(parts)
                out += P.sum(0) / np.maximum(1, (P != 0).sum(0))
            return out
        for c, us in self.by_c.items():
            sc, ov = [], []
            for u in us:
                pf, pw = u["P"]
                s = lookup(pf, u["LP"], f, v)
                o = lookup(pf, pw, f, v)
                qs = []
                for r, (qf, qv) in rq.items():
                    if r in u["Q"]:
                        uf, uw = u["Q"][r]
                        x = lookup(uf, u["LQ"][r], qf, qv)
                        if x != 0:
                            qs.append(x)
                        o += lookup(uf, uw, qf, qv)
                s += sum(qs) / len(qs) if qs else 0.0
                sc.append(s)
                ov.append(o)
            sc, ov = np.array(sc), np.array(ov)
            if self.kind == "B":
                out[c] = sc.max()
            else:
                out[c] = float((ov / ov.sum()) @ sc) if ov.sum() > 0 else 0.0
        return out

    def unit_term(self, keys, x, f, v):
        """今の点に足す介入の単位の点（RModel・SModel はここを替える）。"""
        v = np.asarray(v, dtype=np.float64)
        return self.unit_scores(keys, x["i"], f, v / max(1e-9, float(v.sum())))


def measure(m, te, surf):
    """I48ao の数え ＋ 元が「と」の間を、左の語が人のクラスか否かで分けた当たり・上位 3。"""
    r = AO.measure(m, te, surf)
    wc = m.cid("と")
    c = dict(人=0, 人当たり=0, 人上位3=0, 他=0, 他当たり=0, 他上位3=0)
    for s in te:
        xs = AE.items(m.S, s, "節")
        gaps = G.split(s)[1]
        if not xs or "と" not in gaps[:-1]:
            continue
        keys = AK.keys_of(s)
        rs, _ = m.walk(keys, xs)
        for i, (y, q) in enumerate(zip(gaps, rs)):
            if y == "と" and i + 1 < len(gaps):
                h = "人" if AN.is_human(m.S, keys[i]) else "他"
                c[h] += 1
                c[h + "当たり"] += m.pg.cls[q["選び"]] == "と"
                c[h + "上位3"] += AJ.ranks(q["今×三つ"])[wc] < 3
    d = lambda a, b: round(c[a] / max(1, c[b]), 3)
    r["と_人"] = dict(数=c["人"], 当たり=d("人当たり", "人"), 上位3=d("人上位3", "人"))
    r["と_人以外"] = dict(数=c["他"], 当たり=d("他当たり", "他"), 上位3=d("他上位3", "他"))
    return r


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    ap.add_argument("--kinds", nargs="*", default=["B", "C", "D"])
    ap.add_argument("--orders", type=int, default=3)
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
    base = UModel(S, fill, a.r1).fit(tr, hold).init_units("B")
    base.refresh()
    res = dict(学びの文=len(tr), 試しの文=len(te), R1の型=a.r1, 教える文の数=len(AO.TEACH_ALL), 土台=measure(base, te, surf), 型={})
    out_dir = os.path.dirname(os.path.abspath(a.out))
    stem = os.path.splitext(os.path.basename(a.out))[0]

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"学び {time.time() - t0:.0f} 秒", json.dumps(res["土台"], ensure_ascii=False), flush=True)
    save()
    for kind in a.kinds:
        o = res["型"][kind] = {}
        for seed in range(a.orders):
            order = [AO.TEACH_ALL[k] for k in np.random.default_rng(seed).permutation(len(AO.TEACH_ALL))]
            m = base.clone()
            m.kind = kind
            done = 0
            o[f"順 {seed}"] = {}
            for st in AO.STAGES:
                t1 = time.time()
                upto = len(order) if st is None else st
                for it in order[done:upto]:
                    m.teach_unit(it)
                m.refresh()
                done = upto
                o[f"順 {seed}"][str(upto)] = r = measure(m, te, surf)
                print(kind, "順", seed, upto, "文", round(time.time() - t1), "秒", json.dumps(r, ensure_ascii=False), flush=True)
                save()
            o[f"順 {seed}"]["教えた順"] = [it[1] for it in order]
            m.save_log(os.path.join(out_dir, f"{stem}_{kind}_順{seed}_記録.jsonl"))
            save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
