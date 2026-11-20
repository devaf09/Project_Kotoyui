"""I48bc: 新しい語板 1 版目（使われ方の波）。設計は docs/i48_new_board_v1.md（案、未確定）。本体は書き換えず、差し込み（extras）として足す。

語ごとに、使われ方を種類ごとに別の波（complex64、長さ DK）で持つ。種類 k の項目 j には、決まった乱数の位相の符号（白い雑音）を割り当て、
学びの文で出会うたびにその符号を足す（数えて足すだけ）。1 版目で選びに使うのは、次の 2 つ:
  後ろ: 語の後ろの間の詰め物（間 i の左の語の「後ろ」の波）
  前  : 語の前の間の詰め物（間 i の右の語の「前」の波）
（述語の属性・位置・印の種類は数えだけ持ち、選びにはまだ使わない。状況の波は後から別の種類として足せる口だけ空けておく。）

照らし方（ニュートンリングの読み）: 候補 c の符号 code_c と語の波 W を重ね、DK を B 本の帯に分けて、帯ごとの重なり z_b = Σ_{n∈b} conj(code_c[n]) W[n]。
  強さ（AM）: n̂_c = Re(Σ_b z_b) / DK（その使われ方の数えの見積もり。0 未満は 0）
  そろい具合（FM）: |Σ_b z_b| / Σ_b |z_b|（帯ごとの位相がそろえば 1、にじみなら小さい）
  FM の床: その語の波に、使われていない符号（種を変えた P 本）を重ねた時のそろい具合の 平均 ＋ 標準偏差 × √(2 ln V)（V = 箱の数）
点: P(c|語) = (n̂_c + p_c) / (N_語 + 1)、点 = log(P(c|語) / p_c)（p_c は学びの詰め物の割合。1 回分のならし）。
  型 AM: 全部の候補に点を足す。型 FM→AM: FM の床を越えた候補だけに点を足す（越えない候補は 0）。
数えの少ない語（N_語 が床 2 ln V に届かない）は、点を足さない（今の本体に頼る）。左右の語の点は、種類ごとに別に出して足す。
手で置く物: DK（2,048）、B（16）、FM の床を決める符号の本数 P（32）、乱数の種。
python intervention/i48bc_usage_board.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

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

DK = 2048
B = 16
P_PROBE = 32


def codes(n, dk, seed):
    r = np.random.default_rng(seed)
    return np.exp(2j * np.pi * r.random((n, dk))).astype(np.complex64)


class UsageBoard:
    """差し込み: 使われ方の波（後ろ・前）で詰め物の点を足す。mode は "AM" か "FM→AM"。"""

    KINDS = ("後ろ", "前")

    def __init__(self, m, dk=DK, bands=B, seed=0):
        self.dk, self.bands, self.mode = dk, bands, "FM→AM"
        self.code = {k: codes(m.nc, dk, seed + 101 * (i + 1)) for i, k in enumerate(self.KINDS)}
        self.probe = codes(P_PROBE, dk, seed + 7919)
        self.W = {k: {} for k in self.KINDS}
        self.N = {k: Counter() for k in self.KINDS}
        self.other_kinds = {k: defaultdict(Counter) for k in ("述語", "位置", "印")}   # 1 版目は数えだけ

    def clone(self):
        return self                                            # 学んだ後は読むだけ（写しても同じ物を指す）

    def refresh(self, m):
        pass

    def fit(self, m, tr):
        cnt = {k: defaultdict(Counter) for k in self.KINDS}
        pc = Counter()
        for s in tr:
            keys = MD.keys_of(s)
            if not keys:
                continue
            cont, gaps, head = G.split(s)
            xs = AE.items(m.S, s, "節")
            mk = tuple(sorted(xs[0]["mk"])) if xs else ()
            n = len(keys)
            for k, key in enumerate(keys):
                cnt["後ろ"][key][m.cid(gaps[k])] += 1
                cnt["前"][key][m.cid(head if k == 0 else gaps[k - 1])] += 1
                pred = next((c for c in keys[k + 1:] if c[1] in ("VERB", "ADJ")), None)
                self.other_kinds["述語"][key][pred[0] if pred else "無し"] += 1
                self.other_kinds["位置"][key]["頭" if k == 0 else ("末" if k == n - 1 else "中")] += 1
                self.other_kinds["印"][key][mk] += 1
            for g in gaps:
                pc[m.cid(g)] += 1
        tot = sum(pc.values())
        self.p = np.array([(pc[c] + 1) / (tot + m.nc) for c in range(m.nc)], dtype=np.float64)
        self.floor_n = 2 * np.log(m.nc)
        self.cnt = {k: {key: c for key, c in cnt[k].items() if sum(c.values()) >= self.floor_n} for k in self.KINDS}   # 比べ用に数えも残す
        self.cnt_all = cnt                                         # 床に届かない語の数えも残す（I48be の借りる試し用）
        for k in self.KINDS:
            for key, c in cnt[k].items():
                N = sum(c.values())
                if N < self.floor_n:
                    continue
                idx = np.fromiter(c.keys(), dtype=np.int64)
                val = np.fromiter(c.values(), dtype=np.float32)
                self.W[k][key] = (val[:, None] * self.code[k][idx]).sum(0)   # 出会うたびに符号を足す
                self.N[k][key] = N
        return self

    def read(self, k, key):
        """語の波を全部の候補で読む: (n̂, FM, FM の床)。"""
        W = self.W[k].get(key)
        if W is None:
            return None
        C = self.code[k]
        Z = (np.conj(C) * W[None, :]).reshape(C.shape[0], self.bands, -1).sum(2)   # 帯ごとの重なり
        tot = Z.sum(1)
        nhat = np.maximum(0.0, np.real(tot) / self.dk)
        fm = np.abs(tot) / np.maximum(1e-9, np.abs(Z).sum(1))
        Zp = (np.conj(self.probe) * W[None, :]).reshape(P_PROBE, self.bands, -1).sum(2)
        fp = np.abs(Zp.sum(1)) / np.maximum(1e-9, np.abs(Zp).sum(1))
        floor = float(fp.mean() + fp.std() * np.sqrt(2 * np.log(len(C))))
        return nhat, fm, floor

    def score(self, m, keys, x, f, v):
        out = np.zeros(m.nc, dtype=np.float32)
        i = x["i"]
        sides = [("後ろ", keys[i])]
        if not x["last"]:
            sides.append(("前", keys[i + 1]))
        for k, key in sides:
            r = self.read(k, key)
            if r is None:
                continue
            nhat, fm, floor = r
            pcw = (nhat + self.p) / (self.N[k][key] + 1)
            s = np.log(pcw / self.p)
            if self.mode == "FM→AM":
                s = np.where(fm > floor, s, 0.0)
            out += s.astype(np.float32)
        if not G.other_ok(m):
            out[-1] = 0
        return out


def picks(m, te):
    out = []
    for s in te:
        xs = AE.items(m.S, s, "節")
        if xs:
            out.append([r["選び"] for r in m.walk(MD.keys_of(s), xs)[0]])
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
    res = dict(学びの文=len(tr), 試しの文=len(te), DK=DK, 帯=B, 床の符号=P_PROBE, 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def report(nm, m):
        t1 = time.time()
        r = AP.measure(m, te, surf)
        r["言えるか"] = AX.score(m, te, say, nat)
        r["筋の無い共振"] = LN.monitor(m, te)
        res["型"][nm] = r
        print(nm, round(time.time() - t1), "秒", json.dumps({k: r[k] for k in ("間", "文末", "文", "と_人", "と_人以外")}, ensure_ascii=False), flush=True)
        save()
    report("本体（古い版）", base)
    p0 = picks(base, te)
    t1 = time.time()
    ub = UsageBoard(base).fit(base, tr)
    res["新しい語板"] = dict(学び秒=round(time.time() - t1), 語の数={k: len(v) for k, v in ub.W.items()}, 語の床=round(float(ub.floor_n), 2),
                         大きさ_MB=round(sum(w.nbytes for v in ub.W.values() for w in v.values()) / 1e6, 1))
    print("新しい語板", json.dumps(res["新しい語板"], ensure_ascii=False), flush=True)
    for mode in ("AM", "FM→AM"):
        m = base.clone()
        ub.mode = mode
        m.extras.append(ub)
        m.refresh()
        report("本体 ＋ 新しい語板（" + mode + "）", m)
        p1 = picks(m, te)
        ch = un = hit = 0
        ex = []
        for s, q0, q1 in zip([s for s in te if AE.items(S, s, "節")], p0, p1):
            keys, gaps = MD.keys_of(s), G.split(s)[1]
            xs = AE.items(S, s, "節")
            fic = m.who.fiction(keys)
            ups = {u for _, u in keys}
            for x, c0, c1 in zip(xs, q0, q1):
                if c0 == c1:
                    continue
                ch += 1
                g = m.gap_scores(keys, x, fic)
                logic = g["R2"] + m.gram.score(m.gram.state(ups, [m.pg.cls[c] for c in q1[:x["i"]]], x["last"]))
                if logic[c1] <= np.median(logic[:-1]):
                    un += 1
                    ok = m.pg.cls[c1] == gaps[x["i"]]
                    hit += ok
                    if len(ex) < 15:
                        ex.append(dict(要点="・".join(kk[0] for kk in keys), 間=x["i"], 元=gaps[x["i"]], 本体=m.pg.cls[c0], 新=m.pg.cls[c1]))
        res["型"]["本体 ＋ 新しい語板（" + mode + "）"].update(新しい語板を抜くと替わる選び=ch, 選びの数=sum(len(q) for q in p0),
                                                    想定外の反応=dict(想定外=un, 当たり=hit, 外れ=un - hit, 例=ex))
        print(mode, "替わる選び", ch, "想定外", un, flush=True)
        save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
