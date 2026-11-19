"""I48ay: 役の位相鍵（提案 ⑤ 直交位相鍵 ＋ ⑥ 雑音の床）。本体（i48core.model.Model）は書き換えず、差し込み（extras）として足す。

ねらい: 今の本体は、左の語・右の語・述語を別々の板（R2）で数え、点を足すだけ。そのため「左の語 × 述語」の組み合わせ
（例: 友達〔人〕× 行く〔移動〕→ と）が板に残らない。役ごとの鍵で回した語の波どうしを掛けて（束ねて）重ねると、組み合わせが
ほかの組と混ざらずに 1 本の波に残る。

作り:
  - 語の波: 語の属性ごとに決まった乱数の位相の符号（長さ DK、拡散）を、属性の重さで足した物（i48ab_place の「拡散」と同じ考え）。
  - 役の鍵: 役（左・右・述語。i48an の ROLES と同じ）と意味の印（過去・否定・丁寧・進行・問い、文末）に、BandTable（i48x_mesh）で
    衝突しない周波数を割り当て、鍵 = 位相の傾き exp(2πi f n / DK)。役の表（共同動作・行き先 等）は手で書かない。
  - 間の波 Z: (左, 述語)・(左, 右)・(右, 述語) と (左, 印) の組ごとに、鍵で回した 2 つの波の積を足し、長さ 1 にならす。文末の間は (左, 印) だけ。
  - 板: 詰め物 c ごとに、学びの間の Z を足した B_c と回数 n_c（足すだけ）。
  - 点: Δ_c = Re<B_c, Z>/n_c − Re<ΣB, Z>/N（いつもよりどれだけ強く鳴るか）。σ = 1.4826 × MAD(Δ)（数えのある詰め物の上で）、
    床 t = σ √(2 ln V)（V = 数えのある詰め物の数）。Δ_c > t の詰め物だけ log(Δ_c / t) を今の点に足し、ほかは 0（候補に入れる線は床だけ）。
手で置く物: DK（2,048。波の長さ。速さと容量の都合）、符号の乱数の種。
測る物（比べる相手は今の本体、同じ重さ）:
  間・文末・文、助詞ごとの当たりと害、と の例の順位、と の人・人以外、I48ax の「言える」の割合（物差しは学んでいない文）、
  筋の無い共振、想定外の反応（鍵の点が選びを替え、その選びの論理の筋〔R2 ＋ R3〕が中央値以下の物）、
  波が効いているかの数（R1 を抜くと選びがいくつ変わるか、鍵つきの波を抜くといくつ変わるか）。
python intervention/i48ay_role_keys.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
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
import i48an_attr_board as AN  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48ax_sayable as AX  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48x_mesh as XM  # noqa: E402
from i48core import learn as LN  # noqa: E402
from i48core import model as MD  # noqa: E402

DK = 2048
CACHE = 20000


class RoleKey:
    """差し込み: 役の位相鍵で束ねた間の波の板。"""

    def __init__(self, S, nc, dk=DK, seed=0):
        self.S, self.nc, self.dk, self.seed = S, nc, dk, seed
        bt = XM.BandTable(dk)
        for r in AN.ROLES:
            bt.add("役:" + r, "役:" + r)                       # 役ごとに別の組（違う組どうしの和が衝突しない）
        for mk in tuple(AE.MARK_NAMES) + ("文末",):
            bt.add("印:" + mk, "印")
        n = np.arange(dk)
        self.key = {name: np.exp(2j * np.pi * f * n / dk).astype(np.complex64) for name, f, _ in bt.rows}
        self.freq = {name: f for name, f, _ in bt.rows}
        self.B = np.zeros((nc, dk), dtype=np.complex64)
        self.n = np.zeros(nc)
        self._codes, self._words = {}, {}

    def clone(self):
        k = RoleKey.__new__(RoleKey)
        k.__dict__.update(self.__dict__)
        k.B, k.n = self.B.copy(), self.n.copy()
        return k

    def refresh(self, m):
        pass

    def code(self, f):
        c = self._codes.get(int(f))
        if c is None:
            r = np.random.default_rng(1000003 * int(f) + 7 + self.seed)
            c = np.exp(2j * np.pi * r.random(self.dk)).astype(np.complex64)
            self._codes[int(f)] = c
        return c

    def word(self, key):
        y = self._words.get(key)
        if y is None:
            f, v = self.S.attrs(*key)
            y = np.zeros(self.dk, dtype=np.complex64)
            for fi, vi in zip(f, v):
                y += vi * self.code(fi)
            if len(self._words) < CACHE:
                self._words[key] = y
        return y

    def gap_wave(self, keys, x):
        """間 x の波 Z（長さ 1）。役の語が無い組は足さない。"""
        rk = AN.role_keys(keys, x["i"])
        W = {r: self.key["役:" + r] * self.word(k) for r, k in rk.items() if k is not None}
        Z = np.zeros(self.dk, dtype=np.complex64)
        if not x["last"]:
            for a, b in (("左", "述語"), ("左", "右"), ("右", "述語")):
                if a in W and b in W:
                    Z += W[a] * W[b]
        if "左" in W:
            for mk in tuple(sorted(x["mk"])) + (("文末",) if x["last"] else ()):
                k = self.key.get("印:" + mk)
                if k is not None:
                    Z += W["左"] * k
        nz = float(np.linalg.norm(Z))
        return Z / nz if nz > 0 else Z

    def fit(self, m, tr):
        for s in tr:
            xs = AE.items(self.S, s, "節")
            if not xs:
                continue
            keys = MD.keys_of(s)
            for x in xs:
                c = m.cid(x["y"])
                self.B[c] += self.gap_wave(keys, x)
                self.n[c] += 1
        self.Ball, self.N = self.B.sum(0), max(1.0, self.n.sum())
        return self

    def raw(self, Z):
        """Δ_c と床 t。"""
        live = self.n > 0
        m_all = float(np.real(np.vdot(self.Ball, Z))) / self.N
        mc = np.zeros(self.nc)
        mc[live] = np.real(np.conj(self.B[live]) @ Z) / self.n[live]
        d = np.where(live, mc - m_all, 0.0)
        dl = d[live]
        sig = 1.4826 * float(np.median(np.abs(dl - np.median(dl)))) if live.sum() > 1 else 0.0
        t = sig * np.sqrt(2 * np.log(max(2, int(live.sum()))))
        return d, t

    def score(self, m, keys, x, f, v):
        d, t = self.raw(self.gap_wave(keys, x))
        out = np.zeros(self.nc, dtype=np.float32)
        if t <= 0:
            return out
        up = d > t
        out[up] = np.log(d[up] / t)
        if not G.other_ok(m):
            out[-1] = 0
        return out


def picks(m, te):
    """試しの間ごとの選び（文ごとに頭から）。"""
    out = []
    for s in te:
        xs = AE.items(m.S, s, "節")
        if xs:
            rs, _ = m.walk(MD.keys_of(s), xs)
            out += [r["選び"] for r in rs]
    return np.array(out)


def without_r1(m):
    """R1 を 0 にした写し（全体の波を抜く）。"""
    mm = m.clone()
    zero = np.zeros(m.nc, dtype=np.float32)
    mm.who.score = lambda Xi: zero
    return mm


def unexpected(m, base_picks, te, rk, n_ex=20):
    """想定外の反応: 鍵の点が選びを替え、その選びの論理の筋（R2 ＋ R3）が、その間の候補の中央値以下だった物。"""
    k, cnt, ex, changed = 0, Counter(), [], 0
    n_hit = n_miss = 0
    for s in te:
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        keys = MD.keys_of(s)
        gaps = G.split(s)[1]
        rs, fic = m.walk(keys, xs)
        ups = {u for _, u in keys}
        placed = []
        for x, r in zip(xs, rs):
            c = r["選び"]
            if c != base_picks[k]:
                changed += 1
                g = m.gap_scores(keys, x, fic)
                logic = g["R2"] + m.gram.score(m.gram.state(ups, placed, x["last"]))
                ks = rk.score(m, keys, x, None, None)
                if ks[c] > 0 and logic[c] <= np.median(logic[:-1]):
                    ok = m.pg.cls[c] == gaps[x["i"]]
                    n_hit += ok
                    n_miss += not ok
                    cnt[(x["l"], m.pg.cls[c])] += 1
                    if len(ex) < n_ex:
                        ex.append(dict(要点="・".join(kk[0] for kk in keys), 間=x["i"], 元=gaps[x["i"]], 選び=m.pg.cls[c], 鍵の点=round(float(ks[c]), 3)))
            placed.append(m.pg.cls[c])
            k += 1
    rep = [dict(左の語=a, 詰め物=b, 回数=v) for (a, b), v in cnt.most_common(10) if v >= 2]
    return dict(鍵で替わった選び=changed, 想定外=n_hit + n_miss, 当たり=n_hit, 外れ=n_miss, くり返す組=rep, 例=ex)


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
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    res = dict(学びの文=len(tr), 試しの文=len(te), 物差しの文=len(ref), DK=DK, 型={})
    base = MD.Model(S, fill, a.r1).fit(tr, hold)                # 本体を先に学ぶ（物差しの文を先に読むと、属性の周波数の並びが変わる）
    print(f"学び {time.time() - t0:.0f} 秒", flush=True)
    for s in te:                                                # 試しの語と例の語の属性を、いつもの測りと同じ順に先に置く
        AE.items(S, s, "節")
    import i48core.measure as ME
    for _, _, keys, _ in ME.WANT:
        MD.made_items(S, keys, None)
    fill_ref = AD.Fill(S)
    for s in ref:
        fill_ref.see(s)
    say, nat = AX.Sayable(S, ref), RC.Natural(fill_ref, ref)

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
        return r

    report("本体（同じ重さ）", base)
    p0 = picks(base, te)
    res["波が効いているか"] = dict(R1を抜くと替わる選び=int((picks(without_r1(base), te) != p0).sum()), 選びの数=len(p0))
    save()
    t1 = time.time()
    rk = RoleKey(S, base.nc).fit(base, tr)
    res["鍵"] = dict(周波数=rk.freq, 学び秒=round(time.time() - t1))
    print("鍵の学び", res["鍵"]["学び秒"], "秒", flush=True)
    m = base.clone()
    m.extras.append(rk)
    m.refresh()
    report("本体 ＋ 役の位相鍵", m)
    p1 = picks(m, te)
    res["波が効いているか"].update(鍵を抜くと替わる選び=int((p1 != p0).sum()),
                              鍵つきで_R1を抜くと替わる選び=int((picks(without_r1(m), te) != p1).sum()))
    res["想定外の反応"] = unexpected(m, p0, te, rk)
    res["秒"] = round(time.time() - t0)
    print("波", json.dumps(res["波が効いているか"], ensure_ascii=False), "想定外", res["想定外の反応"]["想定外"], flush=True)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
