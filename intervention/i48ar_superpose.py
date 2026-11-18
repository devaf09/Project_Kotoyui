"""I48ar: 介入の単位をずらして重ねた 1 枚を照らす（持ち主の方針: 介入の板を融合して、共振の存在を見つける。ずらして重ねて、光を当てると何枚目かが分かる）。

- 単位は I48ap と同じ「教えた文 × 詰め物」。単位 k の板は、回数 1 の sep_ratio の形の比の対数（P: 並びの帯、Q: 役ごとの属性の帯）。
- 重ねる: 詰め物 c ごとに、その詰め物の単位（n_c 枚、番号 k = 0..n_c−1）に位相 e^{iθ_jk} を掛けて足し、complex64 の 1 枚にする。
    θ_jk = 2π j k / n_c（j = 1..J）。位相の板を J 枚（j ごとに 1 枚）持つ。J が n_c の倍数なら、離散フーリエ変換の逆で全部の単位の点がそのまま戻る（それ以外は近い値）。手で置く数は J だけ。
  本体には融合しない。単位ごとの記録を残し、取り消しは同じ位相で引く（remove）。
- 照らす: 試す間の波（P の特徴・役ごとの属性、和 1）と、重ねた板の内積 → 詰め物ごとに J 個の複素数 z_j。
    存在: |z_1|（大きさ）、何枚目か: 復号 d_k = Re(Σ_j z_j e^{−iθ_jk}) / J のいちばん大きい単位。
  比べ（I48ap の (B)）: 単位を 1 枚ずつ照らした点 s_k（P ＋ 鳴った役の Q の平均）の最大。
- 測る物:
  (1) 存在: max_k s_k と |z_1|・max_k d_k の相関、関連あり（max s > 0）と関連なしの |z_1| の平均。
  (2) 何枚目か: 復号の一番が、1 枚ずつの一番と同じ割合（上位 1）、1 枚ずつの一番が復号の上位 3 に入る割合。単位が 2 枚以上の詰め物だけ。
  (3) 当てた単位の点（その単位の s）を使った時の、間・文末・文、助詞ごとの当たりと害、「と」の順位（I48ao の数え）。
  (4) 重ねる枚数: 自作 70 文の単位に、学びに使っていない文から機械的に作った単位を 1,000・1 万と足して (1)(2) と照らす時間（1 枚ずつとの比べ）。
2 つの重ね方を比べる:
  (ア) 巡回ずらし: 単位の板を拡散の符号に載せ（帯ごとに決まった乱数の位相を掛けて FFT、ユニタリ）、k 枚目を k × 幅（幅 = D // n_c）巡回でずらして重ねる。
       照らす時は、試す間の波を同じ符号に載せ、FFT で全部のずらし幅の相関を一度に取る（逆 FFT 1 回）。山の高さ = 存在、山の位置 = 何枚目。
       にじみの床: 単位の置き場でない所の相関の大きさの中央値。
  (イ) 位相（上の通り、J 枚）。
  2 枚以上が同時に鳴る組（1 枚ずつの点 s > 0 の単位が 2 つ以上）では、復号の上位 2 が 1 枚ずつの上位 2 と同じ集まりか。
手で置く物: J の並び（1・2・3・8・16）、足す単位の数（1,000・1 万）、(4) を測る試しの間の数（1,000）。
python intervention/i48ar_superpose.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
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
import i48ak_teach as AK  # noqa: E402
import i48an_attr_board as AN  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
import i48ap_units as AP  # noqa: E402

ROLES = AN.ROLES
JS = (1, 2, 3, 8, 16)
EXTRA = tuple(int(x) for x in os.environ.get("AR_EXTRA", "1000,10000").split(","))
GAPS4 = int(os.environ.get("AR_GAPS4", "1000"))


class Stack:
    """単位を詰め物ごとに位相で重ねた板（complex64）と、1 枚ずつの板（比べ用、float32）。"""

    def __init__(self, m, units, J):
        self.m, self.J, self.D = m, J, m.S.D
        self.by_c = {}
        for u in units:
            if not u["取り消し"]:
                self.by_c.setdefault(u["c"], []).append(u)
        self.U, self.H, self.E = {}, {}, {}
        for c, us in self.by_c.items():
            n = len(us)
            U = {nm: np.zeros((n, self.D), dtype=np.float32) for nm in ("P",) + ROLES}
            for k, u in enumerate(us):
                pf, _ = u["P"]
                np.add.at(U["P"][k], pf, u["LP"])
                for r, (qf, _) in u["Q"].items():
                    np.add.at(U[r][k], qf, u["LQ"][r])
            j = np.arange(1, J + 1)
            E = np.exp(2j * np.pi * np.outer(np.arange(n), j) / n).astype(np.complex64)    # (n, J)
            self.U[c], self.E[c] = U, E
            self.H[c] = {nm: (E.T @ U[nm].astype(np.complex64)) for nm in U}               # (J, D)

    def remove(self, c, k):
        """単位 k を同じ位相で引く（取り消し）。"""
        for nm, U in self.U[c].items():
            self.H[c][nm] -= np.outer(self.E[c][k], U[k]).astype(np.complex64)
            U[k] = 0

    def light(self, xP, xQ):
        """重ねた板を照らす: 詰め物ごとの z（J 個）と、復号 d（n_c 個）。"""
        out = {}
        for c, H in self.H.items():
            z = H["P"] @ xP
            for r, x in xQ.items():
                z = z + H[r] @ x
            d = np.real(np.conj(self.E[c]) @ z) / self.J
            out[c] = (z, d, None)
        return out

    def exact(self, xP, xQ):
        """1 枚ずつ照らす（I48ap の (B) の点 s_k）。"""
        out = {}
        for c, U in self.U.items():
            sP = U["P"] @ xP
            qs = np.stack([U[r] @ x for r, x in xQ.items()]) if xQ else np.zeros((1, len(sP)), dtype=np.float32)
            ring = (qs != 0).sum(0)
            out[c] = sP + np.where(ring > 0, qs.sum(0) / np.maximum(1, ring), 0)
        return out


class Shift:
    """(ア) 巡回ずらし。符号 C(u) = FFT(位相の板 ⊙ u) / √D（ユニタリ）。H_c = Σ_k roll(C(U_k), k × 幅)。"""

    def __init__(self, m, units, J=None):
        self.m, self.D = m, m.S.D
        rng = np.random.default_rng(20261118)
        self.plate = {nm: np.exp(2j * np.pi * rng.random(self.D)).astype(np.complex64) for nm in ("P",) + ROLES}
        base = Stack(m, units, 1)
        self.U, self.by_c = base.U, base.by_c
        self.FH, self.w = {}, {}
        for c, U in self.U.items():
            n = len(next(iter(U.values())))
            w = max(1, self.D // n)
            self.w[c] = w
            fh = {}
            for nm, M in U.items():
                H = np.zeros(self.D, dtype=np.complex64)
                for k in range(n):
                    H += np.roll(self.code(nm, M[k]), (k * w) % self.D)
                fh[nm] = np.fft.fft(H).astype(np.complex64)
            self.FH[c] = fh

    def code(self, nm, u):
        return (np.fft.fft(self.plate[nm] * u) / np.sqrt(self.D)).astype(np.complex64)

    def exact(self, xP, xQ):
        return Stack.exact(self, xP, xQ)

    def light(self, xP, xQ):
        X = {"P": np.conj(np.fft.fft(self.code("P", xP)))}
        for r, x in xQ.items():
            X[r] = np.conj(np.fft.fft(self.code(r, x)))
        out = {}
        for c, fh in self.FH.items():
            F = fh["P"] * X["P"]
            for r in xQ:
                F = F + fh[r] * X[r]
            corr = np.fft.ifft(F)                             # corr[t] = Σ_n H[n + t] conj(C(x)[n])
            n = len(next(iter(self.U[c].values())))
            pos = (np.arange(n) * self.w[c]) % self.D
            d = np.real(corr[pos])
            mask = np.ones(self.D, dtype=bool)
            mask[pos] = False
            floor = float(np.median(np.abs(corr[mask]))) if mask.any() else 0.0
            out[c] = (np.array([np.abs(corr).max()]), d, floor)
        return out


def gap_inputs(m, keys, x):
    D = m.S.D
    f, v = m.feat(x)
    v = np.asarray(v, dtype=np.float32)
    xP = np.zeros(D, dtype=np.float32)
    np.add.at(xP, f, v / max(1e-9, float(v.sum())))
    xQ = {}
    for r, key in AN.role_keys(keys, x["i"]).items():
        if key is None:
            continue
        qf, qv = m.S.attrs(*key)
        qv = np.asarray(qv, dtype=np.float32)
        xq = np.zeros(D, dtype=np.float32)
        np.add.at(xq, qf, qv / max(1e-9, float(qv.sum())))
        xQ[r] = xq
    return xP, xQ


def compare(st, gaps_in):
    """(1)(2) と時間。"""
    a_max, a_z, a_d, t1, t3, n_id, floors, m2, m2ok = [], [], [], 0, 0, 0, [], 0, 0
    tl = te = 0.0
    for xP, xQ in gaps_in:
        t0 = time.perf_counter()
        L = st.light(xP, xQ)
        tl += time.perf_counter() - t0
        t0 = time.perf_counter()
        X = st.exact(xP, xQ)
        te += time.perf_counter() - t0
        for c, s in X.items():
            z, d, fl = L[c]
            a_max.append(float(s.max()))
            a_z.append(float(abs(z[0])))
            a_d.append(float(d.max()))
            if fl is not None:
                floors.append(fl)
            if (s > 0).sum() >= 2:                            # 2 枚以上が同時に鳴る
                m2 += 1
                m2ok += int(set(np.argsort(-s)[:2]) == set(np.argsort(-d)[:2]))
            if len(s) >= 2:
                n_id += 1
                best = int(np.argmax(s))
                o = np.argsort(-d)
                t1 += int(o[0] == best)
                t3 += int(best in o[:3])
    a_max, a_z, a_d = map(np.array, (a_max, a_z, a_d))
    cor = lambda a, b: round(float(np.corrcoef(a, b)[0, 1]), 3) if a.std() > 0 and b.std() > 0 else None
    rel = a_max > 0
    n = max(1, len(gaps_in))
    return dict(組=len(a_max), 存在_相関_z1=cor(a_max, a_z), 存在_相関_復号の最大=cor(a_max, a_d),
                z1_関連あり=round(float(a_z[rel].mean()), 4) if rel.any() else None, z1_関連なし=round(float(a_z[~rel].mean()), 4) if (~rel).any() else None,
                何枚目_上位1=round(t1 / max(1, n_id), 3), 何枚目_上位3=round(t3 / max(1, n_id), 3), 見分けた組=n_id,
                照らす秒_重ね=round(tl / n, 5), 照らす秒_1枚ずつ=round(te / n, 5),
                二枚以上鳴る組=m2, 二枚以上_上位2が同じ=round(m2ok / max(1, m2), 3), にじみの床=round(float(np.median(floors)), 5) if floors else None)


class RModel(AP.UModel):
    """当てた単位の点で選ぶ（(3)）。"""

    def gap_scores(self, keys, x, fic):
        g = AK.Model.gap_scores(self, keys, x, fic)
        st = getattr(self, "stack", None)
        if st is None:
            return g
        xP, xQ = gap_inputs(self, keys, x)
        L, X = st.light(xP, xQ), st.exact(xP, xQ)
        add = np.zeros(self.nc, dtype=np.float32)
        for c, (z, d, _) in L.items():
            add[c] = X[c][int(np.argmax(d))]
        g["今"] = g["今"] + add
        return g


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
    m = RModel(S, fill, a.r1).fit(tr, hold).init_units("B")
    for it in AO.TEACH_ALL:
        m.teach_unit(it)
    m.refresh()
    res = dict(学びの文=len(tr), 試しの文=len(te), 単位_自作=len(m.units), JS=JS, 選び={}, 重ね={})
    out = a.out

    def save():
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"学び {time.time() - t0:.0f} 秒、単位 {len(m.units)}", flush=True)
    # (3) 当てた単位の点で選ぶ（自作 70 文の単位）
    for nm, mk in [("ア 巡回ずらし", lambda: Shift(m, m.units))] + [(f"イ 位相 J {J}", (lambda J=J: Stack(m, m.units, J))) for J in JS]:
        t1 = time.time()
        m.stack = mk()
        res["選び"][nm] = AP.measure(m, te, surf)
        print("選び", nm, round(time.time() - t1), "秒", json.dumps(res["選び"][nm], ensure_ascii=False), flush=True)
        save()
    m.stack = None
    # (1)(2)(4) 試しの間（先頭 GAPS4）で、重ねる枚数を増やす
    gaps_in = []
    for s in te:
        keys = AK.keys_of(s)
        for x in AE.items(S, s, "節"):
            gaps_in.append(gap_inputs(m, keys, x))
            if len(gaps_in) >= GAPS4:
                break
        if len(gaps_in) >= GAPS4:
            break
    src = iter(extra_src)
    n_made = len(m.units)
    for target in (0,) + EXTRA:
        while len(m.units) < n_made + target:
            s = next(src)
            keys = AK.keys_of(s)
            gaps = G.split(s)[1]
            if keys:
                m.teach_unit(("機械", s["id"], keys, gaps))
        m.refresh()
        n_units = len([u for u in m.units if not u["取り消し"]])
        res["重ね"][str(n_units)] = {}
        for nm, mk in [("ア 巡回ずらし", lambda: Shift(m, m.units))] + [(f"イ 位相 J {J}", (lambda J=J: Stack(m, m.units, J))) for J in JS]:
            t1 = time.time()
            st = mk()
            res["重ね"][str(n_units)][nm] = r = compare(st, gaps_in)
            print("重ね", n_units, nm, round(time.time() - t1), "秒", json.dumps(r, ensure_ascii=False), flush=True)
            save()
            del st
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
