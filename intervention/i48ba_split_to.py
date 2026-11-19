"""I48ba: と を向きで分ける（持ち主の案: 文字が同じでも意味合いが違えば、少しでもズレ始めたら分ける。案、未確定）。

本体は書き換えない。学びの文の「と」の間だけを、文から割れた向きごとの別の詰め物（と#0, と#1, …）に付け替えて学び、
選んだ後は「と」に戻して測る。
  割り方（文からの数えだけ）: 本体が学んでいない文の「と」の間ごとに、文脈 = (左の内容語のいちばん重いクラス, 右の内容語のいちばん
    重いクラス)（クラスは語板の D 帯、無ければ品詞）。文脈を 1 つ当たり 1/√2 の 2 か所に立てた形にして、向きの数 k = 1〜5 で
    向きの近さの重心に分ける（球面の k 平均）。床は、左と右の結びつきだけを混ぜて壊した文脈で同じ事をした時の上がり。
    本物の上がりが床の上がりを越える所まで k を増やす（昼の下調べと同じ決め方）。
  付け替え: 学びの文・取りおきの文の「と」の間を、いちばん近い重心の番号で と#j に替える。
比べ: 本体（付け替えなし）と、と を分けた本体。
測る物: 間・文末・文、と の人・人以外、と の向きの群ごとの と の当たり、ほかの助詞の当たりと害、と の例 9 つの選び、言える（物差しは
  学んでいない文）、筋の無い共振、想定外の反応（分けた本体で選びが替わり、その選びの論理の筋が中央値以下の物）。
手で置く物: k の上限 5、k 平均の初めの置き方（5 回）と回す数（30）、乱数の種。
python intervention/i48ba_split_to.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations

import argparse
import copy
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
import i48an_attr_board as AN  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48ax_sayable as AX  # noqa: E402
from i48core import learn as LN  # noqa: E402
from i48core import measure as ME  # noqa: E402
from i48core import model as MD  # noqa: E402

TO = "と"


def kmeans(X, w, k, rng, it=30, reps=5):
    best = None
    for _ in range(reps):
        C = X[rng.choice(len(X), k, replace=False, p=w / w.sum())]
        for _ in range(it):
            Cn = C / np.maximum(1e-9, np.linalg.norm(C, axis=1, keepdims=True))
            a = np.argmax(X @ Cn.T, axis=1)
            C = np.stack([(X[a == j] * w[a == j, None]).sum(0) if (a == j).any() else C[j] for j in range(k)])
        Cn = C / np.maximum(1e-9, np.linalg.norm(C, axis=1, keepdims=True))
        sc = X @ Cn.T
        a = np.argmax(sc, 1)
        coh = float((sc[np.arange(len(X)), a] * w).sum() / w.sum())
        if best is None or coh > best[0]:
            best = (coh, Cn)
    return best


class Splitter:
    def __init__(self, cls_of):
        self.cls_of = cls_of

    def ctx(self, cont, i):
        L = self.cls_of(cont[i][0], cont[i][2])
        R = self.cls_of(cont[i + 1][0], cont[i + 1][2]) if i + 1 < len(cont) else "文末"
        return L, R

    def fit(self, ref, rng, kmax=5):
        rows = []
        for s in ref:
            cont, gaps, _ = G.split(s)
            rows += [self.ctx(cont, i) for i, g in enumerate(gaps) if g == TO]
        Ls, Rs = sorted({l for l, _ in rows}), sorted({r for _, r in rows})
        self.li, self.ri = {l: i for i, l in enumerate(Ls)}, {r: i for i, r in enumerate(Rs)}
        self.dim = len(Ls) + len(Rs)

        def build(rows):
            cnt = Counter(rows)
            keys = list(cnt)
            X = np.stack([self.x(k) for k in keys])
            return X, np.array([cnt[k] for k in keys], dtype=float)
        X, w = build(rows)
        Rp = list(rng.permutation([r for _, r in rows]))
        Xn, wn = build(list(zip([l for l, _ in rows], Rp)))
        res, prev, prevn, K, Cs = [], None, None, 1, {}
        for k in range(1, min(kmax, len(X), len(Xn)) + 1):        # 文脈の種類より多くは分けない
            c, Cn = kmeans(X, w, k, rng)
            cn, _ = kmeans(Xn, wn, k, rng)
            Cs[k] = Cn
            res.append((k, round(c, 4), round(cn, 4)))
            if prev is not None:
                if c - prev > cn - prevn and K == k - 1:
                    K = k
            prev, prevn = c, cn
        self.K, self.C, self.curve, self.n = K, Cs[K], res, len(rows)
        return self

    def x(self, key):
        v = np.zeros(self.dim, dtype=np.float32)
        l, r = key
        if l in self.li:
            v[self.li[l]] = 1 / np.sqrt(2)
        if r in self.ri:
            v[len(self.li) + self.ri[r]] = 1 / np.sqrt(2)
        return v

    def group(self, cont, i):
        return int(np.argmax(self.C @ self.x(self.ctx(cont, i))))

    def relabel(self, s):
        """「と」の間の と の語の表記を と#j に替えた写し（原形は と のまま）。"""
        cont, gaps, _ = G.split(s)
        idx = [i for i, g in enumerate(gaps) if g == TO]
        if not idx:
            return s
        T = s["toks"]
        ci = [j for j, t in enumerate(T) if G.is_content(t)]
        s2 = dict(s, toks=[dict(t) for t in T])
        for i in idx:
            j = ci[i] + 1                                       # 間 i の詰め物は内容語 i の直後の 1 語（と だけの間）
            s2["toks"][j]["form"] = f"{TO}#{self.group(cont, i)}"
        return s2


def to_ranks(m, ids):
    """と の例 9 つ: 分けた と のうち、いちばん上の順位（1 が最上）と選び。"""
    out = {}
    for kind, name, keys, gi in ME.WANT:
        rs, _ = m.walk(keys, MD.made_items(m.S, keys, None))
        sc = rs[gi]["今×三つ"]
        order = np.argsort(-sc)
        rk = {c: int(np.nonzero(order == c)[0][0]) + 1 for c in ids}
        out[name] = dict(順位=min(rk.values()), 選び=m.pg.cls[rs[gi]["選び"]])
    return out


def by_group(m, te, sp):
    """試しの文の と の間を、向きの群ごとに数えた と の当たり。"""
    c = Counter()
    for s in te:
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        cont, gaps, _ = G.split(s)
        rs, _ = m.walk(MD.keys_of(s), xs)
        for i, (r, y) in enumerate(zip(rs, gaps)):
            if y == TO and i + 1 < len(gaps):
                j = sp.group(cont, i)
                c[(j, "数")] += 1
                c[(j, "当たり")] += m.pg.cls[r["選び"]] == TO
    return {str(j): dict(数=c[(j, "数")], 当たり=round(c[(j, "当たり")] / max(1, c[(j, "数")]), 3)) for j in range(sp.K)}


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
    top = {}
    for w in cb:
        D = w["attrs"].get("D", {})
        if D and (w["name"] not in top or w.get("freq", 0) > top[w["name"]][1]):
            top[w["name"]] = (max(D.items(), key=lambda z: z[1])[0], w.get("freq", 0))
    cls_of = lambda l, u: top[l][0] if l in top else "品詞:" + u
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    res = dict(学びの文=len(tr), 試しの文=len(te), 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def build(trs, holds):
        S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
        fill, surf = AD.Fill(S), G.Surface()
        for s in trs + tr:                                     # 元の文も見せて、「と」の語の中身（R3 用）を持たせる
            fill.see(s)
        for s in trs:
            cont, gaps, _ = G.split(s)
            for (l, f_, u), g in zip(cont, gaps):
                surf.add(l, f_, u, g)
        return S, fill, surf, MD.Model(S, fill, a.r1).fit(trs, holds)

    def report(nm, m, S, fill, surf, sp, ids):
        t1 = time.time()
        for s in te:
            AE.items(S, s, "節")
        r = AP.measure(m, te, surf)
        r["と の群ごと"] = by_group(m, te, sp)
        r["と の例"] = to_ranks(m, ids)
        fill_ref = AD.Fill(S)
        for s in ref:
            fill_ref.see(s)
        r["言えるか"] = AX.score(m, te, AX.Sayable(S, ref), RC.Natural(fill_ref, ref))
        r["筋の無い共振"] = LN.monitor(m, te)
        res["型"][nm] = r
        print(nm, round(time.time() - t1), "秒", json.dumps({k: r[k] for k in ("間", "文末", "文", "と_人", "と_人以外", "と の群ごと")}, ensure_ascii=False), flush=True)
        save()

    sp = Splitter(cls_of).fit(ref, np.random.default_rng(0))
    res["割り方"] = dict(k=sp.K, そろい具合_k=sp.curve, と_の間=sp.n)
    print("割り方", json.dumps(res["割り方"], ensure_ascii=False), flush=True)
    S0, fill0, surf0, m0 = build(tr, hold)
    report("本体（分けない）", m0, S0, fill0, surf0, sp, [m0.cid(TO)])
    p0 = []
    for s in te:
        xs = AE.items(S0, s, "節")
        if xs:
            p0 += [m0.pg.cls[r["選び"]] for r in m0.walk(MD.keys_of(s), xs)[0]]
    del m0, S0, fill0, surf0
    trs, holds = [sp.relabel(s) for s in tr], [sp.relabel(s) for s in hold]
    S1, fill1, surf1, m1 = build(trs, holds)
    ids = [i for i, c in enumerate(m1.pg.cls) if c is not None and c.startswith(TO + "#")]
    res["分けた と の詰め物"] = [m1.pg.cls[i] for i in ids]
    for i in ids:
        m1.pg.cls[i] = TO                                       # 選んだ後は「と」に戻して測る
    m1.ci[TO] = ids[0] if ids else m1.ci.get(TO)
    report("と を分けた本体", m1, S1, fill1, surf1, sp, ids)
    n_ch = n_un = n_hit = 0
    k = 0
    ex = []
    for s in te:
        xs = AE.items(S1, s, "節")
        if not xs:
            continue
        keys = MD.keys_of(s)
        gaps = G.split(s)[1]
        rs, fic = m1.walk(keys, xs)
        ups = {u for _, u in keys}
        placed = []
        for x, r in zip(xs, rs):
            c = r["選び"]
            if m1.pg.cls[c] != p0[k]:
                n_ch += 1
                g = m1.gap_scores(keys, x, fic)
                logic = g["R2"] + m1.gram.score(m1.gram.state(ups, placed, x["last"]))
                if logic[c] <= np.median(logic[:-1]):
                    n_un += 1
                    n_hit += m1.pg.cls[c] == gaps[x["i"]]
                    if len(ex) < 15:
                        ex.append(dict(要点="・".join(kk[0] for kk in keys), 間=x["i"], 元=gaps[x["i"]], 本体=p0[k], 分けた=m1.pg.cls[c]))
            placed.append(m1.pg.cls[c])
            k += 1
    res["想定外の反応"] = dict(替わった選び=n_ch, 想定外=n_un, 当たり=n_hit, 外れ=n_un - n_hit, 例=ex)
    res["秒"] = round(time.time() - t0)
    print("想定外", json.dumps(res["想定外の反応"], ensure_ascii=False)[:300], flush=True)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
