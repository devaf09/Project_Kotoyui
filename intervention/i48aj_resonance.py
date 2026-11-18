"""I48aj: 候補の出し方と選び方に「共振の度合い」を入れる（持ち主の方針: 出力の高さだけでなく共振の度合いも。語板の側にも共振する物を）。

今の点（I48ag の「先読みなし」）: 節・シフト＋元のグレインの数え（対数の比）＋ prior（証拠の数えの段ごとに強さを決める）。

三つの共振（どれも「いつもよりどれだけ強く鳴るか」の対数の比。prior は入れない）:
  R1 全体の波 : 要点全部を拡散の置き方（i48ab_place の「拡散」、位置は間からの相対のずれ）で重ねた文の波。詰め物ごとに学びの文の波を足して持ち、
                近さ（Re の内積）の平均を、全部の詰め物の平均で割った比。板ごとにならす（数で割る）。
  R2 語板の成分: 内容語の属性の帯に「自分の右に来る詰め物」「自分の左に来る詰め物」を数えで持たせる（語そのものではなく属性なので、知らない語に広がる）。
                候補ごとに、左の語の右向き成分・右の語の左向き成分・右のいちばん近い述語の左向き成分の対数の比（PMI に近い形）の平均。
  R3 文法グレイン: 文の中に何が既にあるか（要点の品詞・既に置いた詰め物の中の機能語・間が文末か途中か）を帯に載せた小さなグレイン。
                既に置いた詰め物に左右されるので頭から順に置く（先読みなし、または幅 BEAM の先読み）。学びは元の詰め物で置いた状態。
  入力の波は和が 1 になるようにならすので、どれも「1 つの帯あたりの平均の対数の比」= 何倍強く鳴るか、の対数。合わせ方は対数の和（掛け算）で、重みは置かない。
候補の出し方の比べ（I48aj の (a)(b)(c)）:
  (a) 共振だけ: prior 0、板ごとの波の向きの近さ（コサイン）
  (b) 述語の側から: R2 のうち、右のいちばん近い述語の左向き成分
  (c) 候補 = 今の点・(a)・(b) の上位 K の和集合。選ぶ点は 順位の和（小さいほど良い）、または 3 つの点の積（log 今の点の確率 + log コサイン + (b)）
測る物: 候補の上位 3・5 に元の詰め物が入る割合（R3 は元の詰め物で置いた状態で）、間・文末・文の一致、語の表、「の」を選ぶ割合、持ち主の例 2 つの順位。
板のならし: p(d|詰め物) = (数え + β p(d)) / (回数 + β)（R1 は近さの平均を同じ形で全体の平均へ寄せる）。数の少ない板ほど比 1 に寄る。
  β は 1〜10 万回分（10 倍おき）から、成分ごとに取りおきの文（400 文）で選ぶ（その成分だけの上位 3 に元が入る割合）。
手で置く物: BEAM（10）、K（5）、β の候補の並び、R1 の比の下限 R1_FLOOR（1e-3）、ALPHA（i48ab と同じ 0.1、p(d) のならし）。
python intervention/i48aj_resonance.py --conllu ... --content ... --func ... --out out.json [--train N] [--test N] [--D 8192]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ab_place as PL  # noqa: E402
import i48ac_center_grain as CG  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402

MODE = "シフト＋元"
BEAM = 10
K = 5
BETAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)                     # 板のならしの強さ（数え β 回分）の候補。取りおきの文で、成分ごとに選ぶ
R1_FLOOR = 1e-3
R1_VARS = ("基準", "減衰・1/距離", "減衰・指数", "段1", "段2", "段3", "注視・強め", "注視・網", "注視・寄せ", "架空・前", "架空・後", "架空・下地")
PRED = ("VERB", "ADJ")
FUNC_UPOS = ("ADP", "AUX", "SCONJ", "PART", "CCONJ")
EXAMPLES = ((("昨日", "NOUN"), ("友達", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")), 1, "と"), \
           ((("昨日", "NOUN"), ("公園", "NOUN"), ("犬", "NOUN"), ("散歩", "NOUN"), ("する", "VERB")), 1, "で")


def next_pred(cont, i):
    """間 i の右のいちばん近い述語 (原形, 品詞)。無ければ None。cont は (原形, 表記, 品詞) または (原形, 品詞)。"""
    for c in cont[i + 1:]:
        if c[-1] in PRED:
            return c[0], c[-1]
    return None


class Ratio:
    """数えの板（詰め物 × 帯）。点 = Σ x_d log(p(d|詰め物) / p(d))、x は和 1 にならした入力。「その他」は言わない。"""

    def __init__(self, n_cls, D):
        self.W = np.zeros((n_cls, D), dtype=np.float64)

    def add(self, c, f, v):
        v = np.asarray(v, dtype=np.float64)
        np.add.at(self.W[c], f, v / max(1e-9, float(v.sum())))   # 1 回 = 重さ 1（板の数え = 出た回数）

    def build(self, beta=BETAS[0]):
        """p(d|詰め物) = (W + β p(d)) / (回数 + β)。数の少ない板ほど全体（比 1）へ寄る。"""
        D = self.W.shape[1]
        allw = self.W.sum(0)
        pd = (allw + G.ALPHA) / (allw.sum() + G.ALPHA * D)
        tot = self.W.sum(1, keepdims=True)
        self.beta = beta
        self.L = (np.log((self.W + beta * pd) / (tot + beta)) - np.log(pd)).astype(np.float32)
        if not G.OTHER:
            self.L[-1] = -1e9
        return self

    def score(self, f, v):
        v = np.asarray(v, dtype=np.float32)
        return self.L[:, f] @ (v / max(1e-9, float(v.sum())))


class Whole:
    """R1: 文の波（拡散）と、詰め物ごとの学びの文の波の和の近さ。"""

    def __init__(self, S, n_cls, var="基準"):
        assert var in R1_VARS, var
        self.var, self.fict, self.cur = var, None, ()
        self.S, self.pl = S, PL.Placer(S, "拡散")
        self.B = np.zeros((n_cls, S.D), dtype=np.complex64)
        self.n = np.zeros(n_cls)
        self._w = {}

    def word(self, l, u):
        y = self._w.get((l, u))
        if y is None:
            y = self.pl.word(*self.S.attrs(l, u))
            if len(self._w) < 2000:
                self._w[(l, u)] = y
        return y

    def sent(self, keys):
        """keys: [(原形, 品詞)]。語 k を k × SHIFT 巡回でずらして重ねる（間 i から見る時は −i × SHIFT 戻す）。"""
        X = np.zeros(self.S.D, dtype=np.complex64)
        for k, (l, u) in enumerate(keys):
            X += np.roll(self.word(l, u), k * self.S.shift)
        return X

    def at(self, X, i):
        return np.roll(X, -i * self.S.shift)

    # ---- R1 の型（持ち主の案: 距離の減衰・遠くの 1 つを注視・架空の行） ----
    def weight(self, d):
        v = self.var
        if v == "減衰・1/距離":
            return 1.0 / d
        if v == "減衰・指数":
            return math.exp(-(d - 1))
        if v.startswith("段"):
            return 1.0 if d <= int(v[1]) else 0.0
        return 1.0

    def attend(self, keys, i):
        """焦点の左の語の属性と、離れた語（距離 2 以上）それぞれの属性の重なり。いちばん鳴った 1 語の番号（無ければ None）。"""
        q = self.S.dense(*self.S.attrs(*keys[i]))
        best = None
        for k, key in enumerate(keys):
            d = i - k + 1 if k <= i else k - i
            if d < 2:
                continue
            o = float(q @ self.S.dense(*self.S.attrs(*key)))
            if best is None or o > best[0]:
                best = (o, k)
        return best[1] if best else None

    def gap(self, keys, i, fict=()):
        """間 i から見た文の波（型ごと）。語 k は (k − i) × SHIFT の巡回のずれ。"""
        v, sh = self.var, self.S.shift
        fict = list(fict or ())
        if v == "架空・前":
            keys2, i2 = fict + list(keys), i + len(fict)
        elif v == "架空・後":
            keys2, i2 = list(keys) + fict, i
        else:
            keys2, i2 = keys, i
        X = np.zeros(self.S.D, dtype=np.complex64)
        for k, (l, u) in enumerate(keys2):
            d = i2 - k + 1 if k <= i2 else k - i2
            w = self.weight(d)
            if w:
                X += w * np.roll(self.word(l, u), (k - i2) * sh)
        if v.startswith("注視"):
            k = self.attend(keys, i)
            if k is not None:
                y = self.word(*keys[k])
                if v == "注視・強め":                          # その語の波をもう 1 回分足す（重さ 2）
                    X += np.roll(y, (k - i) * sh)
                elif v == "注視・網":                          # その語を中心にした ±1 の窓の波を、別の帯（mix のずれ）に足す
                    for j in range(max(0, k - 1), min(len(keys), k + 2)):
                        X += np.roll(self.word(*keys[j]), (j - k) * sh + self.S.mix)
                elif v == "注視・寄せ":                        # その語の波を焦点の左隣（左の語と同じ位置）に置く
                    X += y
        if v == "架空・下地" and fict:                        # 架空の行で先に作った波を、別の帯（mix のずれ）に下地として置く
            X += np.roll(self.sent(fict), self.S.mix)
        return X

    def bag(self, keys):
        y = np.zeros(self.S.D, dtype=np.float32)
        for key in keys:
            y += self.S.dense(*self.S.attrs(*key))
        return y @ self.P

    def index(self, all_keys):
        """架空の行の引き先（学びの文）。文の属性の袋を乱数の射影（256 次）で縮め、コサインでいちばん近い文を引く。学びの文どうしは自分を除く。"""
        if not self.var.startswith("架空"):
            return
        self.P = np.random.default_rng(12345).standard_normal((self.S.D, 256)).astype(np.float32)
        self.keys = all_keys
        M = np.stack([self.bag(k) for k in all_keys])
        self.M = M / np.maximum(1e-9, np.linalg.norm(M, axis=1, keepdims=True))
        self.fict = []
        bs = max(50, min(1000, int(2e7 // max(1, len(M)))))   # 似た度の表を 2,000 万要素ずつ（全学習で数 GB にしない。答えは変わらない）
        for b in range(0, len(M), bs):
            sim = self.M[b:b + bs] @ self.M.T
            sim[np.arange(len(sim)), np.arange(b, b + len(sim))] = -9
            self.fict.extend(int(j) for j in np.argmax(sim, axis=1))

    def fiction(self, keys, self_idx=None):
        if not self.var.startswith("架空"):
            return ()
        if self_idx is not None:
            return self.keys[self.fict[self_idx]]
        q = self.bag(keys)
        q /= max(1e-9, float(np.linalg.norm(q)))
        return self.keys[int(np.argmax(self.M @ q))]

    def add(self, c, Xi):
        self.B[c] += Xi
        self.n[c] += 1

    def build(self, beta=BETAS[0]):
        self.Ball = self.B.sum(0)
        self.N = self.n.sum()
        self.beta = beta
        return self

    def score(self, Xi):
        m_all = float(np.real(np.vdot(self.Ball, Xi))) / self.N
        m = (np.real(self.B.conj() @ Xi) + self.beta * m_all) / (self.n + self.beta)
        r = np.log(np.maximum(R1_FLOOR, m / m_all)) if m_all > 0 else np.zeros(len(m))
        if not G.OTHER:
            r[-1] = -1e9
        return r.astype(np.float32)


class Gram:
    """R3: 文の中に既にある物の小さなグレイン。"""

    def __init__(self, n_cls, fill):
        self.fill, self.idx, self.n_cls = fill, {}, n_cls
        self.rows = []

    def state(self, upos_set, placed, last):
        names = {"品詞:" + u for u in upos_set}
        for g in placed:
            for t in self.fill.toks.get(g, []):
                if t["upos"] in FUNC_UPOS:
                    names.add("置:" + t["lemma"])
        names.add("間:文末" if last else "間:途中")
        return sorted(names)

    def f(self, names, grow=False):
        if grow:
            for n in names:
                self.idx.setdefault(n, len(self.idx))
        return np.array([self.idx[n] for n in names if n in self.idx], dtype=np.int64)

    def add(self, c, names):
        self.rows.append((c, self.f(names, grow=True)))

    def build(self, beta=BETAS[0]):
        if self.rows:
            self.R = Ratio(self.n_cls, max(1, len(self.idx)))
            for c, f in self.rows:
                self.R.add(c, f, np.ones(len(f)))
            self.rows = []
        self.R.build(beta)
        return self

    def score(self, names):
        f = self.f(names)
        return self.R.score(f, np.ones(len(f))) if len(f) else np.zeros(self.n_cls, dtype=np.float32)


def ranks(s):
    """点の配列 → 順位（0 が最上）。"""
    r = np.empty(len(s), dtype=np.int64)
    r[np.argsort(-s, kind="stable")] = np.arange(len(s))
    return r


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="基準", choices=R1_VARS)
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold = tr_all[-2000:]
    tr = tr_all[:-2000][: a.train]
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
    TRs = [(s, AE.items(S, s, "節")) for s in tr]
    TR = [x for _, xs in TRs for x in xs]
    cnt = Counter(x["y"] for x in TR)
    pair_n = Counter((x["l"], x["r"]) for x in TR if not x["last"])
    feat = lambda x: AE.feats(S, x, MODE)
    pg = G.PairGrain(a.D).fit(((*feat(x), x["y"]) for x in TR), cnt)
    nc = len(pg.cls)
    ci = {y: i for i, y in enumerate(pg.cls) if y is not None}
    cid = lambda y: ci.get(y, nc - 1)
    print(f"読み込みと今のグレイン {time.time() - t0:.0f} 秒、学び {len(tr)} 文、間 {len(TR)}", flush=True)
    # prior の強さ（証拠の段ごと、取りおきで決める。I48ag と同じ）
    HO = [x for s in hold for x in AE.items(S, s, "節")]
    best_w = {}
    for b in range(len(AD.EV_BINS)):
        hx = [x for x in HO if AD.ev_bin(pair_n[(x["l"], x["r"])] if not x["last"] else 10) == b][:1500]
        F = [feat(x) for x in hx]
        best_w[b] = max((sum(pg.cls[int(np.argmax(w * pg.prior + pg.logp[:, f] @ v))] == x["y"] for (f, v), x in zip(F, hx)), w)
                        for w in (0.0, 0.25, 0.5, 1.0))[1] if hx else 1.0
    # 三つの共振の板
    t1 = time.time()
    who = Whole(S, nc, a.r1)
    TRK = [[(l, u) for l, _, u in G.split(s)[0]] for s, _ in TRs]
    who.index(TRK)
    rR, rL, rP = Ratio(nc, a.D), Ratio(nc, a.D), Ratio(nc, a.D)
    gram = Gram(nc, fill)
    nt = CG.NeighborTable()
    for j, (s, xs) in enumerate(TRs):
        cont, gaps, _ = G.split(s)
        keys = TRK[j]
        fic = who.fiction(keys, j) if keys else ()
        ups = {u for _, u in keys}
        for i, g in enumerate(gaps):
            c = cid(g)
            last = i + 1 == len(gaps)
            who.add(c, who.gap(keys, i, fic))
            rR.add(c, *S.attrs(*keys[i]))
            if not last:
                rL.add(c, *S.attrs(*keys[i + 1]))
                p = next_pred(keys, i)
                if p:
                    rP.add(c, *S.attrs(*p))
            gram.add(c, gram.state(ups, gaps[:i], last))
        for x in xs:
            nt.add(x["l"], x["r"] if not x["last"] else ("文末",) + x["mk"], x["y"])
    # ならしの強さ β を、成分ごとに取りおきの文で選ぶ（その成分だけの点の上位 3 に元が入る割合）
    HOs = [(s, AE.items(S, s, "節")) for s in hold[:400]]
    beta = {}
    for nm, obj in (("R1", who), ("R2右向き", rR), ("R2左向き", rL), ("R2述語", rP), ("R3", gram)):
        best = None
        for b in BETAS:
            obj.build(b)
            hit = 0
            for s, xs in HOs:
                cont, gaps, _ = G.split(s)
                keys = [(l, u) for l, _, u in cont]
                if not keys:
                    continue
                fic = who.fiction(keys) if nm == "R1" else ()
                for i, g in enumerate(gaps):
                    last = i + 1 == len(gaps)
                    if nm == "R1":
                        sc = who.score(who.gap(keys, i, fic))
                    elif nm == "R2右向き":
                        sc = rR.score(*S.attrs(*keys[i]))
                    elif nm == "R2左向き":
                        if last:
                            continue
                        sc = rL.score(*S.attrs(*keys[i + 1]))
                    elif nm == "R2述語":
                        p = None if last else next_pred(keys, i)
                        if not p:
                            continue
                        sc = rP.score(*S.attrs(*p))
                    else:
                        sc = gram.score(gram.state({u for _, u in keys}, gaps[:i], last))
                    hit += ranks(sc)[cid(g)] < 3
            if best is None or hit > best[0]:
                best = (hit, b)
        obj.build(best[1])
        beta[nm] = best[1]
    Wn = pg.W / np.maximum(1e-9, np.linalg.norm(pg.W, axis=1, keepdims=True))
    print(f"三つの共振の板 {time.time() - t1:.0f} 秒、拡散の符号 {len(who.pl._codes)}、文法の帯 {len(gram.idx)}、prior {best_w}、β {beta}", flush=True)

    def gap_scores(keys, x, pw):
        """間ごとの、置いた物に左右されない点。"""
        i, last = x["i"], x["last"]
        f, v = feat(x)
        base = AD.posterior(pg, f, v, pw)
        base[-1] = 0
        cos = (Wn[:, f] @ v).astype(np.float32)
        cos[-1] = -1
        r1 = who.score(who.gap(keys, i, who.cur))
        parts = [rR.score(*S.attrs(*keys[i]))]
        pred = None
        if not last:
            parts.append(rL.score(*S.attrs(*keys[i + 1])))
            p = next_pred(keys, i)
            if p:
                pred = rP.score(*S.attrs(*p))
                parts.append(pred)
        r2 = np.mean(parts, axis=0)
        return dict(今=np.log(np.maximum(base, 1e-30)).astype(np.float32), コサイン=cos, 述語=pred, R1=r1, R2=r2)

    COMBOS = {"三つ": ("R1", "R2", "R3"), "今×三つ": ("今", "R1", "R2", "R3"), "三つ−R1": ("R2", "R3"), "三つ−R2": ("R1", "R3"), "三つ−R3": ("R1", "R2")}

    def run_sent(keys, xs, ups, beam):
        """組ごとの、頭から順に置いた並び。R3 を含む組は幅 beam で先読み。"""
        n = len(xs)
        who.cur = who.fiction(keys)
        gs = [gap_scores(keys, x, best_w[AD.ev_bin(pair_n[(x["l"], x["r"])] if not x["last"] else 10)]) for x in xs]
        out = {}
        for nm, parts in COMBOS.items():
            fixed = [sum(g[p] for p in parts if p != "R3") for g in gs]
            bw = beam if "R3" in parts else 1
            bm = [((), 0.0)]
            for i in range(n):
                nb = []
                for seq, sc in bm:
                    s_ = fixed[i] + (gram.score(gram.state(ups, [pg.cls[c] for c in seq], i + 1 == n)) if "R3" in parts else 0)
                    s_[-1] = -1e9
                    for c in np.argsort(-s_)[:bw]:
                        nb.append((seq + (int(c),), sc + float(s_[c])))
                bm = sorted(nb, key=lambda z: -z[1])[:bw]
            out[nm] = [pg.cls[c] for c in bm[0][0]]
        out["今の点"] = [pg.cls[int(np.argmax(g["今"]))] for g in gs]
        out["R1だけ"] = [pg.cls[int(np.argmax(g["R1"]))] for g in gs]
        for nm in ("和集合・順位の和", "和集合・積"):
            seq = []
            for g in gs:
                cand = set(np.argsort(-g["今"])[:K]) | set(np.argsort(-g["コサイン"])[:K]) | (set(np.argsort(-g["述語"])[:K]) if g["述語"] is not None else set())
                cand.discard(nc - 1)
                cand = sorted(cand)
                if nm == "和集合・順位の和":
                    rk = ranks(g["今"]) + ranks(g["コサイン"]) + (ranks(g["述語"]) if g["述語"] is not None else 0)
                    seq.append(pg.cls[min(cand, key=lambda c: (rk[c], -g["今"][c]))])
                else:
                    sc = g["今"] + np.log(np.maximum(g["コサイン"], 1e-9)) + (g["述語"] if g["述語"] is not None else 0)
                    seq.append(pg.cls[max(cand, key=lambda c: sc[c])])
            out[nm] = seq
        return out, gs

    t2 = time.time()
    c = Counter()
    PICKS = ("今の点", "R1だけ", "三つ", "三つ・先読みなし", "今×三つ", "今×三つ・先読みなし", "三つ−R1", "三つ−R2", "三つ−R3", "和集合・順位の和", "和集合・積", "語の表")
    INC = ("今", "コサイン", "述語", "R1", "R2", "R3", "三つ", "今×三つ")
    for s in te:
        xs = AE.items(S, s, "節")
        if not xs:
            continue
        cont, gaps, head = G.split(s)
        keys = [(l, u) for l, _, u in cont]
        ups = {u for _, u in keys}
        picks, gs = run_sent(keys, xs, ups, BEAM)
        p1, _ = run_sent(keys, xs, ups, 1)
        picks["三つ・先読みなし"], picks["今×三つ・先読みなし"] = p1["三つ"], p1["今×三つ"]
        picks["語の表"] = [nt.predict(x["l"], x["r"] if not x["last"] else ("文末",) + x["mk"]) for x in xs]
        c["文"] += 1
        c["間n"] += len(gaps) - 1
        c["元:の"] += sum(g == "の" for g in gaps[:-1])
        for nm in PICKS:
            seq = picks[nm]
            for i, (g, y) in enumerate(zip(seq, gaps)):
                c[nm + (":間" if i + 1 < len(gaps) else ":文末")] += g == y
                c[nm + ":の"] += g == "の" and i + 1 < len(gaps)
            sent = "".join(surf.form(l, u, g) + g for (l, f_, u), g in zip(cont, seq))
            c[nm + ":文"] += G.norm(head + sent) == G.norm(s["text"])
        for i, (g, y) in enumerate(zip(gs, gaps)):                 # 候補に元が入るか（R3 は元の詰め物で置いた状態）
            yc = cid(y)
            r3 = gram.score(gram.state(ups, gaps[:i], i + 1 == len(gaps)))
            sc = dict(g, **{"R3": r3, "三つ": g["R1"] + g["R2"] + r3, "今×三つ": g["今"] + g["R1"] + g["R2"] + r3})
            c["inc_n"] += 1
            for nm in INC:
                if sc[nm] is None:
                    continue
                c[nm + ":n"] += 1
                r = ranks(sc[nm])[yc]
                c[nm + ":3"] += r < 3
                c[nm + ":5"] += r < 5
            for kk in (3, 5):
                u = set(np.argsort(-g["今"])[:kk]) | set(np.argsort(-g["コサイン"])[:kk]) | (set(np.argsort(-g["述語"])[:kk]) if g["述語"] is not None else set())
                u.discard(nc - 1)
                c[f"和集合:{kk}"] += yc in u
                c[f"和集合:{kk}:大きさ"] += len(u)
        if c["文"] % 200 == 0:
            print(f"試し {c['文']} 文 {time.time() - t2:.0f} 秒", flush=True)
    res = dict(R1の型=a.r1, 学びの文=len(tr), 試しの文=c["文"], D=a.D, prior=best_w, β=beta, BEAM=BEAM, K=K, 元の_の=round(c["元:の"] / max(1, c["間n"]), 3),
               選び={}, 候補に元={}, 例=[], 秒=round(time.time() - t0))
    for nm in PICKS:
        res["選び"][nm] = dict(間=round(c[nm + ":間"] / max(1, c["間n"]), 3), 文末=round(c[nm + ":文末"] / max(1, c["文"]), 3),
                              文=round(c[nm + ":文"] / max(1, c["文"]), 3), の=round(c[nm + ":の"] / max(1, c["間n"]), 3))
    for nm in INC:
        res["候補に元"][nm] = dict(上位3=round(c[nm + ":3"] / max(1, c[nm + ":n"]), 3), 上位5=round(c[nm + ":5"] / max(1, c[nm + ":n"]), 3), 間の数=c[nm + ":n"])
    for kk in (3, 5):
        res["候補に元"][f"和集合 {kk}"] = dict(入る=round(c[f"和集合:{kk}"] / max(1, c["inc_n"]), 3), 平均の大きさ=round(c[f"和集合:{kk}:大きさ"] / max(1, c["inc_n"]), 1))
    # 持ち主の例
    for keys, gi, want in EXAMPLES:
        n = len(keys)
        A = [S.attrs(l, u) for l, u in keys]
        xs = [dict(A=A, i=i, mk=("過去",), last=i + 1 == n, l=keys[i][0], r=keys[i + 1][0] if i + 1 < n else None) for i in range(n)]
        ups = {u for _, u in keys}
        picks, gs = run_sent(keys, xs, ups, BEAM)
        g = gs[gi]
        r3 = gram.score(gram.state(ups, picks["三つ"][:gi], False))
        sc = dict(g, **{"R3": r3, "三つ": g["R1"] + g["R2"] + r3, "今×三つ": g["今"] + g["R1"] + g["R2"] + r3})
        wc = cid(want)
        o = dict(要点="・".join(l for l, _ in keys), 間=gi, 架空の行="・".join(l for l, _ in who.cur), 見る詰め物=want, 順位={}, 上位3={},
                 文={nm: "".join(surf.form(l, u, y) + y for (l, u), y in zip(keys, seq)) for nm, seq in picks.items()})
        for nm in INC:
            if sc[nm] is None:
                continue
            o["順位"][nm] = int(ranks(sc[nm])[wc]) + 1 if want in ci else None
            o["上位3"][nm] = [pg.cls[i] for i in np.argsort(-sc[nm])[:3]]
        res["例"].append(o)
        print("例", json.dumps(o, ensure_ascii=False), flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("選び", "候補に元", "元の_の", "秒")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
