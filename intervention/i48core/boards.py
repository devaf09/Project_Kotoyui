"""板: 今の点のグレイン（PairGrain）・語の表（WordTable）・R2 の比の板（Ratio）・R1 の全体の波（Whole）・R3 の文法の板（Gram）。

実装は i48ab_long_grain（PairGrain・WordTable）と i48aj_resonance（Ratio・Whole・Gram・ranks）から移した物。古いファイルは同じ名前で見せる。
設定（ALPHA・OTHER・other_ok）は i48ab_long_grain に置いたまま（試しが G.OTHER を書き換えるので、使う時に読む）。
手で置く物: BETAS（板のならしの強さの候補）、R1_FLOOR（R1 の比の下限）、FUNC_UPOS（R3 で数える機能語の品詞）。
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

import i48ab_long_grain as G
import i48ab_place as PL

BETAS = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)                     # 板のならしの強さ（数え β 回分）の候補。取りおきの文で、成分ごとに選ぶ
R1_FLOOR = 1e-3
R1_VARS = ("基準", "減衰・1/距離", "減衰・指数", "段1", "段2", "段3", "注視・強め", "注視・網", "注視・寄せ", "架空・前", "架空・後", "架空・下地")
FUNC_UPOS = ("ADP", "AUX", "SCONJ", "PART", "CCONJ")


class PairGrain:
    """詰め物の選び方の数え上げ（多項、足すだけ）。クラスは学びの上位 n_cls の詰め物と「その他」。"""

    def __init__(self, D, n_cls=300):
        self.D, self.n_cls = D, n_cls
        self.cls, self.W, self.prior = None, None, None

    def fit(self, rows, counts):
        """rows: (f, v, 詰め物) を順に出す物（一度だけ読む）。counts: 詰め物の数（クラスを決める）。"""
        self.cls = [y for y, _ in counts.most_common(self.n_cls)]
        ci = {y: i for i, y in enumerate(self.cls)}
        other = len(self.cls)
        self.cls.append(None)
        self.W = np.zeros((len(self.cls), self.D), dtype=np.float64)
        pr = np.zeros(len(self.cls))
        for f, v, y in rows:
            i = ci.get(y, other)
            np.add.at(self.W[i], f, v)
            pr[i] += 1
        self.prior = np.log((pr + 1) / (pr.sum() + len(pr)))
        tot = self.W.sum(1, keepdims=True)
        self.logp = np.log((self.W + G.ALPHA) / (tot + G.ALPHA * self.D))
        if not G.other_ok(self):
            self.logp[other] = -1e9                         # 「その他」は言わない
        return self

    def predict(self, f, v, k=1, prior_w=1.0):
        s = prior_w * self.prior + self.logp[:, f] @ v
        o = np.argsort(-s)[:k]
        return [self.cls[i] for i in o]


class WordTable:
    def __init__(self):
        self.t2, self.t1, self.t0 = defaultdict(Counter), defaultdict(Counter), Counter()

    def fit(self, rows):
        """rows: [(原形 i, 原形 i+1 または印の組, 詰め物)]"""
        for a, b, y in rows:
            self.t2[(a, b)][y] += 1
            self.t1[a][y] += 1
            self.t0[y] += 1
        return self

    def predict(self, a, b):
        for c in (self.t2.get((a, b)), self.t1.get(a)):
            if c:
                return c.most_common(1)[0][0]
        return self.t0.most_common(1)[0][0]


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
        if not G.other_ok(self):
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
        if not G.other_ok(self):
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


def __getattr__(name):
    if name in ("sep_ratio",):
        import importlib
        return getattr(importlib.import_module("i48ak_teach"), name)
    if name == "ALPHA":
        return G.ALPHA
    raise AttributeError(name)
