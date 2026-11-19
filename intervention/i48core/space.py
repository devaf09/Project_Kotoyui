"""語板の属性の空間: 属性 → 周波数、語 → (周波数の配列, 重みの配列)、入口の波と逆回しの特徴。

実装は i48ab_long_grain から移した物。
手で置く物: TOPK（帯ごとに使う属性の数）、STEP・SHIFT・MIX の歩幅（D と互いに素の奇数。黄金比などから）。
"""
from __future__ import annotations

import math

import numpy as np

from .text import CONTENT

TOPK = 4


class AttrSpace:
    """属性 → 周波数。語 → (周波数の配列, 重みの配列)。"""

    def __init__(self, content_boards, func, D, bands=("B", "C", "D", "E", "G"), topk=TOPK):
        self.D, self.func, self.bands, self.topk = D, func, bands, topk
        self.step = self._odd_coprime(int(D * 0.6180339887))
        self.shift = self._odd_coprime(int(D * 0.4142135623))
        self.mix = self._odd_coprime(int(D * 0.7320508075))
        self.cb = {}
        for w in content_boards:
            if w["name"] not in self.cb or w.get("freq", 0) > self.cb[w["name"]].get("freq", 0):
                self.cb[w["name"]] = w
        self.idx = {}
        self._cache = {}

    def _odd_coprime(self, x):
        x |= 1
        while math.gcd(x, self.D) != 1:
            x += 2
        return x

    def f_of(self, name):
        if name not in self.idx:
            self.idx[name] = len(self.idx)
        return (self.idx[name] * self.step) % self.D

    def attrs(self, lemma, upos, xpos="", form=""):
        key = (lemma, upos, xpos)
        if key in self._cache:
            return self._cache[key]
        out = [("品詞:" + upos, 1.0)]
        if upos in CONTENT:
            w = self.cb.get(lemma)
            if w is not None:
                for b in self.bands:
                    A = w["attrs"].get(b, {})
                    for a, v in sorted(A.items(), key=lambda z: -z[1])[: self.topk]:
                        out.append((b + "|" + a, float(v)))
        else:
            w = self.func.board(dict(lemma=lemma, form=form or lemma, xpos=xpos))
            if w is not None:
                for b, A in w["attrs"].items():
                    for a, v in A.items():
                        out.append(("機能語|" + b + ":" + a, float(v)))
            else:
                out.append(("記号:" + (form or lemma), 1.0))
        f = np.array([self.f_of(a) for a, _ in out], dtype=np.int64)
        v = np.array([x for _, x in out], dtype=np.float32)
        v /= max(1e-9, float(np.linalg.norm(v)))
        self._cache[key] = (f, v)
        return f, v

    def tok(self, t):
        return self.attrs(t["lemma"], t["upos"], t["xpos"], t["form"])

    # ---- 1. 入口の波と、位置ごとの読み出し ----
    def encode(self, toks):
        """語 k の属性の周波数を k × SHIFT だけずらして重ねる（周波数のシフト。I48j・I48o のスライドと同じ考え）。"""
        X = np.zeros(self.D, dtype=np.complex64)
        for k, t in enumerate(toks):
            f, v = self.tok(t)
            np.add.at(X, (f + k * self.shift) % self.D, v)
        return X

    def read(self, X, k):
        return np.real(np.roll(X, -k * self.shift)).astype(np.float32)

    def raw(self, f, v):
        y = np.zeros(self.D, dtype=np.float32)
        np.add.at(y, f, v)
        return y

    def decode(self, X, n, M, lex_raw, rounds=2):
        """位置 0..n-1 の語を、候補の語の属性の波 M（行は長さ 1）との近さで読む。rounds 回、ほかの位置で読んだ語の波を引いて読み直す
        （浮かび上がらせ: 重なった波から、読めた語を差し引く）。返り値: 読みの回ごとの候補の番号の並び。"""
        R0 = np.stack([self.read(X, k) for k in range(n)])
        g = np.argmax(R0 @ M.T, axis=1)
        out = [g.copy()]
        for _ in range(rounds):
            Xr = np.zeros(self.D, dtype=np.float32)
            for k in range(n):
                Xr += np.roll(lex_raw[g[k]], k * self.shift)
            Rr = np.stack([np.roll(Xr, -k * self.shift) for k in range(n)])
            R = R0 - Rr + lex_raw[g]
            g = np.argmax(R @ M.T, axis=1)
            out.append(g.copy())
        return out

    def dense(self, f, v):
        y = np.zeros(self.D, dtype=np.float32)
        np.add.at(y, f, v)
        n = float(np.linalg.norm(y))
        return y / n if n > 0 else y

    # ---- 2. 逆回しの特徴（2 語の波の積の帯 ＋ それぞれの波 ＋ 印の帯） ----
    def pair(self, a, b, marks=()):
        fa, va = a
        if b is None:                                       # 文末: 最後の語の波 ＋ 意味の印の波
            fs = [fa]
            vs = [va]
            for m in sorted(marks):
                fs.append(np.array([(self.f_of("印:" + m) + self.shift) % self.D]))
                vs.append(np.array([1.0], dtype=np.float32))
            fs.append(np.array([(self.f_of("印:文末") + self.shift) % self.D]))
            vs.append(np.array([1.0], dtype=np.float32))
            return np.concatenate(fs), np.concatenate(vs)
        fb, vb = b
        fx = ((fa[:, None] + fb[None, :] * self.mix) % self.D).ravel()
        vx = (va[:, None] * vb[None, :]).ravel()
        return (np.concatenate([fx, fa, (fb + self.shift) % self.D]),
                np.concatenate([vx, va * 0.5, vb * 0.5]).astype(np.float32))


def __getattr__(name):
    if name in ("EV_BINS", "Fill", "ev_bin"):
        import importlib
        return getattr(importlib.import_module("i48ad_lookahead"), name)
    raise AttributeError(name)
