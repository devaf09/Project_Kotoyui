"""I48ab: 入口の波の置き方の比べ（順方向: 文の波 → 位置ごとの語）。i48ab_long_grain.AttrSpace の属性の周波数を使う。

置き方（語 k、文の語の数 n）:
  頭から  : 属性の帯を k × SHIFT ずらす（基準。短い文は前の方のずれしか使わない）
  文末から: (n − 1 − k) × SHIFT ずらす（述語の 1 つ前が、文の長さによらず同じずれ）
  位相の板: 語の帯に決まった乱数の位相を掛け、位置は時間のずれ（位相の傾き）で持つ（D 全体に広がるが、属性 1 つは周波数 1 つのまま）
  拡散    : 属性ごとに D 全体にわたる決まった乱数の符号（複素の位相、または ±1）を持たせ、語 = 属性の符号の重みつきの和。位置は k × SHIFT の
            巡回のずれ。どの位置もどの文の長さも D 全体を同じように使う
  畳み込み: 頭からのずれを、幅のある核（位置 k−1・k・k+1 に 0.5・1・0.5）で置く（位置が 1 つずれても近い波）
読み出し: 位置 k のずれを戻した波と、候補の語の波との近さ。読めた語の波をほかの位置から引いて読み直す（rounds 回）。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import numpy as np

MODES = ("頭から", "文末から", "位相の板", "拡散", "拡散・符号", "畳み込み")


class Placer:
    def __init__(self, S, mode, seed=0):
        self.S, self.mode, self.D = S, mode, S.D
        self.rng = np.random.default_rng(seed)
        self._codes = {}
        self.plate = np.exp(2j * np.pi * self.rng.random(self.D)).astype(np.complex64)

    def _code(self, f):
        """属性の周波数 f の、D 全体の符号（拡散の時）。"""
        c = self._codes.get(int(f))
        if c is None:
            r = np.random.default_rng(1000003 * int(f) + 7)
            if self.mode == "拡散・符号":
                c = (r.integers(0, 2, self.D) * 2 - 1).astype(np.complex64) / np.sqrt(self.D)
            else:
                c = np.exp(2j * np.pi * r.random(self.D)).astype(np.complex64) / np.sqrt(self.D)
            self._codes[int(f)] = c
        return c

    def word(self, f, v):
        if self.mode.startswith("拡散"):
            y = np.zeros(self.D, dtype=np.complex64)
            for fi, vi in zip(f, v):
                y += vi * self._code(fi)
            return y
        y = np.zeros(self.D, dtype=np.complex64)
        np.add.at(y, f, v)
        if self.mode == "位相の板":
            y = y * self.plate
        return y

    def shifts(self, k, n):
        s = self.S.shift
        if self.mode == "文末から":
            return [((n - 1 - k) * s, 1.0)]
        if self.mode == "畳み込み":
            return [((k + d) * s, w) for d, w in ((-1, 0.5), (0, 1.0), (1, 0.5)) if k + d >= 0]
        return [(k * s, 1.0)]

    def put(self, y, sh):
        if self.mode == "位相の板":                         # 時間のずれ = 周波数ごとの位相の傾き
            return y * np.exp(-2j * np.pi * np.arange(self.D) * sh / self.D).astype(np.complex64)
        return np.roll(y, sh)

    def take(self, X, sh):
        if self.mode == "位相の板":
            return X * np.exp(2j * np.pi * np.arange(self.D) * sh / self.D).astype(np.complex64)
        return np.roll(X, -sh)

    def encode(self, words):
        n = len(words)
        X = np.zeros(self.D, dtype=np.complex64)
        for k, y in enumerate(words):
            for sh, w in self.shifts(k, n):
                X += w * self.put(y, sh)
        return X

    def decode(self, X, n, L, rounds=2):
        """L: 候補の語の波（行、複素）。近さ = Re<候補, 読み>/|候補|。返り値: 読みの回ごとの番号の並び。"""
        Ln = L / np.maximum(1e-9, np.linalg.norm(L, axis=1, keepdims=True))
        cen = [self.shifts(k, n)[0][0] if self.mode != "畳み込み" else k * self.S.shift for k in range(n)]
        Y0 = np.stack([self.take(X, cen[k]) for k in range(n)])
        g = np.argmax(np.real(Y0 @ np.conj(Ln).T), axis=1)
        out = [g.copy()]
        for _ in range(rounds):
            Xr = self.encode([L[i] for i in g])
            Yr = np.stack([self.take(Xr, cen[k]) for k in range(n)])
            Y = Y0 - Yr + L[g]
            g = np.argmax(np.real(Y @ np.conj(Ln).T), axis=1)
            out.append(g.copy())
        return out

    def usage(self, X):
        """D のうち、波が入っている周波数の割合。"""
        a = np.abs(X)
        return float((a > 1e-6 * max(1e-12, a.max())).mean())
