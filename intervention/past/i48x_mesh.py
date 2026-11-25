"""I48x: 感情・型・報酬を後から足すための土台（持ち主 2026-11-15「これをベースにすれば開発速度はかなり早くなる」）。

4 つの口:
1. 帯の登録（BandTable）: 帯を 1 つの表（名前、周波数、誰の物か = 組）で持つ。帯を足す時は add() に 1 行足すだけ。
   周波数は、組どうしの和（複素の波の 2 乗が作る周波数、D で回る）が、ほかのどの和とも帯とも重ならない所に自動で置く（I48w）。
   check() で重ならないことを確かめる（テストで自動に走る）。
2. 刺激の口（Stimulus）: 「何が起きたら（出来事）、どの帯に、どちら向きに入るか」を 1 行で登録する。向きは置き、強さは学ぶ（learn）。
3. 遅れて付く点の口（Delayed）: 印の名前と、いつ付くか（next = 相手の次の行を見てから前の返事に、end = 会話の終わりに返事全体へ薄く配る）を
   登録する。said() で言った返事の鍵を渡し、resolve() で相手の次の行（または会話の終わり）の結果を渡すと、印の数え（良の経験）に入る。
   value() は P(良) と同じ形 (良 + 0.1) / (言った数 + 1) の平均。--bind・貢献の 2 段目・3 段目は、この口に印を 1 つ登録するだけで足せる。
4. 門の口（Gate）: 門は「通す / 絞る / 固定の返事に替える」を返す同じ形。GateChain に並べ、上から順に見る（深刻な相談 → 白銀律 → 言ってよい時か）。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import itertools
from collections import Counter, defaultdict

# ---- 1. 帯の登録 ----


class BandTable:
    def __init__(self, D=2048):
        self.D = D
        self.rows = []                                      # (名前, 周波数, 組)

    def freqs(self, group=None):
        return [f for n, f, g in self.rows if group is None or g == group]

    def add(self, name, group):
        """帯を 1 本足す。周波数は小さい方から、check() が通る所に置く。"""
        if any(n == name for n, _, _ in self.rows):
            raise ValueError(f"同じ名前の帯がある: {name}")
        for f in range(1, self.D):
            if f in self.freqs():
                continue
            self.rows.append((name, f, group))
            if self.check():
                return f
            self.rows.pop()
        raise ValueError("置ける周波数が無い")

    def check(self):
        """違う組どうしの和 (f_a + f_b) mod D が、ほかのどの和（同じ組どうし・2 倍・別の組の組み合わせ）とも、帯の周波数とも重ならないか。"""
        fs = [(f, g) for _, f, g in self.rows]
        base = {f for f, _ in fs}
        sums = defaultdict(list)
        for i, j in itertools.combinations_with_replacement(range(len(fs)), 2):
            sums[(fs[i][0] + fs[j][0]) % self.D].append((i, j))
        for i, j in itertools.combinations(range(len(fs)), 2):
            if fs[i][1] != fs[j][1]:
                s = (fs[i][0] + fs[j][0]) % self.D
                if len(sums[s]) > 1 or s in base:
                    return False
        return True

    def freq(self, name):
        return next(f for n, f, _ in self.rows if n == name)


# ---- 2. 刺激の口 ----


class Stimulus:
    def __init__(self):
        self.links = defaultdict(list)                      # 出来事 → [(帯, 向き)]
        self.w = Counter()                                  # (出来事, 帯) → 強さ（学ぶ）

    def link(self, event, band, sign=1.0, w0=1.0):
        self.links[event].append((band, float(sign)))
        self.w[(event, band)] = w0

    def apply(self, events, state):
        """起きた出来事（名前 → 大きさ）を、帯の強さ state（名前 → 0〜1）に足す。"""
        for e, x in events.items():
            for band, sign in self.links.get(e, ()):
                state[band] = min(1.0, max(0.0, state.get(band, 0.0) + sign * self.w[(e, band)] * x))
        return state

    def learn(self, events, before, target, lr=0.1):
        """答え（target: 帯 → 強さ）に近づくように、強さだけを動かす（向きは動かさない）。"""
        pred = self.apply(events, dict(before))
        for e, x in events.items():
            for band, sign in self.links.get(e, ()):
                if band in target:
                    self.w[(e, band)] = max(0.0, self.w[(e, band)] + lr * (target[band] - pred[band]) * sign * x)


# ---- 3. 遅れて付く点の口 ----


class Delayed:
    NEXT, END = "next", "end"

    def __init__(self):
        self.marks = {}                                     # 印の名前 → いつ付くか
        self.said = defaultdict(Counter)                    # 印 → 鍵 → 言った数
        self.good = defaultdict(Counter)                    # 印 → 鍵 → 良の数（END は配った分）
        self.pending = defaultdict(list)                    # 印 → この会話で言った返事の鍵の並び（END 用）
        self.last = {}                                      # 印 → 前の返事の鍵（NEXT 用）
        self.log = Counter()

    def register(self, mark, when):
        assert when in (self.NEXT, self.END)
        self.marks[mark] = when

    def said_keys(self, mark, keys, eligible=True):
        """返事を言った。keys = その返事の鍵（所、無ければ行）。eligible = この印の対象の返事か（--bind なら結びか）。"""
        for k in keys:
            self.said[mark][k] += 1
        self.log[mark + ":言った"] += 1
        if self.marks[mark] == self.NEXT:
            self.last[mark] = list(keys) if eligible else None
        else:
            self.pending[mark].append(list(keys))
        if eligible:
            self.log[mark + ":対象"] += 1

    def resolve(self, mark, ok, weight=1.0):
        """NEXT: 相手の次の行を見て、前の返事が良かったか。END: 会話の終わりに、返事全体へ weight を薄く配る。"""
        if self.marks[mark] == self.NEXT:
            ks = self.last.pop(mark, None)
            if ok and ks:
                for k in ks:
                    self.good[mark][k] += weight
                self.log[mark + ":良"] += 1
        else:
            P = self.pending.pop(mark, [])
            if ok and P:
                for ks in P:
                    for k in ks:
                        self.good[mark][k] += weight / len(P)
                self.log[mark + ":良"] += 1

    def value(self, mark, keys):
        s, g = self.said[mark], self.good[mark]
        return sum((g[k] + 0.1) / (s[k] + 1.0) for k in keys) / max(1, len(keys))


# ---- 4. 門の口 ----

PASS, NARROW, FIXED = "通す", "絞る", "固定"


class GateChain:
    """門の並び。各門は gate(ctx) → (PASS, None) / (NARROW, 暗くする条件) / (FIXED, 返事) を返す。上から順に見て、固定が出たらそこで止める。"""

    def __init__(self):
        self.gates = []

    def add(self, name, fn):
        self.gates.append((name, fn))

    def run(self, ctx):
        narrow = []
        for name, fn in self.gates:
            kind, x = fn(ctx)
            if kind == FIXED:
                return FIXED, x, name
            if kind == NARROW:
                narrow.append((name, x))
        return (NARROW, narrow, None) if narrow else (PASS, None, None)
