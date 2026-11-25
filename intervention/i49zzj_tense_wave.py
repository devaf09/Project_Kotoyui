"""I49zzj: 状況波（時制）の小さな実験の部品（持ち主の案、未確定の試し）。

持ち主の案: 動詞の語板に、時制の形（未来・現在・現在進行・過去・過去進行）を属性として並べ、周波数の上でなだらかにつなぐ。
  状況波（時制）を 1 つ持つ。文を聞くたびに、その文の動詞の形が持つ時制の波で、状況波を動かす。
  「今」「さっき」のような時の語は、一緒に出た文の時制の波を足して覚える（数えて足すだけ）。
  時の語を聞いたら、覚えた波で状況波を動かす。状況波に近い形を選べば、会話の中で時制がそろう。

ここに有る物:
  TenseBoard: 時制の波（周波数の軸の上の山）と、動詞の形の表。
  TimeWords: 時の語ごとに、一緒に出た時制の波を足して持つ。比べ役として、周波数 1 つ（山の位置の平均）で持つ形も数える。
  Situation: 状況波。聞いた波を足し、前の分は薄れる（消さない）。

手で置いた物: 時制の並び順（持ち主の言った順）、山の間隔と幅、軸の長さ、状況波の薄れる速さ。
乱数は使わない。float32。
"""
from __future__ import annotations

import numpy as np

TENSES = ("未来", "現在", "現在進行", "過去", "過去進行")
BINS = 64          # 周波数の軸の長さ（手で置いた値）
STEP = 12.0        # 隣の時制との間隔（手で置いた値）
WIDTH = 8.0        # 山の幅。隣と重なる広さ（手で置いた値）
HALF = 1.0         # 状況波が半分に薄れるまでの文の数（手で置いた値）


def bump(center, width=None, bins=BINS):
    """周波数の軸の上の、center を頂上にした山（長さ 1）。width を書かなければ WIDTH。"""
    width = WIDTH if width is None else width
    f = np.arange(bins, dtype=np.float32)
    v = np.exp(-((f - np.float32(center)) ** 2) / np.float32(2.0 * width * width)).astype(np.float32)
    return v / np.linalg.norm(v)


def cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else 0.0


class TenseBoard:
    """時制の波と、動詞の形の表。forms: {動詞: {時制: 形}}。"""

    def __init__(self, forms, order=TENSES):
        self.order = tuple(order)
        first = (BINS - STEP * (len(self.order) - 1)) / 2.0
        self.center = {t: first + STEP * i for i, t in enumerate(self.order)}
        self.waves = {t: bump(c) for t, c in self.center.items()}
        self.forms = forms
        self.tense_of = {s: t for fs in forms.values() for t, s in fs.items()}

    def wave(self, tense):
        return self.waves[tense]

    def wave_of_form(self, form):
        return self.waves[self.tense_of[form]]

    def nearest(self, v):
        """波 v に一番近い時制と、時制ごとの近さ。"""
        c = {t: cos(v, w) for t, w in self.waves.items()}
        return max(self.order, key=lambda t: c[t]), c

    def pick_form(self, verb, v):
        """動詞 verb の形のうち、波 v に一番近い時制の形。"""
        t = max(self.order, key=lambda t: cos(v, self.waves[t]) if t in self.forms[verb] else -2.0)
        return self.forms[verb][t]

    def center_of(self, v):
        """波 v の山の位置の平均（周波数 1 つにまとめた値）。"""
        v = np.abs(v)
        return float((v * np.arange(len(v))).sum() / v.sum()) if v.sum() > 0 else None


class TimeWords:
    """時の語ごとに、一緒に出た時制の波を足して持つ。"""

    def __init__(self, board):
        self.board = board
        self.sum = {}      # 語 → 足した波（山の形のまま持つ）
        self.fsum = {}     # 語 → 山の位置の和（周波数 1 つで持つ比べ役）
        self.n = {}

    def learn(self, word, form):
        t = self.board.tense_of[form]
        self.sum[word] = self.sum.get(word, np.zeros(BINS, dtype=np.float32)) + self.board.wave(t)
        self.fsum[word] = self.fsum.get(word, 0.0) + self.board.center[t]
        self.n[word] = self.n.get(word, 0) + 1

    def wave(self, word, single=False):
        """覚えた波。single=True なら、周波数 1 つ（位置の平均）に山を立てた波。知らない語は None。"""
        if word not in self.sum:
            return None
        return bump(self.fsum[word] / self.n[word]) if single else self.sum[word]


class Situation:
    """状況波（時制）。聞いた波を足し、前の分は薄れる。"""

    def __init__(self, half=None):
        self.keep = np.float32(0.5 ** (1.0 / (HALF if half is None else half)))
        self.state = np.zeros(BINS, dtype=np.float32)

    def hear(self, wave):
        if wave is not None:
            n = float(np.linalg.norm(wave))
            self.state = self.state * self.keep + (wave / n if n > 0 else wave)
        return self.state
