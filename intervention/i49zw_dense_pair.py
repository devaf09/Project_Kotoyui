"""I49zw: 2 語の組を、広げてから掛けて束ねる（持ち主がもらった助言から。未確定の試し）。

助言:「波動計算（ホログラフィック表現／循環畳み込み）で 2 語の組み合わせを掛け算で表現するには、先に FFT 等で全周波数に
エネルギーを拡散させた密な複素スペクトルに変換してから要素積を取る必要があるんじゃないか？」
今までの束ね（I49l・I49m、SeatVec(pair=True)）は、属性の波どうしを周波数ごとに掛けていた。属性の波は、その語が持つ属性の所にだけ
力が有るまばらな波なので、掛けて残るのは「2 語に共通の属性」だけで、組の印にならなかった。
ここでは、属性の波を dim 本の並びに置き直し、FFT で全体に広げてから要素ごとに掛ける。元の並びの側では畳み込みになり、
「語 A の属性 i と 語 B の属性 j の組」ごとに別の場所へ力が立つ。似た語の組どうしは、似た印になる（近さは 2 語の近さの掛け算）。

  PairSolve(attr_of, mode="広げる", dim=8192)
    mode: "広げる" = 広げてから掛ける ／ "まばら" = 今までと同じ、属性の波をそのまま要素ごとに掛ける（比べ役）
    learn(tri): 3 つの席それぞれについて、その席の語の覚えに、ほかの 2 語の組の印（長さ 1）を足す
    ask(隠す席, tri, 候補): 見せた 2 語の組の印と、候補がその席で覚えた印の和との近さ（cos の実部）。一番近い候補を答える
    tick(): 何もしない（薄れさせていない）
    stat(a, b): 2 語の属性の数、共通の属性の数、まばらな掛け算で残る長さ（数える用）
    amp（mode="広げる" の時だけ。助言の「密な複素スペクトル」のもう半分、大きさをそろえる事の確かめ）:
      "そのまま"   属性の割合をそのまま置く（初めの形）
      "属性を 1"   持っている属性は、割合を見ずに全部 1 として置く
      "位相だけ"   広げた後、どの本も大きさを 1 にして位相だけを残す（力を全部の本に同じだけ配る）
  置き直し: 席 s の語の属性 i は、(g_s × (i + 1)) mod dim 本目に置く。g_s は席ごとに違う奇数（どの席の語かで印が変わるように）。
手で置く物: dim、g_s の作り方（dim × 黄金比の小数部 ×（席 + 1）に一番近い奇数）、印の長さを 1 にそろえる事、近さを cos の実部にした事。乱数は置かない。
"""
from __future__ import annotations

import numpy as np

PHI = 0.6180339887498949


class PairSolve:
    def __init__(self, attr_of, mode="広げる", dim=8192, amp="そのまま"):
        assert mode in ("広げる", "まばら") and amp in ("そのまま", "属性を 1", "位相だけ")
        self.attr_of, self.mode, self.dim, self.amp = attr_of, mode, int(dim), amp
        self.g = [int(self.dim * PHI * (s + 1)) % self.dim | 1 for s in range(3)]
        self._d = {}
        self.mem = [{}, {}, {}]                            # 席 → 語 → 覚えた印の和

    def _attr(self, w):
        v = self.attr_of(w)
        return None if v is None else np.asarray(v, dtype=np.complex64)

    def dense(self, w, s):
        """席 s の語 w の、広げた波。"""
        key = (w, s)
        if key not in self._d:
            v = self._attr(w)
            x = None
            if v is not None and float(np.linalg.norm(v)) > 0:
                idx = np.nonzero(v)[0]
                put = np.zeros(self.dim, dtype=np.complex64)
                np.add.at(put, (self.g[s] * (idx + 1)) % self.dim, 1.0 if self.amp == "属性を 1" else v[idx])
                x = np.fft.fft(put).astype(np.complex64)
                if self.amp == "位相だけ":
                    m = np.abs(x)
                    x = np.where(m > 0, x / np.where(m > 0, m, 1), 0).astype(np.complex64)
            self._d[key] = x
        return self._d[key]

    def sign(self, si, tri):
        """隠す席 si から見た、ほかの 2 語の組の印（長さ 1）。作れなければ None。"""
        sj, sk = [j for j in range(3) if j != si]
        if tri[sj] is None or tri[sk] is None:
            return None
        if self.mode == "まばら":
            a, b = self._attr(tri[sj]), self._attr(tri[sk])
            x = None if a is None or b is None else a * b
        else:
            a, b = self.dense(tri[sj], sj), self.dense(tri[sk], sk)
            x = None if a is None or b is None else a * b
        if x is None:
            return None
        n = float(np.linalg.norm(x))
        return None if n == 0 else (x / n).astype(np.complex64)

    def learn(self, tri):
        for si in range(3):
            p = self.sign(si, tri)
            if p is not None:
                m = self.mem[si]
                m[tri[si]] = p.copy() if tri[si] not in m else m[tri[si]] + p

    def tick(self, n=1):
        pass

    def ask(self, si, tri, cands):
        cands = list(cands)
        out = dict(段="答えられない", 点={}, 答え=None)
        p = self.sign(si, tri)
        if p is not None:
            s = {}
            for c in cands:
                v = self.mem[si].get(c)
                if v is not None:
                    nv = float(np.linalg.norm(v))
                    if nv > 0:
                        s[c] = float(np.real(np.vdot(p, v))) / nv
            if s and max(s.values()) > 0:
                out = dict(段="組", 点=s, 答え=max(s, key=lambda c: (s[c], -cands.index(c))))
        return out

    def stat(self, a, b):
        x, y = self._attr(a), self._attr(b)
        if x is None or y is None:
            return None
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        if nx == 0 or ny == 0:
            return None
        return dict(属性の数=(int(np.count_nonzero(x)), int(np.count_nonzero(y))), 共通の属性=int(np.count_nonzero(x * y)),
                    残る長さ=float(np.linalg.norm((x / nx) * (y / ny))))


class Both:
    """2 つの答え方の点を、そのまま足して選ぶ（重さは 1 対 1。手で置いた値）。"""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def ask(self, si, tri, cands):
        cands = list(cands)
        x, y = self.a.ask(si, tri, cands), self.b.ask(si, tri, cands)
        s = dict(x["点"])
        for c, v in y["点"].items():
            s[c] = s.get(c, 0.0) + v
        if not s or max(s.values()) <= 0:
            return dict(段="答えられない", 点={}, 答え=None)
        return dict(段="足す", 点=s, 答え=max(s, key=lambda c: (s[c], -cands.index(c))))

    def learn(self, tri):
        self.a.learn(tri)
        self.b.learn(tri)

    def tick(self):
        self.a.tick()
        self.b.tick()
