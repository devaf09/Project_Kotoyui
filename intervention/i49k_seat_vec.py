"""I49k: 語の向きを、座り方（どの席で、どの語と一緒に聞かれたか）から足し算で作り直す骨組み（持ち主の案「語板を再設計する必要があるかも」から、未確定）。

今の語板の向きは、差に意味が乗るように作られていない（I49j: ずれを足すと、かえって外れる）。
ここでは、語の向きを「一緒に座った語の元の向きを、席の組ごとに決まった回し方で回して、足した物」にする。数えて足すだけで、少しずつ寄せる学習はしない。
同じ席で同じ相手と聞かれる語どうしは、向きが近くなる。元の向きを持たない語も、相手に向きが有れば向きを持てる。

  SeatVec(base_of, weight="偏り")
    base_of(語) → 元の向き か None      今の語板の向き。足す材料としてだけ使う
    hear(3 つ組)                         座り方を数える（足すだけ）
    of(語) → 作った向き か None          その時までに数えた座り方から作る。相手に向きが 1 つも無ければ None
    weight="数"    聞いた数をそのまま重さにする
    weight="偏り"  たまたまより多く一緒に聞いた分だけを重さにする: max(0, log(一緒の数 × 全体 ÷ (語の数 × 相手の数)))
  回し方（席の違いを向きに残すため。席の組 =（自分の席, 相手の席））:
    bind="ずらす"  向きの並びを、席の組ごとに決まった数だけずらす（np.roll）
    bind="位相"    周波数ごとに違う角度で位相を回す。角度は席の組ごとに決まった傾きで周波数に比例する（席の組ごとに決まった時間だけ遅らせるのと同じ）
  Unbind(sv, online=False)   語を鍵にした表をまったく持たずに答える形。問いの見せた 2 語を、隠した席から見た回し方で回して足し、
                             それに作った向きが近い候補を答える（作った向きの中に「この席でこの相手と座った」が入っているかを共振で見る）。
                             online=True なら、答え合わせの後にその 3 つ組も sv に聞かせる（weight="数" の時だけ）。
手で置く物: 回し方（ずらす数 = (席の組の番号 + 1) × 次元 ÷ 7。位相の傾き = (席の組の番号 + 1) × 黄金比の小数部）、重さの 2 通り、
            元の向きを長さ 1 にそろえる事、Unbind の近さを cos の実部にした事。数は薄れさせていない。乱数は置かない。
"""
from __future__ import annotations

import math

import numpy as np

PAIRS = [(i, j) for i in range(3) for j in range(3) if i != j]


class SeatVec:
    def __init__(self, base_of, weight="偏り", bind="ずらす"):
        assert weight in ("数", "偏り") and bind in ("ずらす", "位相")
        self.base_of, self.weight, self.bind = base_of, weight, bind
        self.acc = {}                                     # 語 → 足した向き（weight="数" の時だけ。聞くたびに足す）
        self._ph = {}                                     # (席の組の番号, 次元) → 位相の回し方
        self.ctx = {}                                     # 語 → {(自分の席, 相手の席, 相手の語): 数}
        self.n_w = {}                                     # (自分の席, 語) → 数
        self.n_c = {}                                     # (自分の席, 相手の席, 相手の語) → 数
        self.n_p = {}                                     # (自分の席, 相手の席) → 数
        self._b = {}                                      # 語 → 長さ 1 の元の向き（無ければ None）
        self._v = {}                                      # 語 → 作った向き（hear のたびに空にする）

    def hear(self, tri):
        t = tuple(tri)
        for si, w in enumerate(t):
            d = self.ctx.setdefault(w, {})
            self.n_w[(si, w)] = self.n_w.get((si, w), 0) + 1
            for sj, v in enumerate(t):
                if sj == si:
                    continue
                k = (si, sj, v)
                d[k] = d.get(k, 0) + 1
                self.n_c[k] = self.n_c.get(k, 0) + 1
                self.n_p[(si, sj)] = self.n_p.get((si, sj), 0) + 1
                if self.weight == "数":
                    b = self._base(v)
                    if b is not None:
                        x = self.rot(b, (si, sj))
                        self.acc[w] = x.copy() if w not in self.acc else self.acc[w] + x
        self._v.clear()

    def rot(self, b, pair):
        p = PAIRS.index(tuple(pair))
        if self.bind == "ずらす":
            return np.roll(b, (p + 1) * (len(b) // 7))
        key = (p, len(b))
        if key not in self._ph:
            k = np.arange(len(b), dtype=np.float64)
            self._ph[key] = np.exp(2j * np.pi * ((p + 1) * 0.6180339887498949 * k)).astype(np.complex64)
        return b * self._ph[key]

    def _base(self, w):
        if w not in self._b:
            v = self.base_of(w)
            if v is not None:
                v = np.asarray(v, dtype=np.complex64)
                n = float(np.linalg.norm(v))
                v = v / n if n > 0 else None
            self._b[w] = v
        return self._b[w]

    def _w(self, w, k, c):
        if self.weight == "数":
            return float(c)
        si, sj, _ = k
        return max(0.0, math.log(c * self.n_p[(si, sj)] / (self.n_w[(si, w)] * self.n_c[k])))

    def of(self, w):
        if self.weight == "数":
            return self.acc.get(w)
        if w in self._v:
            return self._v[w]
        out = None
        for k, c in self.ctx.get(w, {}).items():
            b = self._base(k[2])
            if b is None:
                continue
            g = self._w(w, k, c)
            if g <= 0:
                continue
            x = g * self.rot(b, (k[0], k[1]))
            out = x if out is None else out + x
        self._v[w] = out
        return out


class Unbind:
    def __init__(self, sv, online=False):
        assert not online or sv.weight == "数"
        self.sv, self.online = sv, online
        self.rec = []

    def tick(self, n=1):
        pass

    def learn(self, tri):
        if self.online:
            self.sv.hear(tri)

    def ask(self, si, tri, cands):
        cands = list(cands)
        aim = None
        for sj in range(3):
            if sj == si:
                continue
            b = self.sv._base(tri[sj])
            if b is not None:
                x = self.sv.rot(b, (si, sj))
                aim = x if aim is None else aim + x
        out = dict(段="答えられない", 点={}, 答え=None)
        if aim is not None:
            na = float(np.linalg.norm(aim))
            s = {}
            for c in cands:
                v = self.sv.of(c)
                if v is not None:
                    nv = float(np.linalg.norm(v))
                    if nv > 0 and na > 0:
                        s[c] = float(np.real(np.vdot(aim, v))) / (na * nv)
            if s and max(s.values()) > 0:
                best = max(s, key=lambda c: (s[c], -cands.index(c)))
                out = dict(段="解く", 点=s, 答え=best)
        self.rec.append((0, si, tuple(tri[j] for j in range(3) if j != si), out["段"], out["答え"]))
        return out
