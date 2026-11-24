"""I49zg: 波は固定のまま、読む時の振幅（どの属性の周波数を重く見るか）だけを、誤差で直す（持ち主の案の第 1 段、未確定の試し）。

持ち主の言葉:「波動・干渉の物理構造を守りつつ、パラメータ（周波数・位相・振幅）の調整には誤差逆伝播を解禁する」「段階分けで進めて」
今までの決まり（数えて足すだけ、値を直接動かさない）から外れる試し。採用ではなく、効くかどうかを数えるための物。
ここで動かすのは、周波数ごとの読む時の振幅 a（0 以上、初めは全部 1）だけ。周波数、位相、覚えた波（SeatVec の中身）は動かさない。
覚えた波に手を入れないので、振幅を動かしても、前に足した分とずれない（I49p で崩れた形にはならない）。乱数は使わない。

  AmpRead(sv, rule="外れた時", lr=0.01, by_seat=False, temp=0.05)
    sv: i49k_seat_vec.SeatVec（weight="数"）。答え方は Unbind（side="候補"、norm="cos"）と同じで、内積と長さに振幅を掛ける:
        点 = Σ a・Re(狙い*・候補の波) ÷ (√Σ a・|狙い|² × √Σ a・|候補の波|²)
    rule:
      "外れた時"  答えが外れた時だけ、a += lr × √次元 × (正解の寄与 − 選んだ語の寄与) を長さ 1 にそろえた物（パーセプトロンの形）
      "毎回"      毎回、a += lr × √次元 × (正解の寄与 − 候補の寄与を確からしさで均した物) ÷ 正解の寄与の長さ
                  確からしさ = 点 ÷ temp のソフトマックス（交差エントロピーの勾配の形。当たりが確かになるほど動きが小さくなる）
      寄与 = 周波数ごとの Re(狙い*・候補の波) ÷ (狙いの長さ × 候補の波の長さ)。長さの中の a は、勾配では動かない物として扱う。
    by_seat: 隠した席ごとに別の振幅を持つ
    ask / learn / tick は Unbind と同じ形（FlowPick で包める）。直すのは learn の時（答え合わせの後）で、その後に 3 つ組を sv に聞かせる。
    lr=0 なら Unbind と同じ答えになる。
    spread(): 振幅の広がり（一番小さい、一番大きい、0 になった割合、上位 1 割が全体に占める割合）。
手で置く物: 直す速さ lr、temp、振幅を 0 以上に切る事、初めの振幅 1、寄与を長さでそろえる事。
"""
from __future__ import annotations

import numpy as np


class AmpRead:
    def __init__(self, sv, rule="外れた時", lr=0.01, by_seat=False, temp=0.05):
        assert sv.weight == "数" and not (sv.pair or sv.mark)
        assert rule in ("外れた時", "毎回")
        self.sv, self.rule, self.lr, self.by_seat, self.temp = sv, rule, float(lr), by_seat, float(temp)
        self.a = None                                     # (席 か 1, 次元) の振幅
        self._asked = []
        self.n_fix = 0                                    # 振幅を直した回数

    def tick(self, n=1):
        pass

    def _amp(self, si, dim):
        if self.a is None:
            self.a = np.ones((3 if self.by_seat else 1, dim), dtype=np.float32)
        return self.a[si if self.by_seat else 0]

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
        if aim is None or float(np.linalg.norm(aim)) <= 0:
            return out
        names, vs = [], []
        for c in cands:
            v = self.sv.of(c)
            if v is not None and float(np.linalg.norm(v)) > 0:
                names.append(c)
                vs.append(v)
        if not names:
            return out
        a = self._amp(si, len(aim))
        V = np.stack(vs)
        re = np.real(V * np.conj(aim)[None, :]).astype(np.float32)          # 周波数ごとの寄与（長さで割る前）
        na = float(np.sqrt(np.sum(a * np.abs(aim) ** 2)))
        nv = np.sqrt(np.sum(a[None, :] * np.abs(V) ** 2, axis=1))
        live = nv > 0
        if na <= 0 or not live.any():
            return out
        s = np.full(len(names), -np.inf, dtype=np.float64)
        s[live] = (re[live] @ a) / (na * nv[live])
        g = re / (float(np.linalg.norm(aim)) * np.linalg.norm(V, axis=1))[:, None]
        self._asked.append((si, names, g, s, aim, V, tuple(tri)))
        pts = {c: float(x) for c, x in zip(names, s) if np.isfinite(x)}
        if pts and max(pts.values()) > 0:
            out = dict(段="解く", 点=pts, 答え=max(pts, key=lambda c: (pts[c], -cands.index(c))))
        return out

    def _fix(self, si, names, g, s, truth):
        if self.lr <= 0 or truth not in names:
            return
        t = names.index(truth)
        if self.rule == "外れた時":
            w = int(np.argmax(s))
            if w == t:
                return
            d = g[t] - g[w]
            n = float(np.linalg.norm(d))
        else:
            z = np.where(np.isfinite(s), s, -1e9) / self.temp
            p = np.exp(z - z.max())
            p /= p.sum()
            d = g[t] - p.astype(np.float32) @ g
            n = float(np.linalg.norm(g[t]))
        if n <= 0:
            return
        a = self._amp(si, g.shape[1])
        a += np.float32(self.lr * np.sqrt(g.shape[1]) / n) * d
        np.maximum(a, 0.0, out=a)
        self.n_fix += 1

    def learn(self, tri):
        for si, names, g, s, *_ in self._asked:
            self._fix(si, names, g, s, tri[si])
        self._asked = []
        self.sv.hear(tri)

    def spread(self):
        if self.a is None:
            return None
        a = np.sort(self.a.ravel())[::-1]
        top = max(1, len(a) // 10)
        return dict(小=round(float(a[-1]), 4), 大=round(float(a[0]), 4), 零の割合=round(float(np.mean(a <= 0)), 4),
                    上位一割の占める割合=round(float(a[:top].sum() / max(1e-9, a.sum())), 4), 直した回数=self.n_fix)
