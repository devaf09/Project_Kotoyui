"""I49zq: 席の数を決めない「枠」。3 つ組（だれが・何を・どうした）を、助詞ごとの席に広げる（次の段階の 1 つ目。未確定の試し）。

枠 = 席の数だけの長さの組。席に語が無い所は None。席は助詞の名前のまま（「に」を「どこに」と決めつけない）。
  SEATS = が・を・に・で・と・へ・から・どうした
3 つの部品。どれも ask(隠す席, 枠, 候補) / learn(枠) / tick() の形で、今までの Mix（数えて選ぶ）にそのまま渡せる。

  FrameTable(vec_of, half=1024)   数えて覚える表（I49e の関係の表を、席の数を決めない形にした物）
    見せた席が n 個の時の下がり方（上の段に 1 つでも数が有れば、それを使う）:
      n/n            見せた語が全部同じ枠の数
      m/n・借りる    m 個は同じ語、残りは似た語（近さ |cos| を掛ける）。m = n-1, …, 1
      m/n            m 個は同じ語、残りは見ない。m = n-1, …, 1
      借りる         全部、似た語の数を近さを掛けて足す
    I49e と違う所: 「述語側を先に見る」という順を置かない。同じ語の数だけで段を決める。
  FrameSeatTable(vec_of, half=1024)   I49zs: 「どの席が同じ語だったか」で段を分ける表。I49zr で FrameTable が今までの表に届かなかった所の直し。
    段 = （同じ語だった席の組, 残りを似た語で借りるか見ないか）。同じ語の数が多い段が先（ここは FrameTable と同じ）。
    同じ数の段どうしの順は手で置かない。答え合わせ（learn）のたびに、段ごとの 1 位が当たったかを数え、それまでに多く当たってきた段を先に使う。
    数えは問う前の分だけ。同じなら、借りる方、席の並びの早い方。tally は (隠す席, 段) → [当たり, 問い]。
  FrameVec(base_of) / FrameSolve(fv, online=True)   波で解く形（I49k の SeatVec・Unbind の、席の数を決めない形）
    語の波 = 一緒に座った語の元の向きを、（自分の席, 相手の席）ごとに決まった位相の回し方で回して足した物。
    問われたら、見せた語を隠した席から見た回し方で回して足し、それに波が近い候補を答える。
    席が 3 つ（n=3）の時は、SeatVec(weight="数", bind="位相") と同じ波になる。
手で置く物: 席の並び、表の下がる順、近さ |cos|、薄れる速さ（1,024）、位相の傾き（席の組の番号 + 1）× 黄金比の小数部。乱数は置かない。
"""
from __future__ import annotations

import numpy as np

SEATS = ("が", "を", "に", "で", "と", "へ", "から", "どうした")
PHI = 0.6180339887498949


def _unit(v):
    v = np.asarray(v, dtype=np.complex64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


class FrameTable:
    def __init__(self, vec_of, half=1024.0):
        self.vec_of = vec_of
        self.keep = float(0.5 ** (1.0 / half))
        self.now = 0
        self.fr = {}                                      # 枠 → [数, 最後に触った時]
        self.by = {}                                      # (席, 語) → その語をその席に持つ枠の集まり
        self._u = {}
        self.rec = []                                     # (時, 隠す席, 段, 答え)

    def tick(self, n=1):
        self.now += int(n)

    def _count(self, t):
        c, at = self.fr[t]
        return c * self.keep ** (self.now - at)

    def learn(self, frame):
        t = tuple(frame)
        self.fr[t] = [self._count(t) + 1.0 if t in self.fr else 1.0, self.now]
        for j, w in enumerate(t):
            if w is not None:
                self.by.setdefault((j, w), set()).add(t)

    def _unit_of(self, w):
        if w not in self._u:
            v = self.vec_of(w)
            self._u[w] = None if v is None else _unit(v)
        return self._u[w]

    def near(self, a, b):
        if a == b:
            return 1.0
        ua, ub = self._unit_of(a), self._unit_of(b)
        if ua is None or ub is None:
            return 0.0
        return min(1.0, float(np.abs(np.vdot(ua, ub))))

    def ask(self, si, frame, cands):
        cands = list(cands)
        shown = [(j, w) for j, w in enumerate(frame) if j != si and w is not None]
        n = len(shown)
        ok = set(cands)
        soft = {m: {} for m in range(1, n)}               # m 個は同じ語、残りは似た語
        hard = {m: {} for m in range(1, n + 1)}           # m 個は同じ語、残りは見ない
        pool = set()
        for key in shown:
            pool |= self.by.get(key, set())
        for t in pool:
            c = t[si]
            if c not in ok:
                continue
            ex = sum(t[j] == w for j, w in shown)
            cnt = self._count(t)
            for m in range(1, ex + 1):
                hard[m][c] = hard[m].get(c, 0.0) + cnt
            if ex < n:
                g = 1.0
                for j, w in shown:
                    if t[j] != w:
                        g *= 0.0 if t[j] is None else self.near(w, t[j])
                        if g == 0.0:
                            break
                if g > 0:
                    for m in range(1, ex + 1):
                        soft[m][c] = soft[m].get(c, 0.0) + g * cnt
        order = [("%d/%d" % (n, n), hard.get(n, {}))] if n else []
        for m in range(n - 1, 0, -1):
            order += [("%d/%d・借りる" % (m, n), soft[m]), ("%d/%d" % (m, n), hard[m])]
        out = dict(段="答えられない", 点={}, 答え=None)
        for name, s in order:
            if any(v > 0 for v in s.values()):
                out = dict(段=name, 点=s, 答え=max(cands, key=lambda c: (s.get(c, 0.0), -cands.index(c))))
                break
        if out["答え"] is None and n:
            s = {}
            for t in self.fr:
                c = t[si]
                if c not in ok:
                    continue
                g = 1.0
                for j, w in shown:
                    g *= 0.0 if t[j] is None else self.near(w, t[j])
                    if g == 0.0:
                        break
                if g > 0:
                    s[c] = s.get(c, 0.0) + g * self._count(t)
            if any(v > 0 for v in s.values()):
                out = dict(段="借りる", 点=s, 答え=max(cands, key=lambda c: (s.get(c, 0.0), -cands.index(c))))
        self.rec.append((self.now, si, out["段"], out["答え"]))
        return out


class FrameSeatTable(FrameTable):
    def __init__(self, vec_of, half=1024.0):
        super().__init__(vec_of, half)
        self.tally, self._wait = {}, []

    def _rate(self, si, key):
        h = self.tally.get((si, key))
        return h[0] / h[1] if h and h[1] else 0.0

    def ask(self, si, frame, cands):
        cands = list(cands)
        shown = [(j, w) for j, w in enumerate(frame) if j != si and w is not None]
        n = len(shown)
        ok = set(cands)
        lv = {}                                           # (同じ語だった席の組, 借りるか) → 候補 → 点
        pool = set()
        for key in shown:
            pool |= self.by.get(key, set())
        for t in pool:
            c = t[si]
            if c not in ok:
                continue
            same = tuple(j for j, w in shown if t[j] == w)
            cnt = self._count(t)
            g = 1.0
            for j, w in shown:
                if t[j] != w:
                    g *= 0.0 if t[j] is None else self.near(w, t[j])
                    if g == 0.0:
                        break
            for sub in _subsets(same):
                s = lv.setdefault((sub, False), {})
                s[c] = s.get(c, 0.0) + cnt
                if len(sub) < n and g > 0:
                    s = lv.setdefault((sub, True), {})
                    s[c] = s.get(c, 0.0) + g * cnt

        def top(s):
            return max(cands, key=lambda c: (s.get(c, 0.0), -cands.index(c)))
        live = [(k, s) for k, s in lv.items() if any(v > 0 for v in s.values())]
        live.sort(key=lambda ks: (-len(ks[0][0]), -self._rate(si, ks[0]), not ks[0][1], ks[0][0]))
        self._wait.append((si, [(k, top(s)) for k, s in live]))
        out = dict(段="答えられない", 点={}, 答え=None)
        if live:
            k, s = live[0]
            name = "＋".join(SEATS[j] for j in k[0]) + ("・借りる" if k[1] else "")
            out = dict(段="%s｜%s" % (SEATS[si], name), 点=s, 答え=top(s))
        elif n:
            s = {}
            for t in self.fr:
                c = t[si]
                if c not in ok:
                    continue
                g = 1.0
                for j, w in shown:
                    g *= 0.0 if t[j] is None else self.near(w, t[j])
                    if g == 0.0:
                        break
                if g > 0:
                    s[c] = s.get(c, 0.0) + g * self._count(t)
            if any(v > 0 for v in s.values()):
                out = dict(段="借りる", 点=s, 答え=top(s))
        self.rec.append((self.now, si, out["段"], out["答え"]))
        return out

    def learn(self, frame):
        for si, tops in self._wait:
            for k, a in tops:
                h = self.tally.setdefault((si, k), [0, 0])
                h[0] += a == frame[si]
                h[1] += 1
        self._wait = []
        super().learn(frame)


def _subsets(same):
    """空でない部分の組を全部（席は多くて 3〜4 個なので、小さい）。"""
    out = []
    for m in range(1, 1 << len(same)):
        out.append(tuple(same[i] for i in range(len(same)) if m >> i & 1))
    return out


class FrameVec:
    def __init__(self, base_of, n=len(SEATS)):
        self.base_of, self.n = base_of, int(n)
        self.acc, self._b, self._ph = {}, {}, {}

    def pair(self, si, sj):
        return si * (self.n - 1) + (sj if sj < si else sj - 1)

    def rot(self, b, p):
        key = (p, len(b))
        if key not in self._ph:
            k = np.arange(len(b), dtype=np.float64)
            self._ph[key] = np.exp(2j * np.pi * ((p + 1) * PHI * k)).astype(np.complex64)
        return b * self._ph[key]

    def _base(self, w):
        if w not in self._b:
            v = self.base_of(w)
            self._b[w] = None if v is None else _unit(v)
        return self._b[w]

    def hear(self, frame):
        for si, w in enumerate(frame):
            if w is None:
                continue
            for sj, v in enumerate(frame):
                if sj == si or v is None:
                    continue
                b = self._base(v)
                if b is not None:
                    x = self.rot(b, self.pair(si, sj))
                    self.acc[w] = x.copy() if w not in self.acc else self.acc[w] + x

    def of(self, w):
        return self.acc.get(w)


class FrameSolve:
    def __init__(self, fv, online=True):
        self.sv, self.online, self.rec = fv, online, []

    def tick(self, n=1):
        pass

    def learn(self, frame):
        if self.online:
            self.sv.hear(frame)

    def aim(self, si, frame):
        a = None
        for sj, v in enumerate(frame):
            if sj == si or v is None:
                continue
            b = self.sv._base(v)
            if b is not None:
                x = self.sv.rot(b, self.sv.pair(si, sj))
                a = x if a is None else a + x
        return a

    def ask(self, si, frame, cands):
        cands = list(cands)
        a = self.aim(si, frame)
        out = dict(段="答えられない", 点={}, 答え=None)
        if a is not None:
            na = float(np.linalg.norm(a))
            s = {}
            for c in cands:
                v = self.sv.of(c)
                if v is not None:
                    nv = float(np.linalg.norm(v))
                    if nv > 0 and na > 0:
                        s[c] = float(np.real(np.vdot(a, v))) / (na * nv)
            if s and max(s.values()) > 0:
                out = dict(段="解く", 点=s, 答え=max(s, key=lambda c: (s[c], -cands.index(c))))
        self.rec.append((0, si, out["段"], out["答え"]))
        return out
