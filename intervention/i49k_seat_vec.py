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
    bind="なし"    回さない。席を見ず、一緒に出た語をそのまま足す（行の語だけで作る形。hear_line で 1 行の語をまとめて聞かせられる）
    bind="位相"    周波数ごとに違う角度で位相を回す。角度は席の組ごとに決まった傾きで周波数に比例する（席の組ごとに決まった時間だけ遅らせるのと同じ）
  Unbind(sv, online=False)   語を鍵にした表をまったく持たずに答える形。問いの見せた 2 語を、隠した席から見た回し方で回して足し、
                             それに作った向きが近い候補を答える（作った向きの中に「この席でこの相手と座った」が入っているかを共振で見る）。
                             online=True なら、答え合わせの後にその 3 つ組も sv に聞かせる（weight="数" の時だけ）。
  pair=True（weight="数" の時だけ）: 一緒に座った 2 語を 1 つに束ねた向きも足す。束ね方 = 周波数ごとに掛け合わせる（位相を足す）。
                             足すだけでは「A とも B とも座った」と「A と B の組と座った」が区別できないので、組を 1 本の向きにして持つ。
                             Unbind は、pair=True の時は問いの 2 語を束ねた向きも足して探す。
  Unbind の side: "候補" = 候補の作った向きの中に、見せた 2 語が入っているかを見る（今までの形。聞いた事の無い候補は比べられない）。
                  "見せた語" = 見せた語の作った向きを回し戻して、隠した席に来る語の元の向き（属性なら、属性の割合）を言い当て、
                               候補の元の向きと比べる。聞いた事の無い候補も、元の向きが有れば比べられる。
                  "両方" = 2 つの点を足す。
  Unbind の norm: 候補の作った向きの長さで割るかどうか。"cos" = 長さで割る（たくさん聞いた語ほど、1 つの相手との近さは小さくなる）、
                             "なし" = 割らない（その相手と一緒に聞いた数に近い量になる。よく出る語が有利）、"半分" = 長さの平方根で割る（その間）。
  center=True（weight="数" の時だけ）: 作った向きから「たまたまの分」を引く。たまたまの分 = 足した回数 × どの語にも足された向きの平均。
                             よく出る語（食べる など）の成分が、どの語にも大きく入って混ざるのを減らすため。足し算と引き算だけで、聞くたびに直せる。
手で置く物: 回し方（ずらす数 = (席の組の番号 + 1) × 次元 ÷ 7。位相の傾き = (席の組の番号 + 1) × 黄金比の小数部）、重さの 2 通り、
            元の向きを長さ 1 にそろえる事、Unbind の近さを cos の実部にした事。数は薄れさせていない。乱数は置かない。
"""
from __future__ import annotations

import math

import numpy as np

PAIRS = [(i, j) for i in range(3) for j in range(3) if i != j]


class SeatVec:
    def __init__(self, base_of, weight="偏り", bind="ずらす", center=False, pair=False):
        assert weight in ("数", "偏り") and bind in ("ずらす", "位相", "なし")
        assert not (center or pair) or weight == "数"
        self.pair = pair
        self.base_of, self.weight, self.bind, self.center = base_of, weight, bind, center
        self.n_add = {}                                   # 語 → 足した回数
        self.tot = None                                   # 全部の語に足した向きの合計
        self.n_tot = 0
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
                        self.n_add[w] = self.n_add.get(w, 0) + 1
                        self.tot = x.astype(np.complex128) if self.tot is None else self.tot + x
                        self.n_tot += 1
            if self.pair and self.weight == "数":
                x = self.both(si, t)
                if x is not None:
                    self.acc[w] = x.copy() if w not in self.acc else self.acc[w] + x
                    self.n_add[w] = self.n_add.get(w, 0) + 1
                    self.tot = x.astype(np.complex128) if self.tot is None else self.tot + x
                    self.n_tot += 1
        self._v.clear()

    def hear_line(self, words):
        """1 行の語（席は見ない。bind="なし"・weight="数" の時だけ）。行の中のそれぞれの語に、同じ行のほかの語の元の向きを全部足す。"""
        assert self.bind == "なし" and self.weight == "数"
        words = list(dict.fromkeys(words))
        bs = [self._base(w) for w in words]
        have = [b for b in bs if b is not None]
        if len(words) < 2 or not have:
            return
        swell = np.sum(have, axis=0)
        for w, b in zip(words, bs):
            k = len(have) - (b is not None)
            if k == 0:
                continue
            x = swell if b is None else swell - b
            self.acc[w] = x.copy() if w not in self.acc else self.acc[w] + x
            self.n_add[w] = self.n_add.get(w, 0) + k

    def both(self, si, tri):
        """隠す席 si から見た、ほかの 2 語を 1 つに束ねた向き（周波数ごとに掛け合わせて位相を足し、長さ 1 にそろえ、席ごとの回し方で回す）。"""
        o = [self._base(tri[j]) for j in range(3) if j != si]
        if o[0] is None or o[1] is None:
            return None
        x = o[0] * o[1]
        n = float(np.linalg.norm(x))
        return None if n == 0 else self.rot(x / n, len(PAIRS) + si)

    def rot(self, b, pair):
        if self.bind == "なし":
            return b
        p = pair if isinstance(pair, int) else PAIRS.index(tuple(pair))
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

    def unrot(self, x, pair):
        """rot の逆（回し戻す）。"""
        if self.bind == "なし":
            return x
        p = pair if isinstance(pair, int) else PAIRS.index(tuple(pair))
        if self.bind == "ずらす":
            return np.roll(x, -(p + 1) * (len(x) // 7))
        self.rot(x, p)                                    # 回し方を作っておく
        return x * np.conj(self._ph[(p, len(x))])

    def of(self, w):
        if self.weight == "数":
            v = self.acc.get(w)
            if v is None or not self.center:
                return v
            return (v - self.n_add[w] * (self.tot / self.n_tot)).astype(np.complex64)
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
    def __init__(self, sv, online=False, norm="cos", side="候補"):
        assert not online or sv.weight == "数"
        assert norm in ("cos", "半分", "なし") and side in ("候補", "見せた語", "両方")
        self.sv, self.online, self.norm, self.side = sv, online, norm, side
        self.rec = []

    def tick(self, n=1):
        pass

    def learn(self, tri):
        if self.online:
            self.sv.hear(tri)

    def _from_shown(self, si, tri, cands):
        """見せた語の作った向きを回し戻して、隠した席に来る語の元の向きを言い当て、候補の元の向きと比べる（聞いた事の無い候補も比べられる）。"""
        guess = None
        for sj in range(3):
            if sj == si:
                continue
            v = self.sv.of(tri[sj])
            if v is None:
                continue
            x = self.sv.unrot(v, (sj, si))
            n = float(np.linalg.norm(x))
            if n > 0:
                guess = x / n if guess is None else guess + x / n
        s = {}
        if guess is not None:
            ng = float(np.linalg.norm(guess))
            for c in cands:
                b = self.sv._base(c)
                if b is not None and ng > 0:
                    s[c] = float(np.real(np.vdot(guess, b))) / ng
        return s

    def ask(self, si, tri, cands):
        cands = list(cands)
        if self.side != "候補":
            s2 = self._from_shown(si, tri, cands)
            if self.side == "両方":
                keep, self.side = self.side, "候補"
                s1 = self.ask(si, tri, cands)["点"]
                self.side = keep
                self.rec.pop()
                s2 = {c: s1.get(c, 0.0) + s2.get(c, 0.0) for c in cands if c in s1 or c in s2}
            out = dict(段="答えられない", 点={}, 答え=None)
            if s2 and max(s2.values()) > 0:
                out = dict(段="解く", 点=s2, 答え=max(s2, key=lambda c: (s2[c], -cands.index(c))))
            self.rec.append((0, si, tuple(tri[j] for j in range(3) if j != si), out["段"], out["答え"]))
            return out
        aim = None
        for sj in range(3):
            if sj == si:
                continue
            b = self.sv._base(tri[sj])
            if b is not None:
                x = self.sv.rot(b, (si, sj))
                aim = x if aim is None else aim + x
        if self.sv.pair:
            x = self.sv.both(si, tri)
            if x is not None:
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
                        div = nv if self.norm == "cos" else (nv ** 0.5 if self.norm == "半分" else 1.0)
                        s[c] = float(np.real(np.vdot(aim, v))) / (na * div)
            if s and max(s.values()) > 0:
                best = max(s, key=lambda c: (s[c], -cands.index(c)))
                out = dict(段="解く", 点=s, 答え=best)
        self.rec.append((0, si, tuple(tri[j] for j in range(3) if j != si), out["段"], out["答え"]))
        return out
