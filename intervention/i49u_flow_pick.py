"""I49u: 並んだ上位の候補から、「話のそれまでの流れ」で選び直す（持ち主の言葉「内面の波はかなりいいかも」から。未確定の試し）。

I49t で、解く形は正解を上位に入れているが、点の並んだ候補から選ぶ所で落としていると分かった。
ここでは元の答え方（rel）は変えずに、上位の候補だけを、流れの波で並べ直す。乱数は使わない。
流れの波 = この話でそれまでに聞いた 3 つ組の語の、元の向き（base_of）を足した物。話が終わると空に戻す（tick）。
内面の波そのものではなく、その代わりに置いた物。

  FlowPick(rel, base_of, pick, top=3, minus_self=False)
    pick:
      "そのまま"        rel の答えのまま
      "上位"            点の上位 top 個の中で、流れの波に一番近い語
      "九割以内"        一位の 9 割以上の点の候補の中で、流れの波に一番近い語
      "順位を足す"      全候補で、点の順位と流れへの近さの順位を足して、一番小さい語
      "出た語"          点の上位 top 個の中で、この話で一番最近に出た語。どれも出ていなければ一位のまま（比べ役）
      "出た語・九割以内"  上と同じ事を、一位の 9 割以上の点の候補の中だけでする
      "出た語→板"        上位 top 個の中にこの話で出た語が有ればそれ。無ければ、一位の 9 割以上の候補の中で板に一番近い語
    minus_self: 候補自身がこの話で出た分を、流れの波から引いてから比べる（「もう出た語を選ぶだけ」になっていないかを見る）
    ask / learn / tick は rel と同じ形。picked は選び直しで答えが変わった数。
  I49z（仮の板、持ち主の案・未確定）で足した物:
    half: 3 つ組を half 個聞くと半分に薄れる（古い属性は消える）。None なら薄れない
    by_seat: 席ごとに別の板を持つ（その席に前に来た語の属性だけを足す）。問う時は、隠した席の板と比べる
    carry: 話が終わっても板を空に戻さない（薄れに任せる）
  I49za（行の折り畳みの決まりを板に入れる、持ち主の案・未確定）で足した物:
    usual: 板の属性を「ふだん」（今までに聞いた全部の語の属性の平均。席ごとの板なら、その席の平均）と比べる
      None              比べない
      "明るい分だけ"     ふだんより明るい分だけを残す（板の平均 − ふだん の、正の所）
      "残す・ほかは半分"  ふだんより明るい属性はそのまま、ほかは半分に薄める（記憶板 I40b の FADE と同じ形）
      "引く"            板からも候補からも、ふだんを引いてから比べる
    hear_usual(tri): ふだんだけに足す（先に聞かせた材料の分）
  I49ze（板の語を足さずに並べる。持ち主の言葉「並べて一斉に走査もOK」から。未確定の試し）で足した物:
    board: "足す"（今まで通り、板は 1 本の足した波） ／ "並べる"（聞いた語の波を 1 本ずつ並べて持ち、重さだけを薄れさせる）
      "並べる" の近さ = 並べた波を一斉に走査して、重さ × 候補との近さ の一番大きい物。
      重さが float32 で 0 になった波は、並びから外す（薄れ切った物。閾は置かない）。
  I49zf（板が答えを替え過ぎていないかを見る。未確定の試し）で足した物:
    gate: 板で選び直す時の関門
      None          今まで通り、いつも板に一番近い候補に替える
      "出た語"       候補の中に、この話で出た語が有る時だけ、板で選び直す（決まりで書いた比べ役）
      "同じ波"       板に同じ波が並んでいる候補だけの中から、板で選ぶ。無ければ一位のまま（board="並べる" の時だけ）
                    同じ波 = 近さが 1（float32 の丸めの分だけ下まで）。語の表は引かない。
      "同じ波・重さ"  上と同じ関門。選ぶ時も、似た波の分は見ず、同じ波の重さ（一番新しい分）だけで比べる。同じ重さなら点の順
手で置く物: 上位の数 top、九割、流れを足す重さ 1、薄れる速さ half、話ごとに空に戻すかどうか、「ほかは半分」の半分。
"""
from __future__ import annotations

import numpy as np


class FlowPick:
    def __init__(self, rel, base_of, pick="上位", top=3, minus_self=False, half=None, by_seat=False, carry=False, usual=None, board="足す", gate=None):
        assert not (minus_self and (half is not None or by_seat or carry or usual))
        assert board in ("足す", "並べる") and not (board == "並べる" and (usual or minus_self))
        self.board = board
        assert gate in (None, "出た語", "同じ波", "同じ波・重さ") and not (gate in ("同じ波", "同じ波・重さ") and board != "並べる")
        self.gate = gate
        assert usual in (None, "明るい分だけ", "残す・ほかは半分", "引く")
        self.usual = usual
        self.u_sum, self.u_n = [None, None, None, None], [0, 0, 0, 0]      # 席 3 つ + 全部
        self.rel, self.base_of, self.pick, self.top, self.minus_self = rel, base_of, pick, top, minus_self
        self.keep = None if half is None else float(0.5 ** (1.0 / half))
        self.by_seat, self.carry = by_seat, carry
        self.picked = 0
        self._reset()

    def _reset(self):
        self.ctx, self.said, self.t = None, {}, 0
        self.seat_ctx = [None, None, None]
        self.w, self.seat_w = 0.0, [0.0, 0.0, 0.0]
        self._buf, self._bw, self._bs, self._lo, self._hi = None, None, None, 0, 0     # 並べた波（古い順）、重さ、席

    @property
    def rows(self):
        return [] if self._buf is None else self._buf[self._lo:self._hi]

    @property
    def row_w(self):
        return [] if self._buf is None else [float(v) for v in self._bw[self._lo:self._hi]]

    def _push(self, v, sj):
        if self._buf is None:
            self._buf = np.zeros((64, v.shape[0]), dtype=v.dtype)
            self._bw, self._bs = np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.int8)
        if self._hi == self._buf.shape[0]:
            n = self._hi - self._lo
            if n > self._buf.shape[0] // 2:                 # 生きている波が多ければ、置き場を倍にする
                for name in ("_buf", "_bw", "_bs"):
                    old = getattr(self, name)
                    new = np.zeros((old.shape[0] * 2,) + old.shape[1:], dtype=old.dtype)
                    new[:n] = old[self._lo:self._hi]
                    setattr(self, name, new)
            else:                                           # 薄れ切った分の場所を詰める
                for a in (self._buf, self._bw, self._bs):
                    a[:n] = a[self._lo:self._hi].copy()
            self._lo, self._hi = 0, n
        self._buf[self._hi], self._bw[self._hi], self._bs[self._hi] = v, 1.0, sj
        self._hi += 1

    def same_weight(self, c, si=None):
        """板に並んだ波のうち、候補と同じ波の重さの一番大きい物。無ければ None。"""
        b = self.base_of(c)
        if b is None or self._hi == self._lo:
            return None
        nb = float(np.linalg.norm(b))
        if nb <= 0:
            return None
        sl = slice(self._lo, self._hi)
        same = np.real(self._buf[sl] @ np.conj(b)) / nb >= 1.0 - 1e-4
        if self.by_seat and si is not None:
            same &= self._bs[sl] == si
        return float(self._bw[sl][same].max()) if same.any() else None

    def same_on_board(self, c, si=None):
        return self.same_weight(c, si) is not None

    def near_rows(self, c, si=None):
        b = self.base_of(c)
        if b is None or self._hi == self._lo:
            return None
        nb = float(np.linalg.norm(b))
        if nb <= 0:
            return None
        sl = slice(self._lo, self._hi)
        v = np.real(self._buf[sl] @ np.conj(b)) / nb * self._bw[sl]
        if self.by_seat and si is not None:
            v = v[self._bs[sl] == si]
        return float(v.max()) if v.size else None

    def hear_usual(self, tri):
        for sj, w in enumerate(tri):
            b = self.base_of(w)
            if b is not None:
                for k in (sj, 3):
                    self.u_sum[k] = b.copy() if self.u_sum[k] is None else self.u_sum[k] + b
                    self.u_n[k] += 1

    def near(self, c, si=None):
        if self.board == "並べる":
            return self.near_rows(c, si)
        b = self.base_of(c)
        seat = self.by_seat and si is not None
        x = self.seat_ctx[si] if seat else self.ctx
        if b is None or x is None:
            return None
        if self.usual:
            k = si if seat else 3
            wsum = self.seat_w[si] if seat else self.w
            if self.u_sum[k] is None or wsum <= 0:
                return None
            u = np.real(self.u_sum[k]) / self.u_n[k]
            x = np.real(x) / wsum
            if self.usual == "明るい分だけ":
                x = np.maximum(0.0, x - u)
            elif self.usual == "残す・ほかは半分":
                x = np.where(x > u, x, 0.5 * x)
            else:
                x, b = x - u, np.real(b) - u
            nx, nb = float(np.linalg.norm(x)), float(np.linalg.norm(b))
            if nx <= 1e-9 or nb <= 1e-9:
                return None
            return float(np.real(np.vdot(x, b))) / (nx * nb)
        if self.minus_self and c in self.said:
            x = x - self.said[c][0] * b
        nx, nb = float(np.linalg.norm(x)), float(np.linalg.norm(b))
        if nx <= 1e-6 or nb <= 0:
            return None
        return float(np.real(np.vdot(x, b))) / (nx * nb)

    def ask(self, si, tri, cands):
        o = self.rel.ask(si, tri, cands)
        pts = o["点"]
        if self.pick == "そのまま" or o["答え"] is None or not pts:
            return o
        cands = list(cands)
        order = sorted(pts, key=lambda c: (-pts[c], cands.index(c)))
        near90 = [c for c in order if pts[c] >= 0.9 * pts[order[0]]]
        if self.pick in ("九割以内", "出た語・九割以内"):
            pool = near90
        elif self.pick == "順位を足す":
            pool = order
        else:
            pool = order[:self.top]
        if self.pick in ("出た語", "出た語・九割以内") or (self.pick == "出た語→板" and any(c in self.said for c in pool)):
            best = max(pool, key=lambda c: (self.said.get(c, (0, -1))[1], -order.index(c)))
        else:
            if self.pick == "出た語→板":
                pool = near90
            if self.gate == "出た語" and not any(c in self.said for c in pool):
                return o
            if self.gate in ("同じ波", "同じ波・重さ"):
                pool = [c for c in pool if self.same_on_board(c, si)]
                if not pool:
                    return o
            if self.gate == "同じ波・重さ":
                nr = {c: self.same_weight(c, si) for c in pool}
            else:
                nr = {c: self.near(c, si) for c in pool}
            if all(v is None for v in nr.values()):
                return o
            lo = min(v for v in nr.values() if v is not None) - 1.0
            nr = {c: (lo if v is None else v) for c, v in nr.items()}
            if self.pick == "順位を足す":
                by = sorted(pool, key=lambda c: (-nr[c], order.index(c)))
                best = min(pool, key=lambda c: (order.index(c) + by.index(c), order.index(c)))
            else:
                best = max(pool, key=lambda c: (nr[c], -order.index(c)))
        if best != o["答え"]:
            self.picked += 1
            o = dict(o, 答え=best)
        return o

    def learn(self, tri):
        self.rel.learn(tri)
        self.t += 1
        if self.keep is not None:
            if self.ctx is not None:
                self.ctx = self.ctx * self.keep
            self.seat_ctx = [None if x is None else x * self.keep for x in self.seat_ctx]
            self.w *= self.keep
            self.seat_w = [v * self.keep for v in self.seat_w]
            if self._hi > self._lo:
                self._bw[self._lo:self._hi] *= np.float32(self.keep)
                while self._lo < self._hi and self._bw[self._lo] <= 0.0:      # 古い順なので、前から外れる
                    self._lo += 1
        for sj, w in enumerate(tri):
            b = self.base_of(w)
            if b is not None:
                self.ctx = b.copy() if self.ctx is None else self.ctx + b
                self.seat_ctx[sj] = b.copy() if self.seat_ctx[sj] is None else self.seat_ctx[sj] + b
                self.w += 1.0
                self.seat_w[sj] += 1.0
                if self.board == "並べる":
                    nb = float(np.linalg.norm(b))
                    if nb > 0:
                        self._push(b / nb, sj)
            n = self.said.get(w, (0, 0))[0]
            self.said[w] = (n + 1, self.t)
        if self.usual:
            self.hear_usual(tri)

    def tick(self):
        self.rel.tick()
        if not self.carry:
            self._reset()
        else:
            self.said = {}                                # この話で出た語は、話ごとに空に戻す
