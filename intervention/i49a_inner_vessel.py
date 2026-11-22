"""I49a: 内面の骨組み（意識の器）の最小版。会話にはまだつながない（持ち主の案、未確定）。

ねらい: 入力→出力の道具とは別に、入力が無い時も時間が進み、自分の状態を持ち続ける「器」の外形だけを先に作る
（ドライバの作り方: open / read に当たる最低限の口を先に置き、中身は後から入れ替える）。

口は 4 つ:
  tick()        時間が 1 刻み進む（入力なし）。波は薄れ、感度はゆっくり戻る。
  hear(x)       入力（1 本のベクトル）を聞く。聞く前に、今の状態で入力を言い当て、外れ（驚き）を測る。
  want()        今の状態から読める量を返す。したい事の候補はまだ決めない（名札は振る舞いを見てから）。
  settle(x)     自分の出した結果を一度戻す（聞くのと同じ道を通し、出どころだけ別に記録する）。

中身（虹）:
  少しずつ違う速さの波を N 本並べる。k 本目は半減期 h_k 刻み（1 から longest まで等比）で薄れる。
  どの速さが要るかは先に決めず、全部回して記録し、後から「その波で次の振る舞いが読めるか」で選ぶ。名札は付けない。

自分でする調整（目当ては 2 つ、どちらも正解が要らない）:
  言い当て      聞く前の状態と入力の向きの近さ。外れ（驚き）= 1 - |cos|。いつもの外れより大きいと感度が上がる。
  中くらいを保つ  波の大きさが上限の半分より小さいと感度が上がり、大きいと下がる。
  調整の速さは、その波自身の速さ a_k をそのまま使う（別の値を置かない）。使う計算は四則だけ。

手で置く物（止め役だけ）:
  cap       波の大きさの上限（越えたら縮める）
  gain_max  感度の上限と下限（gain_max と 1 / gain_max）
  longest   いちばん遅い波の半減期、bands = 波の本数
  think() の budget（入力なしで回す刻みの上限）
見張り（watch）は数えるだけで、直さない: 張り付き（上限で縮めた回数）と、死んだ（全部の波がほぼ 0 の刻み）。

記録: 刻みごとに、出どころ（無し・入力・自分）、波ごとの大きさ・感度・外れを全部残す（dump で npz）。
乱数は使わない。同じ入力の列なら同じ記録になる。

python intervention/i49a_inner_vessel.py --demo out.npz   # 自作の入力で動きを記録するだけの小さな試し
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

SRC = ("無し", "入力", "自分")


class Vessel:
    def __init__(self, dim, bands=20, longest=1024.0, cap=1.0, gain_max=16.0):
        self.D, self.N = int(dim), int(bands)
        h = np.power(float(longest), np.arange(self.N, dtype=np.float64) / max(1, self.N - 1))   # 半減期 1 … longest
        self.half = h.astype(np.float32)
        self.a = (1.0 - np.power(0.5, 1.0 / h)).astype(np.float32)                                # 1 刻みで入れ替わる割合
        self.cap, self.gmax = np.float32(cap), np.float32(gain_max)
        self.S = np.zeros((self.N, self.D), dtype=np.complex64)       # 波（虹）
        self.gain = np.ones(self.N, dtype=np.float32)                 # 感度
        self.usual = np.ones(self.N, dtype=np.float32)                # いつもの外れ（初めは「何も言い当てられない」= 1）
        self.last = np.ones(self.N, dtype=np.float32)                 # 直前の外れ
        self.t = 0
        self.log = dict(src=[], act=[], gain=[], sur=[])
        self.watch = dict(張り付き=np.zeros(self.N, dtype=np.int64), 死んだ=0, 刻み=0)

    # ---- 中の道具 ----
    def act(self):
        return np.linalg.norm(self.S, axis=1).astype(np.float32)

    def _balance(self):
        """中くらいを保つ: 大きさが上限の半分より小さければ感度を上げ、大きければ下げる。"""
        self.gain *= 1.0 + self.a * (0.5 - self.act() / self.cap)

    def _stop(self):
        n = self.act()
        over = n > self.cap
        if over.any():
            self.S[over] *= (self.cap / n[over])[:, None]
            self.watch["張り付き"][over] += 1
        np.clip(self.gain, 1.0 / self.gmax, self.gmax, out=self.gain)

    def _record(self, src, sur):
        n = self.act()
        if float(n.max()) < 1e-6 * float(self.cap):
            self.watch["死んだ"] += 1
        self.watch["刻み"] += 1
        self.log["src"].append(src)
        self.log["act"].append(n)
        self.log["gain"].append(self.gain.copy())
        self.log["sur"].append(sur.copy())
        self.t += 1

    # ---- 口 ----
    def tick(self):
        """入力なしで 1 刻み。波は薄れ、感度は静かなほど上がる（次の入力に敏感になる）。"""
        self.S *= (1.0 - self.a)[:, None]
        self._balance()
        self._stop()
        self._record(0, self.last)

    def guess(self, x):
        """今の状態で x を言い当てた時の外れ（波ごと）。状態は変えない。"""
        x = np.asarray(x, dtype=np.complex64).reshape(self.D)
        nx = float(np.linalg.norm(x))
        n = self.act()
        sur = np.ones(self.N, dtype=np.float32)
        ok = (n > 0) & (nx > 0)
        if ok.any():
            dot = np.abs(self.S[ok] @ np.conj(x))
            sur[ok] = 1.0 - np.minimum(1.0, dot / (n[ok] * nx))
        return sur

    def whole(self):
        """全部の波を足した向き（長さ 1）。全部 0 なら None。重みは掛けない（大きい波ほど多く入る）。"""
        s = self.S.sum(axis=0)
        n = float(np.linalg.norm(s))
        return (s / n).astype(np.complex64) if n > 0 else None

    def hear(self, x, src=1):
        x = np.asarray(x, dtype=np.complex64).reshape(self.D)
        sur = self.guess(x)                                          # 聞く前の状態で言い当てる
        self.gain *= 1.0 + self.a * (sur - self.usual)               # いつもより外れたら感度を上げる
        self.usual += self.a * (sur - self.usual)
        self._balance()
        np.clip(self.gain, 1.0 / self.gmax, self.gmax, out=self.gain)
        self.S = (1.0 - self.a)[:, None] * self.S + (self.a * self.gain)[:, None] * x[None, :]
        self.last = sur
        self._stop()
        self._record(src, sur)
        return sur

    def settle(self, x):
        """自分の出した結果を一度戻す。道は hear と同じで、出どころだけ「自分」と記録する。"""
        return self.hear(x, src=2)

    def think(self, budget):
        """入力なしで回す。budget 刻みで必ず止まる。全部の波が 0 なら早く止まる。返り値は回した刻みの数。"""
        k = 0
        while k < int(budget):
            if not self.S.any():
                break
            self.tick()
            k += 1
        return k

    def want(self):
        """今の状態から読める量。名札の付いた「したい事」はまだ返さない（候補は空）。"""
        n = self.act()
        fast, slow = self.S[0], self.S[-1]
        nf, ns = float(np.linalg.norm(fast)), float(np.linalg.norm(slow))
        gap = 1.0 - float(np.abs(np.vdot(slow, fast))) / (nf * ns) if nf > 0 and ns > 0 else 1.0
        return dict(動きの量=float(n.mean() / self.cap), 外れ=float(self.last.mean()), いつもの外れ=float(self.usual.mean()),
                    速い波と遅い波のずれ=gap, 感度=float(np.log2(self.gain).mean()), 候補=[])

    # ---- 記録 ----
    def arrays(self):
        return dict(src=np.array(self.log["src"], dtype=np.int8), act=np.array(self.log["act"], dtype=np.float32),
                    gain=np.array(self.log["gain"], dtype=np.float32), sur=np.array(self.log["sur"], dtype=np.float32),
                    half=self.half, 張り付き=self.watch["張り付き"], 死んだ=np.int64(self.watch["死んだ"]))

    def dump(self, path):
        np.savez_compressed(path, **self.arrays())


def demo_inputs(dim=64):
    """自作の入力: 話題 A を 60 刻み、黙って 60 刻み、話題 B を 60 刻み、黙って 200 刻み、また話題 A を 30 刻み。"""
    e = np.zeros((2, dim), dtype=np.complex64)
    e[0, : dim // 2] = 1.0 / np.sqrt(dim // 2)
    e[1, dim // 2:] = 1.0 / np.sqrt(dim - dim // 2)
    return [("A", 60), (None, 60), ("B", 60), (None, 200), ("A", 30)], dict(A=e[0], B=e[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="記録の npz の出力先")
    ap.add_argument("--bands", type=int, default=20)
    a = ap.parse_args()
    plan, vec = demo_inputs()
    v = Vessel(64, bands=a.bands)
    for name, n in plan:
        for _ in range(n):
            if name is None:
                v.tick()
            else:
                v.hear(vec[name])
        w = v.want()
        print(name or "（黙る）", n, {k: round(x, 3) for k, x in w.items() if k != "候補"}, flush=True)
    print("見張り", dict(張り付き=int(v.watch["張り付き"].sum()), 死んだ=v.watch["死んだ"], 刻み=v.watch["刻み"]))
    v.dump(a.demo)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
