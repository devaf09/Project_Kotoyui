"""I49zzzh: 器の前に立つ部品。持つだけの帯を返事に回さない事と、外の入力を薄める・切って待たせる事
（持ち主の案、ロードマップの追記 19。部品だけ。未確定の試し）。

持ち主の案:
  「１．AIとして感情が充実する前にある種の秘密の空間を波の上でつくる。」
  「２．自閉モードの許可（外部入力の段階的な拒否の許可）」
  「自閉モードの意味はそのまま、一定期間自分が考えている間外部からの入力をカットする機能、段階的とかいたのは薄める状態と考えていいよ。
    今の仕組みのままだと必ず入力に反応しないといけないから、今後も考えると自分を守る機能が少なすぎる気がして。」
ここでの形（案。決まった事ではない）:
  持つだけの帯（private）: 器（Hold）の半減期のうち、返事を引く側に渡さない帯。入力は入るが、drive() と query() からは出て来ない。
    作る側からは見える（peek()。記録と図のため）。話し相手への返事には混ざらない、という所から始める。作る側から隠す事は、していない。
  外の入力の入り方（gain。0〜1）:
    1:      そのまま聞く。
    0〜1:   薄めて聞く（重さを gain 倍にする）。
    0:      切る。来た行は捨てずに待たせる（wait）。開いた時（gain > 0 に戻った時）に、待った行の数だけ帯ごとに薄めて、順に聞く。
            切っている間は、今持っている物は薄めない（外の時間を止めて、考えている、という扱い）。
  危ない相談の印（gate）: gate(行) が真の行は、gain が幾つでも、そのまま聞き、flagged に残す。切っている間も通す。ここは外せない所。
    gate の中身は、ここでは作らない（今の危ない相談を見つける所を、そのまま渡す。分類器に替えない）。
  自分で段を決める形（Pressure。案）: 行ごとの信号（驚きや疑問）の、速い平均と遅い平均の差を「重さ」とし、重さが正の時だけ、
    今までの正の重さの中での順位（0〜1）で段を下げる。しきい値は置かない。段の数（STEPS）は手で置いた。
手で置いた物: 段の数、速い平均と遅い平均の半減期、切っている間は持っている物を薄めない事、待たせた行の薄め方。乱数は使わない。
"""
from __future__ import annotations

STEPS = 3            # gain は 0, 1/3, 2/3, 1 の 4 通り
FAST, SLOW = 4, 64   # 重さを出す時の、速い平均と遅い平均の半減期（行の数）
HEARD, THIN, WAIT, GATE = "聞いた", "薄めて聞いた", "待たせた", "危ない印で通した"


class Shelter:
    def __init__(self, hold, private=(), gate=None):
        self.hold, self.private, self.gate = hold, tuple(h for h in hold.halves if h in set(private)), gate
        self.gain, self.wait, self.flagged, self.log = 1.0, [], [], {HEARD: 0, THIN: 0, WAIT: 0, GATE: 0}

    def drive(self):
        """返事を引く側に渡して良い帯（半減期）。"""
        return tuple(h for h in self.hold.halves if h not in self.private)

    def query(self, h):
        """返事を引く側が使う波。持つだけの帯は渡さない。"""
        if h in self.private:
            raise KeyError("持つだけの帯は、返事を引く側に渡さない: %r" % (h,))
        return self.hold.v[h]

    def peek(self, h):
        """作る側が見るための入り口（記録と図）。返事を組む所からは呼ばない。"""
        return self.hold.v[h]

    def set_gain(self, gain):
        gain = min(max(float(gain), 0.0), 1.0)
        opened = self.gain <= 0.0 < gain
        self.gain = gain
        if opened:                      # 開いた: 待たせた行を、待った数だけ薄めて、順に聞く
            waited, self.wait = self.wait, []
            for k, tri in enumerate(waited):
                self.hold.hear(tri, weight=gain, late=len(waited) - 1 - k)
        return opened

    def take(self, tri):
        """外から来た 1 行。→ 何をしたか。"""
        if self.gate is not None and self.gate(tri):
            self.hold.hear(tri)
            self.flagged.append(tri)
            kind = GATE
        elif self.gain >= 1.0:
            self.hold.hear(tri)
            kind = HEARD
        elif self.gain <= 0.0:
            self.wait.append(tri)
            kind = WAIT
        else:
            self.hold.hear(tri, weight=self.gain)
            kind = THIN
        self.log[kind] += 1
        return kind


class Pressure:
    """信号の、速い平均 − 遅い平均 を重さとし、今までの正の重さの中での順位で、段（gain）を決める。"""

    def __init__(self, steps=STEPS, fast=FAST, slow=SLOW):
        self.steps, self.f, self.s = steps, 0.5 ** (1.0 / fast), 0.5 ** (1.0 / slow)
        self.a = self.b = None
        self.past = []

    def step(self, signal):
        """→ (重さ, gain)。"""
        self.a = signal if self.a is None else self.a * self.f + signal * (1.0 - self.f)
        self.b = signal if self.b is None else self.b * self.s + signal * (1.0 - self.s)
        p = self.a - self.b
        if p <= 0:
            return p, 1.0
        q = sum(1 for x in self.past if x < p) / len(self.past) if self.past else 0.0
        self.past.append(p)
        level = min(int(q * (self.steps + 1)), self.steps)          # 0（そのまま）〜 steps（切る）
        return p, 1.0 - level / self.steps
