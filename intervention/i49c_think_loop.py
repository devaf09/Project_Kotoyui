"""I49c: 自発的に話す思考ループの最小版（持ち主の案、未確定）。器（I49a の Vessel）の上に乗せる。会話にはまだつながない。

ねらい: 入力を外から押し込まれるのでなく、ループが自分で「次の語を読む」「考える」「声に出す」「待つ」を選ぶ形を先に作る。
実験段階なので、観察と調整の口を付ける。最後に取り外して、後から調整しにくくできる（seal）。

1 歩（step）= 器の 1 刻み。指揮者の表が、項目ごとの「したさ」を出し、いちばん大きい項目を 1 つだけ行う。
  次へ    読み手に語が残っていれば、次の 1 語を読む（hear）。ベクトルの無い語は、時間だけ進む。
          したさ = 0.5 − 動きの量（波の大きさの平均 / 上限）。内面が静かなら読みたく、沸いていれば読まない。
  考える  今の内面の全体（全部の波を足した向き）を、自分に戻す（settle）。声には出さない。
          したさ = 内面のばらつき − いつもの外れ（いつも聞く物より、内面の方がまとまっていなければ、考えたい）
  出力    考えるのと同じ事をし、その向きを外へも出す（声に出した考え）。
          したさ = 感度の平均（log2）×（前に言った事からの離れ − 内面のばらつき）
          静かで刺激が足りず、しかも前に言った事と今の内面が、内面のばらつき以上に離れた時だけ話す。同じ事はくり返さない。
  待つ    何もしない（時間だけ進む）。したさ = 0。
  内面のばらつき = 全体の向きを、波それぞれが言い当てた時の外れの平均（速い波と遅い波が違う事を考えていると大きい）。
  前に言った事からの離れ = 1 − |cos(前に出した向き, 今の全体)|。まだ何も言っていなければ 1。
  同点の時は、表の上の項目を選ぶ（次へ・考える・出力・待つ の順）。
  この式は暫定（様子見の物）。したさは全部記録するので、後から式を替えて比べられる。

手で置く物:
  run() の budget（歩数の上限、止め役）
  つまみ（項目ごとの下駄 bias。初めは全部 0。実験の間だけ動かせる）
  表の並び順（同点の時の順）
したさの式に置いた数は 0.5（器の「中くらい」と同じ値）と 0 だけ。ほかは器が既に持つ量（大きさ・いつもの外れ・感度）。

観察と調整（実験用の部品。seal() で外す）:
  attach(node)   1 歩ごとに node.after(loop, rec) が呼ばれる。rec は、選んだ項目・全部のしたさ・読んだ語・出した向き。
  set(name, v)   つまみを動かす（走っている途中でもよい）。動かした事は記録に残る。
  seal()         部品を全部外し、つまみを今の値で固める。以後 attach と set は使えない。記録（log）は残る。

乱数は使わない。同じ語の列と同じつまみなら、同じ記録になる。

python intervention/i49c_think_loop.py --demo out.npz   # 自作の語で動きを見るだけの小さな試し
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i49a_inner_vessel import Vessel  # noqa: E402

ITEMS = ("次へ", "考える", "出力", "待つ")


class Words:
    """読み手。語を 1 つずつ渡す。語 = (名前, ベクトル または None, 印の dict)。後から足せる（push）。"""

    def __init__(self, words=()):
        self.q = list(words)
        self.i = 0

    def push(self, words):
        self.q.extend(words)

    def has_next(self):
        return self.i < len(self.q)

    def next(self):
        w = self.q[self.i]
        self.i += 1
        return w if len(w) == 3 else (w[0], w[1], {})


class Node:
    """実験用の部品の型。after が 1 歩ごとに呼ばれる。"""

    def after(self, loop, rec):
        pass


class Names(Node):
    """観察用: 読んだ語を覚えておき、出力の向きにいちばん近い語の名前を付ける（声の中身を人が読むためだけの物）。"""

    def __init__(self):
        self.names, self.vecs, self.said = [], [], []

    def after(self, loop, rec):
        if rec["項目"] == "次へ" and rec["ベクトル"] is not None and rec["語"] not in self.names:
            v = np.asarray(rec["ベクトル"], dtype=np.complex64)
            self.names.append(rec["語"])
            self.vecs.append(v / np.linalg.norm(v))
        if rec["項目"] == "出力" and self.vecs:
            c = np.abs(np.conj(np.array(self.vecs)) @ rec["出した向き"])
            k = int(c.argmax())
            self.said.append((rec["歩"], self.names[k], round(float(c[k]), 3)))


class Loop:
    def __init__(self, vessel, words=None):
        self.v = vessel
        self.words = words if words is not None else Words()
        self.bias = {k: 0.0 for k in ITEMS}                         # つまみ（下駄）
        self.nodes = []
        self.sealed = False
        self.n = 0
        self.out = []                                               # 外へ出した物: (歩, 向き)
        self.log = dict(item=[], drive=[], spread=[], word=[], knob=[])

    # ---- 実験用の口 ----
    def attach(self, node):
        if self.sealed:
            raise RuntimeError("固めた後は、部品を付けられない")
        self.nodes.append(node)
        return node

    def set(self, name, value):
        if self.sealed:
            raise RuntimeError("固めた後は、つまみを動かせない")
        if name not in self.bias:
            raise KeyError(name)
        self.bias[name] = float(value)
        self.log["knob"].append((self.n, name, float(value)))

    def seal(self):
        self.nodes = []
        self.sealed = True
        self.bias = dict(self.bias)

    # ---- 指揮者 ----
    def spread(self, whole=None):
        """内面のばらつき: 全体の向きを、波それぞれが言い当てた時の外れの平均。全部 0 なら 0。"""
        whole = self.v.whole() if whole is None else whole
        if whole is None:
            return 0.0
        n = self.v.act()
        return float(self.v.guess(whole)[n > 0].mean())

    def drives(self):
        v = self.v
        whole = v.whole()
        sp = self.spread(whole)
        ok = whole is not None
        hungry = 0.5 - float((v.act() / v.cap).mean())
        apart = 1.0 if not (ok and self.out) else 1.0 - float(np.abs(np.vdot(self.out[-1][1], whole)))
        d = {"次へ": hungry if self.words.has_next() else -np.inf,
             "考える": sp - float(v.usual.mean()) if ok else -np.inf,
             "出力": float(np.log2(v.gain).mean()) * (apart - sp) if ok else -np.inf,
             "待つ": 0.0}
        return {k: d[k] + self.bias[k] for k in ITEMS}, sp, whole

    def step(self):
        d, sp, whole = self.drives()
        item = max(ITEMS, key=lambda k: d[k])                       # 同点は表の上の項目（max は初めの物を返す）
        rec = dict(歩=self.n, 項目=item, したさ=d, ばらつき=sp, 語=None, ベクトル=None, 印={}, 出した向き=None)
        if item == "次へ":
            name, x, marks = self.words.next()
            rec.update(語=name, ベクトル=x, 印=marks)
            if x is None:
                self.v.tick()
            else:
                self.v.hear(x)
        elif item == "考える":
            self.v.settle(whole)
        elif item == "出力":
            self.out.append((self.n, whole))
            rec["出した向き"] = whole
            self.v.settle(whole)
        else:
            self.v.tick()
        self.log["item"].append(ITEMS.index(item))
        self.log["drive"].append([d[k] for k in ITEMS])
        self.log["spread"].append(sp)
        self.log["word"].append(rec["語"])
        self.n += 1
        for node in self.nodes:
            node.after(self, rec)
        return rec

    def run(self, budget):
        """budget 歩で必ず止まる。待つを選び、器が空で、読む語も無ければ早く止まる。返り値は進んだ歩数。"""
        k = 0
        while k < int(budget):
            rec = self.step()
            k += 1
            if rec["項目"] == "待つ" and not self.v.S.any() and not self.words.has_next():
                break
        return k

    # ---- 記録 ----
    def arrays(self):
        a = self.v.arrays()
        a.update(item=np.array(self.log["item"], dtype=np.int8),
                 drive=np.array(self.log["drive"], dtype=np.float32).reshape(-1, len(ITEMS)),
                 spread=np.array(self.log["spread"], dtype=np.float32),
                 out_step=np.array([s for s, _ in self.out], dtype=np.int64))
        return a

    def counts(self):
        it = np.array(self.log["item"], dtype=np.int64)
        return {k: int((it == i).sum()) for i, k in enumerate(ITEMS)}


def demo_words(dim=64):
    """自作の語: 話題 A の語 4 つと話題 B の語 4 つ（話題の中では向きが近い）。A を 3 回り、B を 3 回り、A を 1 回り。乱数なし。
    試運転では、この 3 つのかたまりを、間を空けて読み手に足す。"""
    def vec(topic, k):
        x = np.zeros(dim, dtype=np.complex64)
        x[topic * (dim // 2): (topic + 1) * (dim // 2)] = 1.0       # 話題の共通の向き
        x[topic * (dim // 2) + k] += 6.0                            # 語ごとの向き
        return x / np.linalg.norm(x)
    A = [(f"A{k}", vec(0, k)) for k in range(4)]
    B = [(f"B{k}", vec(1, k)) for k in range(4)]
    return A * 3 + B * 3 + A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="記録の npz の出力先")
    ap.add_argument("--bands", type=int, default=20)
    ap.add_argument("--budget", type=int, default=300, help="3 回目の語の後に回す歩数の上限")
    a = ap.parse_args()
    w = demo_words()
    loop = Loop(Vessel(64, bands=a.bands), Words(w[:12]))           # 語は 3 回に分けて届く（相手が話す・黙る、の代わり）
    names = loop.attach(Names())
    n = loop.run(60)
    loop.words.push(w[12:24])
    n += loop.run(60)
    loop.words.push(w[24:])
    n += loop.run(a.budget)
    line = [w if w is not None else ITEMS[i][0] for i, w in zip(loop.log["item"], loop.log["word"])]
    print("歩", n, "内訳", loop.counts())
    print("流れ（語 = 読んだ、考 = 考える、出 = 出力、待 = 待つ）:")
    print(" ".join(line[:150]), "…" if len(line) > 150 else "")
    print("声に出した物（歩, いちばん近い語, 近さ）:", names.said[:40])
    print("見張り", dict(張り付き=int(loop.v.watch["張り付き"].sum()), 死んだ=loop.v.watch["死んだ"], 刻み=loop.v.watch["刻み"]))
    np.savez_compressed(a.demo, **loop.arrays())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
