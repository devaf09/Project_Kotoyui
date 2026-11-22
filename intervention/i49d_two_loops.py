"""I49d: 思考のループと、感情・状態のループを分けた最小版（持ち主の案、未確定）。会話にはまだつながない。

I49c では、指揮者が感情・状態の量（動きの量・感度）だけで「読む・話す」を決めたので、中身と無関係な一定の調子になった。
ここでは 2 つに分ける。
  感情・状態のループ  器（I49a の Vessel）。時間で流れる。思考のループの 1 歩ごとに 1 刻み進む。
  思考のループ        入れ物（ばらばらの材料置き場）を持ち、指揮者が違和感を見て、回す・組立・出力を選ぶ。

思考のループの部品:
  入れ物 Tray    材料を種類つきで置く: 語（聞いた語）、連想（連想で見つかった語）、印（過去・進行・丁寧・文の終わり など）。
  連想 near      指揮者の手前で働く。向きが近い物を引く口（near(vec, k) → [(名前, ベクトル, 近さ)]）。本体は差し込み。
  違和感         指揮者が持つ。名前つきの数で、全部記録する。
    材料が不足   入れ物の中身を足した向きと、器の内面の全体の向きのずれ（1 − |cos|）。入れ物が空なら 1、器が空なら 0。
                 「内面にある事を、入れ物の材料でまだ言い表せていない」。足しても減らなくなったら、消えたとみなす（閾値は置かない）。
    組立の違和感 組立の口が返す（名前 → 説明）。空なら違和感なし。
  組立 build     入れ物の中身から行を組む口（build(tray, vessel, said) → (行, 違和感の dict)）。本体は差し込み。
                 仮の組立（Plain）は、材料を内面に近い順に並べるだけ。違和感は 3 つ: 途中（聞いた語があるのに文の終わりの印が無い）、
                 材料が少ない（ベクトルを持つ材料が 2 つ未満）、もう言った（前に出した行と同じ顔ぶれ）。
                 「もう言った」のまま次の語が来たら、読む前に入れ物を空にする（言ったばかりの物を持ち越さない）。

  違和感の波 OddWave  違和感を波で持つ置き場（持つだけ。どこにもつながず、動きは変えない）。種類ごとに 1 本の向き、強さは振幅、
                 時間で薄れる。波の速さは器と同じ。tally() で、どの動きの後に下がるか、器のどの波と一緒に動くかを数える。

  組立の口 Scored  行は外の行づくり（tools/i49d_odd.OddScorer と同じ形の物）で詰め物を入れて作る。行づくりの違和感は流れに使わず、波で持つだけ。
  数の付いた印   読み手が印に数（float）を付けて渡すと（展開②の「まだ続けそうか」など）、その歩の違和感として波へ渡す（持つだけ）。

流れ（1 歩に 1 つ）:
  連想   入れ物か内面に近い物のうち、「材料が不足」をいちばん減らす物を 1 つ足す。減らす物が無ければ、次の段へ。
  組立   連想で減らなくなったら（= 不足が消えたら）組み立てる。
  出力   組立に違和感が無ければ、行を外へ出す。出した向きは自分にも戻り（自分の声を聞く）、入れ物は空になる。
  読む   組立に違和感があれば回しに戻る。新しい材料は次の語。語があれば 1 語読む（器も同じ語を聞く）。
  待つ   組立に違和感があり、読む語も無ければ「今は解決できない」として待つ（時間だけ進む）。
         語が届くか、内面が動いて連想で不足が減るようになれば、また回る。

手で置く物:
  run() の budget（歩数の上限）、連想で 1 回に引く数 k（初めは 5）、仮の組立の「2 つ未満」。
観察と調整（実験用。seal() で外す）は I49c と同じ形: attach(node)、set(name, v)（k を動かす）、seal()。
乱数は使わない。

python intervention/i49d_two_loops.py --demo out.npz   # 自作の語で動きを見るだけの小さな試し
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i49a_inner_vessel import Vessel  # noqa: E402
from i49c_think_loop import Node, Words  # noqa: E402,F401

ACTS = ("読む", "連想", "組立", "出力", "待つ")


def unit(x):
    x = np.asarray(x, dtype=np.complex64)
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else None


def apart(a, b):
    """向きのずれ 1 − |cos|。どちらかが無ければ None。"""
    if a is None or b is None:
        return None
    return 1.0 - min(1.0, float(np.abs(np.vdot(a, b))))


class Tray:
    """入れ物。材料 = dict(名前, ベクトル か None, 種類, 歩)。種類は 語・連想・印。"""

    def __init__(self):
        self.items = []

    def add(self, name, vec, kind, step):
        for it in self.items:                             # 同じ名前の材料は 1 つだけ。連想で入っていた物を聞いたら「語」に替える
            if it["名前"] == name and (it["種類"] == "印") == (kind == "印"):
                if kind == "語":
                    it.update(種類="語", 歩=step)
                return
        self.items.append(dict(名前=name, ベクトル=None if vec is None else unit(vec), 種類=kind, 歩=step))

    def clear(self):
        self.items = []

    def names(self, kind=None):
        return [it["名前"] for it in self.items if kind is None or it["種類"] == kind]

    def vecs(self):
        return [it for it in self.items if it["ベクトル"] is not None]

    def has_mark(self, name):
        return name in self.names("印")

    def whole(self, extra=None):
        v = [it["ベクトル"] for it in self.vecs()] + ([] if extra is None else [unit(extra)])
        return unit(np.sum(v, axis=0)) if v else None


class Near:
    """連想の口の見本: 決まった語の集まりから、向きが近い順に k 個引く。"""

    def __init__(self, names, vecs):
        self.names = list(names)
        self.V = np.array([unit(v) for v in vecs], dtype=np.complex64)

    def near(self, vec, k):
        c = np.abs(np.conj(self.V) @ unit(vec))
        order = np.argsort(-c, kind="stable")[: int(k)]
        return [(self.names[i], self.V[i], float(c[i])) for i in order]


class Plain:
    """仮の組立: 材料を内面に近い順に並べるだけ。違和感は 途中・材料が少ない・もう言った。"""

    def build(self, tray, vessel, said):
        whole = vessel.whole()
        mats = tray.vecs()
        if whole is not None:
            mats = sorted(mats, key=lambda it: -float(np.abs(np.vdot(it["ベクトル"], whole))))
        line = [it["名前"] for it in mats]
        odd = {}
        if tray.names("語") and not tray.has_mark("文の終わり"):
            odd["途中"] = "聞いた語があるのに、文の終わりの印が無い"
        if len(mats) < 2:
            odd["材料が少ない"] = "ベクトルを持つ材料が 2 つ未満"
        if said and sorted(said[-1][1]) == sorted(line):
            odd["もう言った"] = "前に出した行と同じ顔ぶれ"
        return line, odd


class Pick:
    """選ぶ口の見本（持ち主と話した「どれを選ぶか」、未確定）: 入れ物の材料から、重さの大きい順に k 個だけ残す。
    weight_of(名前) → 数（例: 会話の数えから出した「拾われやすさ」）。重さの無い語は、いちばん後ろ。k は手で置く数。"""

    def __init__(self, weight_of, k=3):
        self.weight_of, self.k = weight_of, int(k)

    def pick(self, mats):
        def w(it):
            v = self.weight_of(it["名前"])
            return -np.inf if v is None else float(v)
        keep = sorted(range(len(mats)), key=lambda i: (-w(mats[i]), i))[: self.k]
        return [mats[i] for i in sorted(keep)]


class Scored:
    """組立の口の 2 つ目: 仮の組立（Plain）の違和感で流れを決め、行は外の行づくり（scorer.line）で詰め物を入れて作る。
    scorer.line(keys, marks) → 間ごとの dict（詰め物、割合 = 種類 → いつもの値の並びの中の順位の割合 0〜1）。tools/i49d_odd.OddScorer と同じ形。
    行づくりが出す違和感（迷い・自信のなさ など）は、流れには使わない。felt（種類 → 行の中でいちばん大きい割合）に置き、
    思考のループが違和感の波へ渡す（持つだけ）。切る線（上位 1 割 など）は置かない。
    pick（選ぶ口）を渡すと、行づくりに渡す前に材料を選ぶ。語の順番は、聞いた順（連想で入った語は後ろ）。品詞は pos_of（名前 → 品詞）から。品詞の分からない語は行づくりに渡さない。"""

    def __init__(self, scorer, pos_of, mark_names=("過去", "否定", "丁寧", "進行", "問い"), pick=None):
        self.scorer, self.pos_of, self.mark_names = scorer, pos_of, tuple(mark_names)
        self.pick = pick                                  # 選ぶ口（無ければ全部使う）
        self.plain = Plain()
        self.felt = {}

    def build(self, tray, vessel, said):
        _, odd = self.plain.build(tray, vessel, said)
        mats = sorted(tray.vecs(), key=lambda it: (it["種類"] != "語", it["歩"]))
        mats = [it for it in mats if self.pos_of.get(it["名前"])]
        if self.pick is not None:
            mats = self.pick.pick(mats)                   # 並びは聞いた順のまま、選んだ物だけ残す
        keys = [(it["名前"], self.pos_of[it["名前"]]) for it in mats]
        rows = self.scorer.line(keys, [m for m in self.mark_names if tray.has_mark(m)]) if keys else []
        fill = {int(r["間"]): r["詰め物"] for r in rows}
        line = []
        for i, (w, _) in enumerate(keys):
            line.append(w)
            if fill.get(i):
                line.append(fill[i])
        self.felt = {}
        for r in rows:
            for k, v in r["割合"].items():
                self.felt[k] = max(self.felt.get(k, 0.0), float(v))
        if said and said[-1][1] == line:
            odd["もう言った"] = "前に出した行と同じ"
        return line, odd


class OddWave:
    """違和感を波で持つ置き場（持ち主の案、未確定）。持つだけで、どこにもつながない（動きは変えない）。
    種類ごとに 1 本の向き（出て来た順に、互いに直交する向きを割り当てる）。強さ = 振幅。起きた時に足し、時間で薄れる。
    波の速さ（半減期）は、感情・状態の器と同じ物を使う（新しい数を置かない）。速い波 = 今気になった、遅い波 = ずっと引っかかっている。"""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)          # 1 刻みで入れ替わる割合（器と同じ）
        self.kinds = []
        self.S = np.zeros((len(self.a), 0), dtype=np.float32)
        self.raw, self.hist = [], []

    def feel(self, odd):
        """odd: 名前 → 強さ（0〜1）。この 1 歩の違和感。無い種類は 0。"""
        for k in odd:
            if k not in self.kinds:
                self.kinds.append(k)
                self.S = np.concatenate([self.S, np.zeros((len(self.a), 1), dtype=np.float32)], axis=1)
        x = np.array([float(odd.get(k, 0.0)) for k in self.kinds], dtype=np.float32)
        self.S = (1.0 - self.a)[:, None] * self.S + self.a[:, None] * x[None, :]
        self.raw.append(dict(odd))
        self.hist.append(self.S.copy())

    def arrays(self):
        """(歩, 波, 種類) の振幅と、(歩, 種類) の元の強さ。途中で増えた種類は、前の歩を 0 で埋める。"""
        K = len(self.kinds)
        H = np.zeros((len(self.hist), len(self.a), K), dtype=np.float32)
        for t, h in enumerate(self.hist):
            H[t, :, : h.shape[1]] = h
        R = np.array([[float(r.get(k, 0.0)) for k in self.kinds] for r in self.raw], dtype=np.float32).reshape(len(self.raw), K)
        return H, R


class Thinker:
    def __init__(self, vessel, words=None, near=None, builder=None):
        self.v = vessel                                   # 感情・状態のループ
        self.words = words if words is not None else Words()
        self.near = near
        self.builder = builder if builder is not None else Plain()
        self.tray = Tray()
        self.knobs = dict(k=5.0)
        self.nodes, self.sealed = [], False
        self.n = 0
        self.said = []                                    # 外へ出した物: (歩, 行)
        self.tried = False                                # 今の入れ物で、連想がもう減らせないと分かっているか
        self.felt = {}
        self.odd = OddWave(vessel.a)                      # 違和感の波（持つだけ）
        self.log = dict(act=[], lack=[], tray=[], odd=[], word=[], knob=[])

    # ---- 実験用の口 ----
    def attach(self, node):
        if self.sealed:
            raise RuntimeError("固めた後は、部品を付けられない")
        self.nodes.append(node)
        return node

    def set(self, name, value):
        if self.sealed:
            raise RuntimeError("固めた後は、つまみを動かせない")
        if name not in self.knobs:
            raise KeyError(name)
        self.knobs[name] = float(value)
        self.log["knob"].append((self.n, name, float(value)))

    def seal(self):
        self.nodes, self.sealed = [], True

    # ---- 違和感 ----
    def lack(self, extra=None):
        """材料が不足: 入れ物の中身を足した向きと、内面の全体のずれ。入れ物が空なら 1、器が空なら 0。"""
        inner = self.v.whole()
        if inner is None:
            return 0.0
        d = apart(self.tray.whole(extra), inner)
        return 1.0 if d is None else d

    def _best_assoc(self):
        """連想の候補のうち、材料が不足をいちばん減らす物。減らす物が無ければ None。"""
        if self.near is None:
            return None
        inner = self.v.whole()
        query = self.tray.whole() if self.tray.vecs() else inner
        if query is None:
            return None
        have = set(self.tray.names())
        now, best = self.lack(), None
        for name, vec, sim in self.near.near(query, int(self.knobs["k"]) + len(have)):
            if name in have:
                continue
            d = self.lack(vec)
            if d < now and (best is None or d < best[0]):
                best = (d, name, vec, sim)
        return best

    def _read(self, rec, odd=None):
        name, x, marks = self.words.next()
        self.tray.add(name, x, "語", self.n)
        for m, on in marks.items():
            if on:
                self.tray.add(m, None, "印", self.n)
        self.felt = {m: float(v) for m, v in marks.items() if isinstance(v, float)}   # 数の付いた印（展開② など）は、違和感の波へ
        rec.update(した事="読む", 語=name, 印=marks, 違和感=odd or {}, _x=x)
        self.tried = False

    # ---- 1 歩 ----
    def step(self):
        rec = dict(歩=self.n, した事=None, 不足=None, 語=None, 印={}, 足した物=None, 行=None, 違和感={})
        self.felt = {}
        heard = None
        best = None if self.tried else self._best_assoc()
        if best is not None:                              # 連想で不足が減るうちは回す
            _, name, vec, sim = best
            self.tray.add(name, vec, "連想", self.n)
            rec.update(した事="連想", 足した物=(name, round(sim, 3)))
        elif not self.tried and not self.tray.items and self.words.has_next():
            self._read(rec)                               # 入れ物が空で、連想も無い時は、まず読む
            heard = rec.pop("_x")
        elif not self.tried:                              # 減らなくなった（= 不足が消えた）ので組立
            self.tried = True
            line, odd = self.builder.build(self.tray, self.v, self.said)
            rec.update(した事="組立", 行=line, 違和感=odd)
            self.pending = (line, odd)
            self.felt = dict(getattr(self.builder, "felt", {}))     # 組立の口が出した違和感の強さ（流れには使わず、波で持つだけ）
        else:
            line, odd = self.pending
            if not odd:                                   # 違和感なし → 出力
                self.said.append((self.n, line))
                rec.update(した事="出力", 行=line)
                heard = self.tray.whole()
                self.tray.clear()
                self.tried = False
            elif self.words.has_next():                   # 違和感あり → 回しに戻る。新しい材料は次の語
                if "もう言った" in odd:                    # 言った事しか無い入れ物は、次の語を読む前に空にする
                    self.tray.clear()
                self._read(rec, odd)
                heard = rec.pop("_x")
            else:                                         # 読む語も無い → 今は解決できないので待つ
                rec.update(した事="待つ", 違和感=odd)
                if self._best_assoc() is not None:        # 内面が動いて、連想で不足が減るようになったら、また回る
                    self.tried = False
        if heard is None:                                 # 感情・状態のループは、思考のループの 1 歩ごとに 1 刻み進む
            self.v.tick()
        elif rec["した事"] == "出力":
            self.v.settle(heard)
        else:
            self.v.hear(heard)
        rec["不足"] = self.lack()
        self.log["act"].append(ACTS.index(rec["した事"]))
        self.log["lack"].append(rec["不足"])
        self.log["tray"].append(len(self.tray.items))
        self.log["odd"].append(sorted(rec["違和感"]))
        self.log["word"].append(rec["語"])
        rec["感じた物"] = dict(self.felt)
        self.odd.feel(dict(self.felt, **{k: 1.0 for k in rec["違和感"]}, 材料が不足=rec["不足"]))
        self.n += 1
        for node in self.nodes:
            node.after(self, rec)
        return rec

    def run(self, budget):
        k = 0
        while k < int(budget):
            self.step()
            k += 1
        return k

    def counts(self):
        a = np.array(self.log["act"], dtype=np.int64)
        return {name: int((a == i).sum()) for i, name in enumerate(ACTS)}

    def tally(self):
        """違和感の波が、何と一緒に動くかの数え（つなぐ前の下調べ）。
        下がり方: した事ごとに、その歩での違和感（元の強さ）の変わり方の平均。負なら、その動きの後に下がった。
        器との動き: 違和感の元の強さと、感情・状態の器の波ごとの動きの量（同じ歩）の相関。動かない物は nan。"""
        H, R = self.odd.arrays()
        act = np.array(self.log["act"], dtype=np.int64)
        dR = np.diff(R, axis=0, prepend=R[:1])
        drop = {k: {n: (float(dR[act == i, j].mean()) if (act == i).any() else float("nan")) for i, n in enumerate(ACTS)}
                for j, k in enumerate(self.odd.kinds)}
        va = np.array(self.v.log["act"], dtype=np.float32)[-len(R):]
        co = {}
        for j, k in enumerate(self.odd.kinds):
            row = []
            for b in range(va.shape[1]):
                x, y = R[:, j], va[:, b]
                row.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan"))
            co[k] = row
        return dict(下がり方=drop, 器との動き=co)

    def arrays(self):
        a = self.v.arrays()
        H, R = self.odd.arrays()
        a.update(odd_wave=H, odd_raw=R, odd_kinds=np.array(self.odd.kinds))
        a.update(think_act=np.array(self.log["act"], dtype=np.int8), lack=np.array(self.log["lack"], dtype=np.float32),
                 tray=np.array(self.log["tray"], dtype=np.int32), said_step=np.array([s for s, _ in self.said], dtype=np.int64))
        return a


def demo_world(dim=64):
    """自作の語: 話題 A・B に 6 語ずつ（話題の中では向きが近い）。聞かせるのは 4 語ずつで、残りの 2 語ずつは連想でしか出て来ない。"""
    def vec(topic, k):
        x = np.zeros(dim, dtype=np.complex64)
        x[topic * (dim // 2): (topic + 1) * (dim // 2)] = 1.0
        x[topic * (dim // 2) + k] += 6.0
        return x / np.linalg.norm(x)
    vocab = {f"{t}{k}": vec(i, k) for i, t in enumerate("AB") for k in range(6)}

    def sent(names):
        return [(n, vocab[n], {"文の終わり": j == len(names) - 1}) for j, n in enumerate(names)]
    talk = [sent(["A0", "A1", "A2"]), sent(["A3", "A0"]), sent(["B0", "B1", "B2", "B3"]), sent(["A1", "A2"])]
    return vocab, talk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="記録の npz の出力先")
    ap.add_argument("--bands", type=int, default=20)
    ap.add_argument("--pause", type=int, default=40, help="文と文の間に、語を足さずに回す歩数")
    a = ap.parse_args()
    vocab, talk = demo_world()
    th = Thinker(Vessel(64, bands=a.bands), Words(), Near(list(vocab), list(vocab.values())))
    rows = []

    class Show(Node):
        def after(self, loop, rec):
            if rec["した事"] == "読む":
                rows.append(rec["語"])
            elif rec["した事"] == "連想":
                rows.append(f"+{rec['足した物'][0]}")
            elif rec["した事"] == "組立":
                rows.append("組[" + ("・".join(sorted(rec["違和感"])) or "よし") + "]")
            elif rec["した事"] == "出力":
                rows.append("出『" + " ".join(rec["行"]) + "』")
            else:
                rows.append("待")
    th.attach(Show())
    for s in talk:
        th.words.push(s)
        th.run(a.pause)
    th.run(200)
    out, prev, cnt = [], None, 0
    for r in rows + [None]:                               # 続く「待」はまとめて書く
        if r == prev:
            cnt += 1
            continue
        if prev is not None:
            out.append(prev if cnt == 1 else f"{prev}×{cnt}")
        prev, cnt = r, 1
    print("歩", th.n, "内訳", th.counts())
    print("流れ（+語 = 連想で足した、組[…] = 組立と違和感、出『…』 = 出力、待 = 待つ）:")
    print(" ".join(out))
    t = th.tally()
    print("違和感の波（持つだけ）。した事の後の変わり方（負 = 下がった）:")
    for k, d in t["下がり方"].items():
        print(" ", k, {n: round(v, 3) for n, v in d.items()})
    print("違和感と、器の波ごとの動きの量の相関（速い波 → 遅い波）:")
    for k, row in t["器との動き"].items():
        print(" ", k, [round(v, 2) for v in row])
    H, _ = th.odd.arrays()
    print("終わりの時の振幅（速い波・まん中・遅い波）:", {k: [round(float(H[-1, b, j]), 3) for b in (0, H.shape[1] // 2, H.shape[1] - 1)] for j, k in enumerate(th.odd.kinds)})
    print("見張り", dict(張り付き=int(th.v.watch["張り付き"].sum()), 死んだ=th.v.watch["死んだ"], 刻み=th.v.watch["刻み"]))
    np.savez_compressed(a.demo, **th.arrays())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
