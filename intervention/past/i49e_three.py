"""I49e: 3 人で読む案の三歩目の部品。ことゆい（I49d の思考のループ）を、読む場の 3 人目として座らせる（持ち主の案、未確定）。

席 Seat は薄い。ループ本体（I49d の Thinker）は変えずに使う。
  listen(話し手, 語)   番を 1 つ聞く。語 = [(原形, 品詞, ベクトル か None)]。
                       拾われやすさ（LivePickup）に番として渡し、ベクトルを持つ語は 1 語ずつ思考のループに読ませる（器も聞く）。
                       番の途中では「文の終わり」の印を付けないので、ループは「途中」のまま材料をため、話さない。
  speak(aloud=True)    自分の番。入れ物に「文の終わり」の印を置き、ループが出力か待つを選ぶまで回す。出力した行（語の名前の並び）を返す。
                       話さなかった時（材料が少ない・もう言った）は None。無理には話させない。
                       aloud=False（聞くだけの段）では、行は作るが場には出さない（拾われやすさの番に入れない）。
  選び方               組立の口 Scored の選ぶ口に、重さ weight(名前) を渡して作る（helper: live_weight）。
                       重さ = 今の拾われやすさ（聞いて育った見込み）。番で聞いた語の中から、拾われやすい物を k 個選んで言う。
確かめは外から来る: 声に出した行を拾われやすさに番として渡すと、次の先生の番で、ことゆいの語が拾われたかが数えられる（話し手の組「こ → 先生」）。

手で置く物: 話し手の名前「こ」、選ぶ数 k（Pick に渡す）。乱数は使わない。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

ME = "こ"


def live_weight(live, pos_of):
    """選ぶ口に渡す重さ: 今の拾われやすさ。品詞の分からない語は None（いちばん後ろ）。"""
    return lambda name: live.value(name, pos_of[name]) if pos_of.get(name) else None


class Seat:
    def __init__(self, thinker, live, pos_of):
        self.th, self.live, self.pos_of = thinker, live, pos_of     # pos_of は組立の口と同じ dict（聞いた語の品詞を足して行く）
        self.log = []                                               # 自分の番の記録: (番の番号, 行 か None, 声に出したか, 入れ物の材料の数)

    def listen(self, who, words):
        words = list(words)
        self.live.hear(who, [(l, p) for l, p, _ in words])
        heard = [(l, v, {}) for l, p, v in words if v is not None]
        for l, p, _ in words:
            self.pos_of.setdefault(l, p)
        self.th.words.push(heard)
        while self.th.words.has_next():                             # 読み切るまで回す（1 語ごとに「途中」の組立が挟まる）
            self.th.step()

    def speak(self, aloud=True):
        th = self.th
        n_mat = len(th.tray.vecs())
        th.tray.add("文の終わり", None, "印", th.n)                  # 自分の番が来た = ここで言える
        th.tried = False
        line = None
        while True:
            rec = th.step()
            if rec["した事"] == "出力":
                line = list(rec["行"])
                break
            if rec["した事"] == "待つ":
                break
        if line is None:                                            # 話さなかった時は印を外す（次の番の途中で話し出さないように）
            th.tray.items = [it for it in th.tray.items if not (it["種類"] == "印" and it["名前"] == "文の終わり")]
        self.log.append((self.live.n, line, bool(aloud and line), n_mat))
        if aloud and line:
            self.live.hear(ME, [(w, self.pos_of[w]) for w in line if self.pos_of.get(w)])
        return line
