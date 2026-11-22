"""I49e: 3 人で読む案の二歩目の部品。黙って聞いているだけで育つ「拾われやすさ」（持ち主の案「黙ってても何かしら学習になる」、未確定）。

聞く物は「番」の並び。番 = (話し手, 内容語の並び [(原形, 品詞)])。本のページも 1 つの番として渡せる（話し手 = "本"）。
1 つの番を聞くたびに、直前の番（別の話し手）の内容語それぞれについて、今の番に同じ原形が出たか（拾われたか）を数える。

  LivePickup(prior=None, half=1024)
    prior(原形, 品詞) → 0〜1 か None   初めの見込み（教師ありで数えた表 tools/i49d_pickup.PickupTable.value など）。無ければ全体の割合を使う
    hear(話し手, 語)                   番を 1 つ聞く。数える前に、今の見込みで「拾われるか」を言い当てた記録（value, 拾われた）を残す
    value(原形, 品詞)                  今の見込み = (拾われた数 ＋ 初めの見込み) / (聞いた数 ＋ 1)。聞いていない語は、初めの見込みそのまま
    記憶は消さずに薄れる: 番を 1 つ聞くたびに、数に 0.5^(1/half) を掛ける。half は器のいちばん遅い波と同じ 1,024（新しい数を置かない）
    report()                           言い当ての良さ（AUC）を、初めの見込みだけ／聞いて育てた見込み、で並べる。前半と後半にも分ける

手で置く物: 初めの見込みの重さ（聞いた 1 回ぶん）、薄れる速さ（器と同じ 1,024）、「直前の番」だけを見る事。
乱数は使わない。
"""
from __future__ import annotations

import numpy as np


def auc(score, y):
    """順位で出す AUC（同じ値は平均の順位）。片方しか無ければ None。"""
    s, y = np.asarray(score, dtype=np.float64), np.asarray(y, dtype=bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ss, r, i = s[order], np.empty(len(s)), 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


class LivePickup:
    def __init__(self, prior=None, half=1024.0):
        self.prior = prior
        self.keep = float(0.5 ** (1.0 / half))
        self.heard, self.picked = {}, {}                  # (原形, 品詞) → 薄れる数（float32 で足りるが、dict なので float）
        self.all_heard = self.all_picked = 0.0
        self.last = None                                  # 直前の番 (話し手, 語)
        self.rec = []                                     # 言い当ての記録: (番の番号, 初めの見込み, 今の見込み, 拾われた)
        self.n = 0

    def _prior(self, w):
        v = self.prior(*w) if self.prior is not None else None
        if v is None:
            return (self.all_picked + 0.5) / (self.all_heard + 1.0)
        return float(v)

    def value(self, lemma, pos):
        w = (lemma, pos)
        return (self.picked.get(w, 0.0) + self._prior(w)) / (self.heard.get(w, 0.0) + 1.0)

    def hear(self, who, words):
        words = list(dict.fromkeys(tuple(w) for w in words))          # 同じ番の同じ語は 1 回
        if self.last is not None and self.last[0] != who:
            now = {l for l, _ in words}
            for w in self.last[1]:
                hit = w[0] in now
                self.rec.append((self.n, self._prior(w), self.value(*w), hit))
            for k in self.heard:                                      # 薄れる
                self.heard[k] *= self.keep
                self.picked[k] *= self.keep
            self.all_heard *= self.keep
            self.all_picked *= self.keep
            for w in self.last[1]:
                hit = w[0] in now
                self.heard[w] = self.heard.get(w, 0.0) + 1.0
                self.picked[w] = self.picked.get(w, 0.0) + float(hit)
                self.all_heard += 1.0
                self.all_picked += float(hit)
        self.last = (who, words)
        self.n += 1

    def report(self):
        if not self.rec:
            return dict(語の数=0)
        t, p0, p1, y = (np.array(x) for x in zip(*self.rec))
        half = t >= np.median(t)

        def pair(m):
            r = lambda v: None if v is None else round(v, 4)
            return dict(語の数=int(m.sum()), 拾われた割合=round(float(y[m].mean()), 4) if m.any() else None,
                        初めの見込みだけ=r(auc(p0[m], y[m])), 聞いて育てた見込み=r(auc(p1[m], y[m])))
        return dict(全体=pair(np.ones(len(t), dtype=bool)), 前半=pair(~half), 後半=pair(half),
                    聞いた語の種類=len(self.heard), 番の数=int(self.n))
