"""I49zzr: 状況波（位相の形）を自然に出た向きへずらして文の終わりを書き替え、違和感チェックに通す（持ち主の案、未確定の試し）。

持ち主の案:「とりあえず状況波（状況位相？）を、よりよくわからない関係性をもつ空気感位相に発展させたい。
  できた状況波をずらして会話をいくつか作り、違和感チェックの仕組みを通してどのくらい精度の正しい文章を作り出せるかをためしてほしい」
ここに有る物:
  OddCheck: 文の終わりの違和感チェック（新しく作った物）。材料の文の、続いた N 文字の並びを数えておく。
    書き替えた文の、つなぎ目をまたぐ N 文字の並びのうち、材料に 1 回も出ていない物が有れば「違和感あり」。
    分ける道具も辞書も使わない。見るのは文字の並びだけなので、意味の違和感（打ち消しが入れ替わる、など）は分からない。
  air_directions: 状況波に自然に出た強い向き（名前は付けない）を、位相の波として取り出す。
  shifted: 状況波を、向きに沿ってずらす。
  rewrites: 行の終わりを、つなぎ目の表（i49_situation.joins）に有る終わりへ替えた候補の全部と、状況波との合い方。
手で置いた物: N（違和感を見る並びの長さ）、ずらす大きさ BETA、向きの数。
乱数は使わない。float32 / complex64。
"""
from __future__ import annotations

import numpy as np

import i49_situation as SIT

N = 4        # 違和感を見る文字の並びの長さ（手で置いた値）
BETA = 2.0   # ずらす大きさ（元の状況波の大きさの何倍か。手で置いた値）
HEAD, END = "\u0002", "\u0003"   # 文の頭と終わりの印


class OddCheck:
    def __init__(self, sents, n=None):
        self.n = N if n is None else n
        self.count = {}
        for s in sents:
            t = HEAD + s + END
            for i in range(len(t) - self.n + 1):
                g = t[i:i + self.n]
                self.count[g] = self.count.get(g, 0) + 1

    def grams(self, line, k):
        """つなぎ目（終わりの k 文字の手前）をまたぐ n 文字の並び。"""
        t = HEAD + line + END
        seam = len(t) - 1 - k                 # t[seam] が、替えた終わりの初めの字
        return [t[i:i + self.n] for i in range(max(seam - self.n + 1, 0), seam) if i + self.n <= len(t)]

    def unseen(self, line, k, least=1):
        """つなぎ目をまたぐ並びのうち、材料に least 回も出ていない物の数と、並びの数。"""
        g = self.grams(line, k)
        return sum(1 for x in g if self.count.get(x, 0) < least), len(g)

    def passes(self, line, k, least=1):
        """least=2 は、材料の中の文そのものを確かめる時に使う（自分の 1 回を除く）。"""
        return self.unseen(line, k, least)[0] == 0


def air_directions(phase, k=3):
    """状況波に自然に出た強い向き k 本（位相の波）。数えた結び付きの、席の側の強い向きを、席の位相の波で足した物。名前は付けない。"""
    w = phase.a.w
    if not w.size:
        return np.zeros((0, phase.dims), dtype=np.complex64), []
    _, s, vt = np.linalg.svd(w.astype(np.float64), full_matrices=False)
    k = min(k, len(s))
    d = (vt[:k].astype(np.complex64) @ phase.e)
    share = (s / max(float(s.sum()), 1e-12))[:k]
    return d, [float(x) for x in share]


def shifted(base, direction, sign, ref, beta=None):
    """状況波 base を、direction に沿って sign の向きへずらす。大きさは ref（元の状況波のふつうの大きさ）の beta 倍。"""
    beta = BETA if beta is None else beta
    n = float(np.linalg.norm(direction))
    if n <= 0:
        return base
    return (base + np.complex64(sign * beta * ref / n) * direction).astype(np.complex64)


def rewrites(sent, sit, phase, table):
    """行の終わりを替えた候補の全部。→ 点の高い順の (書き替えた行, 点, 替えた終わりの文字の数)。元の行も、替えない候補として入れる（文字の数 0）。"""
    best = {sent: (phase.fit(sit, SIT.tail_of(sent)), 0)}
    for k in range(1, SIT.TAIL + 1):
        if len(sent) < k + SIT.JOIN:
            continue
        for e, c in table.get(sent[-k - SIT.JOIN:-k], {}).items():
            new = sent[:-k] + e
            if c < SIT.SEEN or new == sent:
                continue
            sc = phase.fit(sit, SIT.tail_of(new))
            if new not in best or sc > best[new][0]:
                best[new] = (sc, len(e))
    return sorted(((n, sc, k) for n, (sc, k) in best.items()), key=lambda x: (-x[1], x[0]))
