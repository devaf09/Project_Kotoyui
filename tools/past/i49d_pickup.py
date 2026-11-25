"""I49d の選ぶ口の重さの見本: 会話の資料から「拾われやすさ」を数える（持ち主と話した「どれを選ぶか」、未確定）。

数え方: 会話の資料（tools/i49d_odd.ContinueTable と同じ形、話し手の欄は無いので、続く 2 つの発言を「相手の発言 → 返事」と見る）で、
  相手の発言の内容語（I48 の CONTENT: NOUN・PROPN・VERB・ADJ、STOP を除く。同じ発言の中の同じ語は 1 回）ごとに、
  次の返事の内容語に同じ原形が出た割合（拾われた割合）を、(原形, 品詞) ごとに数える。
  少ない語（聞いた数が RARE 未満）は品詞ごとの割合に戻す。ならしは (拾われた数 + A) / (聞いた数 + 2A)。
  学びと試しの分け方は ContinueTable と同じ（crc32(名前) % 5 が 0 の会話が試し）。
  数えの表は会話の資料（非公開）から作るので、リポジトリの外に置く。
確かめ（試しの側）: 拾われた語と拾われなかった語の AUC（この表の値、語の出た数だけ、発言の中の聞いた順の 3 つ）、品詞ごとの拾われた割合。
手で置く物: 少ない語の境 10・ならし 0.5（展開②と同じ）、分け方、内容語の品詞と STOP（I48 と同じ）。
python tools/past/i49d_pickup.py build --rpc <会話の資料> --save <表の置き場.json>
python tools/past/i49d_pickup.py check --rpc <会話の資料> --load <表の置き場.json> --out <stats.json>
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import math
import os
import sys
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
import i48_wave_board as W48  # noqa: E402


def content(toks):
    """発言の内容語 [(原形, 品詞)]（出た順、同じ原形は最初の 1 回）。"""
    out, seen = [], set()
    for t in toks:
        lm, ps = t["lemma"], t["pos"]
        if ps in W48.CONTENT and lm not in W48.STOP and lm not in seen:
            seen.add(lm)
            out.append((lm, ps))
    return out


def pairs(data, test):
    """(相手の発言の内容語, 返事の原形の集まり) の並び。test が True なら試しの側の会話だけ。"""
    for d in data:
        if (zlib.crc32(d["name"].encode()) % 5 == 0) != test:
            continue
        L = [content(ln["toks"]) for ln in d["lines"]]
        for a, b in zip(L, L[1:]):
            yield a, {lm for lm, _ in b}


class PickupTable:
    RARE, A = 10, 0.5

    def __init__(self, cw, tw, cp, tp, base):
        self.cw, self.tw, self.cp, self.tp, self.base = cw, tw, cp, tp, base

    @staticmethod
    def key(lemma, pos):
        return f"{lemma}/{pos}"

    @classmethod
    def build(cls, rpc_json):
        data = json.load(open(rpc_json, encoding="utf-8"))
        cw, tw, cp, tp = Counter(), Counter(), Counter(), Counter()
        n_c = n = 0
        for a, b in pairs(data, test=False):
            for lm, ps in a:
                hit = lm in b
                k = cls.key(lm, ps)
                tw[k] += 1
                cw[k] += hit
                tp[ps] += 1
                cp[ps] += hit
                n_c += hit
                n += 1
        return cls(dict(cw), dict(tw), dict(cp), dict(tp), n_c / max(1, n))

    def value(self, lemma, pos):
        """拾われやすさ（0〜1）。品詞の分からない語は None。"""
        if pos is None:
            return None
        k = self.key(lemma, pos)
        if self.tw.get(k, 0) >= self.RARE:
            return (self.cw.get(k, 0) + self.A) / (self.tw[k] + 2 * self.A)
        if self.tp.get(pos):
            return (self.cp.get(pos, 0) + self.A) / (self.tp[pos] + 2 * self.A)
        return self.base

    def heard(self, lemma, pos):
        return self.tw.get(self.key(lemma, pos), 0)

    def save(self, path):
        json.dump(dict(cw=self.cw, tw=self.tw, cp=self.cp, tp=self.tp, base=self.base), open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        return cls(d["cw"], d["tw"], d["cp"], d["tp"], d["base"])


def auc(score, y):
    """順位で出す AUC（同じ値は平均の順位）。"""
    s, y = np.asarray(score, dtype=np.float64), np.asarray(y, dtype=bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def check(rpc_json, tb):
    data = json.load(open(rpc_json, encoding="utf-8"))
    v, f, pos_first, y, ps_ = [], [], [], [], []
    known = 0
    for a, b in pairs(data, test=True):
        for i, (lm, ps) in enumerate(a):
            v.append(tb.value(lm, ps))
            h = tb.heard(lm, ps)
            known += h >= tb.RARE
            f.append(math.log1p(h))
            pos_first.append(-i)
            y.append(lm in b)
            ps_.append(ps)
    y = np.array(y)
    per = {}
    for p in sorted(set(ps_)):
        m = np.array([q == p for q in ps_])
        per[p] = dict(語の数=int(m.sum()), 拾われた割合=round(float(y[m].mean()), 4),
                      表の値のAUC=None if (a_ := auc(np.array(v)[m], y[m])) is None else round(a_, 4))
    r = lambda x: None if x is None else round(x, 4)
    return dict(試しの語の数=len(y), 拾われた割合=round(float(y.mean()), 4), 表の語で引けた割合=round(known / max(1, len(y)), 4),
                AUC=dict(表の値=r(auc(v, y)), 出た数だけ=r(auc(f, y)), 聞いた順の早さ=r(auc(pos_first, y))),
                品詞ごと=per, 学びの側=dict(語の種類=len(tb.tw), 少ない境を越えた語の種類=sum(1 for c in tb.tw.values() if c >= tb.RARE),
                                         全体の拾われた割合=round(tb.base, 4)))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--rpc", required=True)
    b.add_argument("--save", required=True)
    c = sp.add_parser("check")
    c.add_argument("--rpc", required=True)
    c.add_argument("--load", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        tb = PickupTable.build(a.rpc)
        tb.save(a.save)
        print(len(tb.tw), "語", round(tb.base, 4))
    else:
        res = check(a.rpc, PickupTable.load(a.load))
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
