"""I49zzu: 相手の行の語と、返事の行の語に、数えられるつながりが有るか（持ち主の案 3-1 の前提の確かめ。未確定の試し）。

持ち主の案:「相手の発話の3つ組 (s1, o1, v1) を受け取り、応答の3つ組 (s2, o2, v2) を出すのを、across_pairs(61節の)で作った遷移波と同じ構造をつかって、
  例えば相手が「～食べた」を「おいしい、何、食べる」というような連想に持って行けるか？」
3 つ組を切り出す前に、もっと手前を見る: 相手の行の中身の語（名詞・動詞・形容詞の見出し）から、返事の行の中身の語が、当てずっぽうより当たるか。
材料: 解析済みの会話（作品ごとの行に、語の見出し・品詞・話し手の印が付いた JSON。話し手は「替わったか」だけを見る）。
数え方（61 節の状況波と同じ形。席 = 相手の行の語、単位 = 返事の行の語）:
  波[返事の語, 相手の語] = その返事の語が出た時に、相手の行にその語が有った割合 − 全体での割合。
  点（相手の行, 返事の語）= 相手の行の語についての波の足し算。
確かめ方（会話の 1 割を外して置く。2 つから選ぶ）:
  本物の返事の語と、別の所の返事から取った語（その返事には無い語）を並べ、点の高い方を選ぶ。5 割なら、つながりは無い。
  分けて数える: 返事の語が、相手の行にも有った（同じ語を繰り返す）／無かった（別の語へ進む）。
  比べ役: 相手の行を、別の所の行と入れ替えた時（5 割になるはず）。同じ話し手が続けた 2 行（相手ではなく、自分の続き）。
手で置いた物: SEATS、MIN、中身の語とする品詞、外す割合。乱数は使わない。
`--out` には語の例（材料から数えた、つながりの強い語）が出るので、リポジトリには入れない。
引数: --cache 解析済み JSON --out 記録.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49_situation as SIT  # noqa: E402

POS = ("NOUN", "VERB", "ADJ", "PROPN")
SEATS = 2000
MIN = 30
HOLD = 0.1
SHOW = ("食べる", "行く", "見る", "仕事", "好き")   # 例を出す相手の語（持ち主の例「～食べた」から）


def words_of(line):
    return sorted({t["lemma"] for t in line["toks"] if t["pos"] in POS})


def pairs_of(work, same=False):
    """→ [(前の行の語, 今の行の語)]。same=False は話し手が替わった所（相手 → 返事）、True は同じ話し手が続けた所。"""
    out = []
    ls = work["lines"]
    for a, b in zip(ls, ls[1:]):
        if (a.get("speaker") == b.get("speaker")) != same:
            continue
        p, c = words_of(a), words_of(b)
        if p and c:
            out.append((p, c))
    return out


class WordAssoc:
    def __init__(self, train, seats=None, min_count=None):
        pc, cc = {}, {}
        for p, c in train:
            for x in p:
                pc[x] = pc.get(x, 0) + 1
            for x in c:
                cc[x] = cc.get(x, 0) + 1
        self.seats = sorted(pc, key=lambda x: (-pc[x], x))[:SEATS if seats is None else seats]
        mc = MIN if min_count is None else min_count
        self.units = sorted((x for x, n in cc.items() if n >= mc), key=lambda x: (-cc[x], x))
        self.six = {x: i for i, x in enumerate(self.seats)}
        self.uix = {x: i for i, x in enumerate(self.units)}
        m = np.zeros((len(self.units), len(self.seats)), dtype=np.float64)
        for p, c in train:
            us = [self.uix[x] for x in c if x in self.uix]
            ps = [self.six[x] for x in p if x in self.six]
            if us and ps:
                m[np.ix_(us, ps)] += 1
        n = np.array([cc[x] for x in self.units], dtype=np.float64)[:, None]
        tot = np.array([pc[x] for x in self.seats], dtype=np.float64) / max(len(train), 1)
        self.w = (m / np.maximum(n, 1) - tot[None, :]).astype(np.float32) if m.size else np.zeros(m.shape, dtype=np.float32)

    def score(self, prev, word):
        ps = [self.six[x] for x in prev if x in self.six]
        return float(self.w[self.uix[word], ps].sum()) if ps and word in self.uix else 0.0

    def top(self, word, k=12):
        """相手の語 word の後の返事に、いつもより多く出る語（波の高い順）。"""
        if word not in self.six:
            return []
        col = self.w[:, self.six[word]]
        return [self.units[i] for i in np.argsort(-col)[:k]]


def two_choice(assoc, test, swap=False):
    """→ 分け方 → dict(問い, 当たり)。同点は半分。swap=True は、相手の行を別の所の行と入れ替える（比べ役）。"""
    out = {k: dict(問い=0, 当たり=0.0) for k in ("全部", "相手の行にも有った語", "相手の行に無かった語")}
    n = len(test)
    for i, (p, c) in enumerate(test):
        if swap:
            p = test[(i + n // 3 + 1) % n][0]
        other = test[(i + n // 2 + 1) % n][1]
        cs = [x for x in c if x in assoc.uix]
        os_ = [x for x in other if x in assoc.uix and x not in c]
        for j, u in enumerate(cs):
            if not os_:
                break
            o = os_[j % len(os_)]
            a, b = assoc.score(p, u), assoc.score(p, o)
            hit = 1.0 if a > b else (0.5 if a == b else 0.0)
            for k in ("全部", "相手の行にも有った語" if u in p else "相手の行に無かった語"):
                out[k]["問い"] += 1
                out[k]["当たり"] += hit
    for c in out.values():
        c["割合"] = round(c["当たり"] / max(c["問い"], 1), 4)
    return out


def run(works):
    order = SIT.golden_order(len(works))
    k = max(int(len(works) * HOLD), 1)
    held, kept = [works[i] for i in order[:k]], [works[i] for i in order[k:]]
    res = dict(作品=len(works), 外した作品=len(held))
    for name, same in (("相手 → 返事", False), ("同じ話し手の続き", True)):
        train = [x for w in kept for x in pairs_of(w, same)]
        test = [x for w in held for x in pairs_of(w, same)]
        a = WordAssoc(train)
        res[name] = dict(数える組=len(train), 確かめる組=len(test), 席=len(a.seats), 返事の語の種類=len(a.units),
                         二つから選ぶ=two_choice(a, test), 相手の行を入れ替えた時=two_choice(a, test, swap=True))
        if not same:
            res["例（リポジトリに入れない）"] = {w: a.top(w) for w in SHOW}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = dict(手で置いた値=dict(SEATS=SEATS, MIN=MIN, HOLD=HOLD, POS=list(POS)), **run(json.load(open(a.cache, encoding="utf-8"))))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
