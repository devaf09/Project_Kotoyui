"""I48ah: 印が無い時に、時の語や周りから印を当てる（小さく）。要点に印が付いていない場合を考え、文の内容語だけから、文末の節の印
（過去・否定・丁寧・進行・問い）の有り無しを当てる。

- 当て方: 印ごとに、有り／無しの 2 つの板を、文の内容語の属性の帯の数え上げ（i48ab の PairGrain、足すだけ）で持つ。内容語の位置は
  文末からの近さ（最後の語 = 1、その前 = 2、それより前は 3 にまとめる）のずれで置く。時の語も内容語なので、そのまま入る（名前でなく属性）。
- 測る物: 印ごとの当たり（基準はいちばん多い方）、時の語のある文での当たり、当てた印で逆回しした時の文末の当たり（答えの印を使った時と比べる）。
手で置く物: 時の語の一覧（測るためだけ、i48ae の TIME_WORDS）、近さの段の数（3）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ae_marks as AE  # noqa: E402

MARKS = ("過去", "否定", "丁寧", "進行", "問い")


def sent_feat(S, s):
    cont, _, _ = G.split(s)
    fs, vs = [], []
    n = len(cont)
    for j, (l, _, u) in enumerate(cont):
        f, v = S.attrs(l, u)
        near = min(3, n - j)
        fs.append((f + near * S.shift) % S.D)
        vs.append(v)
    if not fs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    return np.concatenate(fs), np.concatenate(vs).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--D", type=int, default=8192)
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0][:-2000][: a.train]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))

    def label(s):
        cm = AE.clause_marks(s, func)
        return set(cm[-1]) if cm else set()
    TR = [(sent_feat(S, s), label(s)) for s in tr if G.split(s)[0]]
    TE = [(sent_feat(S, s), label(s), any(l in AE.TIME_WORDS for l, _, _ in G.split(s)[0]), s) for s in te if G.split(s)[0]]
    res = dict(学びの文=len(TR), 試しの文=len(TE), 印ごと={})
    pred = [set() for _ in TE]
    for m in MARKS:
        cnt = Counter(m in y for _, y in TR)
        pg = G.PairGrain(a.D, n_cls=2).fit((((f, v), m in y)[0] + ((m in y),) for (f, v), y in TR), cnt)
        base = cnt.most_common(1)[0][0]
        c = Counter()
        for i, ((f, v), y, tm, _) in enumerate(TE):
            p = pg.predict(f, v)[0]
            if p:
                pred[i].add(m)
            g = m in y
            c["n"] += 1
            c["当たり"] += p == g
            c["基準"] += base == g
            c["有り"] += g
            c["有りを当てた"] += g and p
            c["有りと言った"] += p
            if tm:
                c["時n"] += 1
                c["時当たり"] += p == g
                c["時基準"] += base == g
        res["印ごと"][m] = dict(当たり=round(c["当たり"] / c["n"], 3), 基準=round(c["基準"] / c["n"], 3),
                              再現率=round(c["有りを当てた"] / max(1, c["有り"]), 3), 適合率=round(c["有りを当てた"] / max(1, c["有りと言った"]), 3),
                              時の語あり=dict(件数=c["時n"], 当たり=round(c["時当たり"] / max(1, c["時n"]), 3), 基準=round(c["時基準"] / max(1, c["時n"]), 3)))
        print(m, res["印ごと"][m], flush=True)
    exact = sum(p == y for p, (_, y, _, _) in zip(pred, TE)) / len(TE)
    res["印の組がそっくり当たる"] = round(exact, 3)
    tm_rows = [(p, y) for p, (_, y, tm, _) in zip(pred, TE) if tm]
    res["印の組がそっくり当たる_時の語あり"] = [len(tm_rows), round(sum(p == y for p, y in tm_rows) / max(1, len(tm_rows)), 3)]
    print(res["印の組がそっくり当たる"], res["印の組がそっくり当たる_時の語あり"], time.time() - t0, flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
