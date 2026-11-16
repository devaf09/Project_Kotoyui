"""I48ai: OK を覚える段（持ち主の方針: 正誤の介入で OK なら学習値を記憶。だめな並びは強く入れず、暗い側へ 1/1,000〜1/10,000）。

- 候補のグレイン: 節・シフト＋元（I48ae）。選び方: ふるい・弱（I48ag）。
- 練習: 学びに使っていない文（練習の文）で要点 → 文を作り、間ごとに正誤を見る（試しの場では、元の文の詰め物と同じなら OK）。
    OK   : その間の特徴を、選んだ詰め物の板に重さ 1 で足す（明るい側）
    だめ : その間の特徴を、選んだ詰め物の暗い側に DARK_W の重さで足す（消さない、強く覚えない）
  点: 明るい側の対数の点 − 暗い側の重さで薄めた分（log(1 + 暗い / (明るい + α))）。
- 測る物: 練習の前後で、試しの文の間・文末・文が元と同じ。持ち主の例（友達・散歩・行く → と）。
手で置く物: DARK_W（千分の 1・万分の 1）、練習の文の数。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ac_center_grain as CG  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402

MODE = "シフト＋元"


class Memo:
    def __init__(self, pg, dark_w):
        self.pg, self.w = pg, dark_w
        self.Wd = np.zeros_like(pg.W)

    def refresh(self):
        tot = self.pg.W.sum(1, keepdims=True)
        self.pg.logp = np.log((self.pg.W + G.ALPHA) / (tot + G.ALPHA * self.pg.D))
        self.pg.logp[-1] = -1e9

    def ok(self, c, f, v):
        np.add.at(self.pg.W[c], f, v)

    def bad(self, c, f, v):
        np.add.at(self.Wd[c], f, v * self.w)

    def post(self, f, v, prior_w=1.0):
        s = prior_w * self.pg.prior + self.pg.logp[:, f] @ v - np.log1p(self.Wd[:, f] / (self.pg.W[:, f] + G.ALPHA)) @ v
        s = s - s.max()
        p = np.exp(s)
        return p / p.sum()


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--practice", type=int, default=5000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0][:-2000]
    tr = tr_all[: a.train]
    pr = tr_all[a.train: a.train + a.practice]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    TR = [x for s in tr for x in AE.items(S, s, "節")]
    cnt = Counter(x["y"] for x in TR)
    feat = lambda x: AE.feats(S, x, MODE)
    print(f"読み込み {time.time() - t0:.0f} 秒", flush=True)
    res = dict(学びの文=len(tr), 練習の文=len(pr), 試しの文=len(te), 構成={})

    def evaluate(memo, sents_, label):
        c = Counter()
        for s in sents_:
            xs = AE.items(S, s, "節")
            if not xs:
                continue
            cont, gaps, head = G.split(s)
            seq = []
            for x in xs:
                p = memo.post(*feat(x))
                p[-1] = 0
                seq.append(memo.pg.cls[int(np.argmax(p))])
            c["文"] += 1
            c["間n"] += len(gaps) - 1
            for i, (g, y) in enumerate(zip(seq, gaps)):
                c[":間" if i + 1 < len(gaps) else ":文末"] += g == y
            sent = "".join(surf.form(l, u, g) + g for (l, f_, u), g in zip(cont, seq))
            c[":文"] += G.norm(head + sent) == G.norm(s["text"])
        return dict(間=round(c[":間"] / max(1, c["間n"]), 3), 文末=round(c[":文末"] / max(1, c["文"]), 3), 文=round(c[":文"] / max(1, c["文"]), 3))

    def example(memo):
        out = []
        for keys in ((("昨日", "NOUN"), ("友達", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")),
                     (("昨日", "NOUN"), ("公園", "NOUN"), ("犬", "NOUN"), ("散歩", "NOUN"), ("する", "VERB"))):
            A = [S.attrs(l, u) for l, u in keys]
            n = len(keys)
            xs = [dict(A=A, i=i, mk=("過去",) if i + 1 == n else (("過去",) if i + 1 < n else ()), last=i + 1 == n) for i in range(n)]
            seq, top = [], []
            for x in xs:
                p = memo.post(*feat(x))
                p[-1] = 0
                o = np.argsort(-p)[:3]
                seq.append(memo.pg.cls[o[0]])
                top.append([memo.pg.cls[i] for i in o])
            out.append(dict(文="".join(surf.form(l, u, g) + g for (l, u), g in zip(keys, seq)), 上位3=top))
        return out

    for name, w in (("千分の 1", 1e-3), ("万分の 1", 1e-4)):
        t1 = time.time()
        pg = G.PairGrain(a.D).fit(((*feat(x), x["y"]) for x in TR), cnt)
        memo = Memo(pg, w)
        memo.refresh()
        before = evaluate(memo, te, "前")
        ex_before = example(memo)
        ok = bad = 0
        for s in pr:                                       # 練習: 作って、正誤を見て、OK は足し、だめは薄く暗い側へ
            for x in AE.items(S, s, "節"):
                f, v = feat(x)
                p = memo.post(f, v)
                p[-1] = 0
                ci = int(np.argmax(p))
                if memo.pg.cls[ci] == x["y"]:
                    memo.ok(ci, f, v)
                    ok += 1
                else:
                    memo.bad(ci, f, v)
                    bad += 1
            memo.refresh()
        after = evaluate(memo, te, "後")
        o = dict(秒=round(time.time() - t1), 練習の_OK=ok, 練習の_だめ=bad, 前=before, 後=after, 例_前=ex_before, 例_後=example(memo))
        res["構成"][name] = o
        print(name, json.dumps(o, ensure_ascii=False), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
