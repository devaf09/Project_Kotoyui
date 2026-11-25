"""I48m: 返事の行の始めの句が、相手の前の行の中身の語をどれだけ受けているか（RealPersonaChat）を数える。bot の返事と比べる下ごしらえ。

始めの句: 行の始めの受けの一言（I48l の取り方）を外した後、最初の区切り（、。！？…）までのかたまり。
受けている: 始めの句の中身の語（名詞・動詞・形容詞など）に、
  同じ語 = 相手の前の行の中身の語と同じ原形がある
  近い語 = 学んだ語の波の向きの重なり |cos| が、乱数どうしの重なりの床（σ√(2 ln V)、σ = 1/√D。I48 の読み出しの床と同じ決め方）を越える
句の形: 受けた語を X、ほかの中身の語を品詞の印（名・動・形）、機能語は語の形のままにした並び（「X+は+形+です+ね」など）。
  資料の中で多い形を数える（一覧は手で置かない）。

実行: python intervention/past/i48m_echo_phrase.py --cache <parsed.json> [--logs <bot の記録の works JSON> --runs run1,run2] --out i48m.json
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import math
import os
import random
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i48_wave_board import Waves  # noqa: E402
from i48e_relation_split import CONTENT, CUE  # noqa: E402
from i48l_uke_flow import BREAK, body, uke  # noqa: E402

TAG = {"NOUN": "名", "PROPN": "名", "PRON": "代", "VERB": "動", "ADJ": "形", "ADV": "副", "NUM": "数"}


def first_phrase(toks):
    """受けの一言を外した、行の始めの句（語の並び）。"""
    T = body(toks)
    u = uke(toks)
    i = 0
    if u is not None:                                               # 受けの一言と、その後の区切りを飛ばす
        while i < len(T) and T[i]["text"] not in BREAK:
            i += 1
        while i < len(T) and T[i]["text"] in BREAK:
            i += 1
    j = i
    while j < len(T) and T[j]["text"] not in BREAK:
        j += 1
    return T[i:j]


class Near:
    def __init__(self, wv, D):
        self.wv, self.D = wv, D
        self.floor = math.sqrt(2 * math.log(max(2, len(wv.vocab)))) / math.sqrt(D)
        self.Wn = wv.W / (np.abs(wv.W).sum(1, keepdims=True) / math.sqrt(D) + 1e-9)

    def near(self, a, b):
        i, j = self.wv.ix.get(a), self.wv.ix.get(b)
        if i is None or j is None:
            return False
        c = abs(np.vdot(self.wv.W[i], self.wv.W[j])) / (np.linalg.norm(self.wv.W[i]) * np.linalg.norm(self.wv.W[j]) + 1e-9)
        return c > self.floor


def echo(prev_toks, reply_toks, nr):
    P = {t["lemma"] for t in body(prev_toks) if t["pos"] in CONTENT}
    F = first_phrase(reply_toks)
    same = [t for t in F if t["pos"] in CONTENT and t["lemma"] in P]
    near = [t for t in F if t["pos"] in CONTENT and t["lemma"] not in P and any(nr.near(t["lemma"], p) for p in P)] if P else []
    hit = {id(t) for t in same + near}
    shape = "+".join("X" if id(t) in hit else TAG.get(t["pos"], t["text"]) for t in F) if F else ""
    return dict(has_prev=bool(P), n_first=len(F), same=bool(same), near=bool(near), shape=shape if (same or near) else None,
                first_content=any(t["pos"] in CONTENT for t in F))


def count(pairs, nr, seed=0):
    """数える。偶然の高さ（同じ返事を、別の対話の相手の行と組ませた時）も出し、受けの割合から引いた「偶然より上」も出す。"""
    rng = random.Random(seed)
    others = [a for a, _ in pairs]
    R = [echo(a, b, nr) for a, b in pairs]
    N = [echo(rng.choice(others), b, nr) for _, b in pairs]
    has = [(r, z) for r, z in zip(R, N) if r["has_prev"] and z["has_prev"]]
    f = lambda key, X: round(sum(key(x) for x in X) / max(1, len(X)), 4)
    rs, zs = [r for r, _ in has], [z for _, z in has]
    out = dict(行=len(pairs), 数えた行=len(has), 始めの句に中身の語がある=f(lambda r: r["first_content"], rs))
    for name, key in (("同じ語", lambda r: r["same"]), ("近い語（同じ語を除く）", lambda r: r["near"] and not r["same"]),
                      ("どちらか", lambda r: r["same"] or r["near"])):
        a_, z_ = f(key, rs), f(key, zs)
        out[name] = dict(受ける割合=a_, 偶然=z_, 偶然より上=round(a_ - z_, 4))
    out["多い形"] = Counter(r["shape"] for r in rs if r["shape"]).most_common(25)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--n", type=int, default=20000, help="数える RealPersonaChat の行の数")
    ap.add_argument("--logs")
    ap.add_argument("--runs", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    vocab = sorted({t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] in CUE})
    seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in CUE] for w in works for ln in w["lines"] + [dict(toks=[])]]
    wv = Waves(vocab, a.D, a.seed)
    wv.learn(seq)
    nr = Near(wv, a.D)
    pairs = [(w["lines"][i - 1]["toks"], w["lines"][i]["toks"]) for w in works for i in range(1, len(w["lines"]))]
    pairs = random.Random(a.seed).sample(pairs, min(a.n, len(pairs)))
    res = dict(D=a.D, floor=round(nr.floor, 4), rpc=count(pairs, nr, a.seed), bot={})
    print("RPC", json.dumps({k: v for k, v in res["rpc"].items() if k != "多い形"}, ensure_ascii=False))
    print("RPC 多い形", res["rpc"]["多い形"][:15])
    if a.logs:
        L = json.load(open(a.logs, encoding="utf-8"))["works"]
        for run in [r for r in a.runs.split(",") if r]:
            bp = []
            for w in L:
                if w["name"].split(":")[0] != run:
                    continue
                for i in range(1, len(w["lines"])):
                    ln = w["lines"][i]
                    if ln["who"] == "ボット" and ln.get("made") and w["lines"][i - 1]["who"] == "相手":
                        bp.append((w["lines"][i - 1]["toks"], ln["toks"]))
            res["bot"][run] = count(bp, nr, a.seed)
            print(run, json.dumps({k: v for k, v in res["bot"][run].items() if k != "多い形"}, ensure_ascii=False))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
