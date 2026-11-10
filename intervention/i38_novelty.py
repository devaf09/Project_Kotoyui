"""介入の研究 実験I38a: 新規性（遊び心）の感覚のグレイン（持ち主の案 2026-11-10「自覚に違和感のようなグレイン群を追加しよう」の 1 つ目）

違和感と同じ形にそろえる: 記号を作り、本の台詞で数えた期待と、会話での観測を比べて点にする。
  記号（返事の行から）:
    語  内容語の原形（名詞・固有名詞・動詞・形容詞）。
    形  言い切りの述語の形: 根の品詞 + 根より後ろの表記（「VERB:ていない。」「ADJ:の？」）。根が述語でない行（「はい。」）には無い。
    型  行の並び: 助詞・助動詞・記号は表記のまま、ほかは品詞（「INTJ、NOUNはADJ。」）。続く同じ品詞は 1 つにまとめる。
  観測 o: その記号が、この会話の自分の直近 K 行（既定 3）に出たか（0/1）。
  期待 r: 本の続く台詞で、同じ話し手（2 行前・4 行前・6 行前 …、甲 → 乙 → 甲 とみる）の直近 k 行に、同じ記号が出る割合。
     記号の種類（語・形・型）と k ごとの割合を土台に、その記号そのものの出方で寄せる（本でよく繰り返す語・形ほど r が大きい）。
  点（記号ごと）: 繰り返した（o=1）→ −(1 − r)（本でふつう繰り返さないものを繰り返すほど大きく減らす）。
     使っていない（o=0）→ +(1 − r)（ふつう繰り返さないものを新しく使うほど大きく足す）。ただし語の無い行（「はい。」「ああ、ならん。」）は足さない（安全な最短ルートは 0 以下）。
  行の点 = 0.5 × 記号ごとの点の平均（−0.5〜0.5、長さには比例しない）。
本の行は非公開（Web 小説なら _wn）。数えた板は非公開の置き場へ。
実行（板だけ数える）: python intervention/i38_novelty.py --cache <解析済み JSON> --out <板の JSON>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i34_dialogue_board as D  # noqa: E402

CW = ("NOUN", "PROPN", "VERB", "ADJ")
PRED = ("VERB", "ADJ", "AUX")
KEEP = ("ADP", "AUX", "SCONJ", "PART", "CCONJ")
MARK = set("。、？！?!…")
KINDS = ("語", "形", "型")


def nov_syms(toks):
    """行の記号の集まり（語・形・型）。"""
    out = set()
    T = [t for t in toks if not (t["pos"] in ("PUNCT", "SYM") and not (set(t["text"]) & MARK))]   # かっこは外す
    for t in T:
        if t["pos"] in CW:
            out.add("語:" + t["lemma"])
    root = next((t for t in T if t["dep"] == "ROOT"), None)
    if root is not None and (root["pos"] in PRED or any(t["dep"] == "cop" and t["head"] == root["i"] for t in T)):
        tail = "".join(t["text"] for t in T if t["i"] > root["i"]).rstrip("。")   # 言い切りの「。」は本の台詞に無いことが多いので外す
        out.add(f"形:{root['pos']}:{tail}")
    seq = []
    for t in T:
        x = t["text"] if (t["pos"] in KEEP or t["pos"] in ("PUNCT", "SYM")) else t["pos"]
        if not seq or seq[-1] != x or x in MARK:
            seq.append(x)
    while seq and seq[-1] == "。":
        seq.pop()
    if seq:
        out.add("型:" + "".join(seq))
    return out


def kind(s):
    return s.split(":", 1)[0]


def novelty_board(works, K=3, min_n=3):
    """本の続く台詞で、同じ話し手の直近 k 行（k = 1..K）に同じ記号が出る割合。
    返すもの: 土台 {種類: {k: [繰り返した数, 使った数]}}、記号 {記号: [繰り返した数, 期待の和, 使った数]}（使った数 >= min_n だけ）。"""
    base = {c: {k: [0, 0] for k in range(1, K + 1)} for c in KINDS}
    per = {}
    runs = []
    for w in works:
        run = []
        for ln in w["lines"]:
            if D.isq(ln["text"]) and ln["toks"]:
                run.append(ln)
            else:
                if len(run) >= 3:
                    runs.append(run)
                run = []
        if len(run) >= 3:
            runs.append(run)
    seen = []
    for run in runs:
        S = [nov_syms(ln["toks"]) for ln in run]
        for j in range(2, len(run)):
            own = [S[j - 2 * m] for m in range(1, K + 1) if j - 2 * m >= 0]
            k = len(own)
            prev = set().union(*own)
            for s in S[j]:
                rep = s in prev
                b = base[kind(s)][k]
                b[0] += rep
                b[1] += 1
                seen.append((s, k, rep))
    rate = {c: {k: (v[0] + 0.5) / (v[1] + 1.0) for k, v in d.items()} for c, d in base.items()}
    for s, k, rep in seen:
        e = per.setdefault(s, [0, 0.0, 0])
        e[0] += rep
        e[1] += rate[kind(s)][k]
        e[2] += 1
    per = {s: v for s, v in per.items() if v[2] >= min_n}
    return dict(K=K, base=base, sym=per)


class Novelty:
    """新規性の感覚: 板（本の期待）と、会話の自分の直近の行（観測）を比べる。"""

    def __init__(self, board, a=2.0):
        self.K = board["K"]
        self.rate = {c: {int(k): (v[0] + 0.5) / (v[1] + 1.0) for k, v in d.items()} for c, d in board["base"].items()}
        self.sym = board["sym"]
        self.a = a

    def r(self, s, k):
        """記号 s の期待（自分の直近 k 行に出る割合）。土台の割合 × その記号の寄せ（繰り返した数 / 期待の和、a で 1 に寄せる）。"""
        base = self.rate[kind(s)][max(1, min(k, self.K))]
        v = self.sym.get(s)
        lift = (v[0] + self.a * 1.0) / (v[1] + self.a) if v else 1.0
        return min(0.99, base * lift)

    def score(self, toks, own_prev):
        """行の点と、記号ごとの (記号, 観測, 期待)。own_prev: 自分の直近の行の toks（新しい順、K 行まで使う）。"""
        own = [nov_syms(T) for T in own_prev[: self.K]]
        S = nov_syms(toks)
        if not own or not S:
            return 0.0, []
        prev = set().union(*own)
        has_word = any(kind(s) == "語" for s in S)
        det, vs = [], []
        for s in sorted(S):
            o = s in prev
            r = self.r(s, len(own))
            v = -(1.0 - r) if o else ((1.0 - r) if has_word else 0.0)
            vs.append(v)
            det.append((s, o, r))
        return 0.5 * sum(vs) / len(vs), det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=3)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    b = novelty_board(works, a.K)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(b, f, ensure_ascii=False)
    for c, d in b["base"].items():
        print(c, {k: f"{v[0]}/{v[1]} = {v[0] / max(1, v[1]):.3f}" for k, v in d.items()})
    print("記号", len(b["sym"]))


if __name__ == "__main__":
    main()
