"""周波数の語板（tools/i48_freq_board.py、118 語・自作 105 文、乱数の形・W 1）で、「その語に特有の属性だけ残す」読み方（案、未確定）。

箱 b（文脈の語）を距離 d で回して、帯ごとに属性の鳴り（属性の形との重なりの実部）を出し、帯ごとに上位 K の属性をその箱・距離の「残る属性」とする。
特有の属性: 残る属性のうち、同じ品詞のほかの語の箱（同じ距離）で、半分以上の箱の上位 K にも出る属性を除いた物。
  平均を引く・比を取る形は使わず、「ほかの箱の上位 K に出るか出ないか」（出現の有無）の数だけで決める。
候補の点:
  (A) そのまま: 今までの共振（全部の周波数）
  (B) 特有だけ: 文脈の箱ごとの特有の属性のうち、候補の語板にある属性の重み（帯ごとに長さ 1）の和
  (C) そのまま ＋ 特有: (A) と (B) を、それぞれ候補の中で平均 0・ばらつき 1 にそろえて足す
測る物: 学習した文の 1 位、学習に無い大体正しい文の 1 位・上位 5、おかしな文を見分ける AUC（本当の語の、候補の中でそろえた z の行の平均・最小）。
手で置く物: K（3・5）、除く境（同じ品詞の箱の半分）。
python tools/i48_freq_board4.py --content .../content_boards.json --out out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48_freq_board as FB  # noqa: E402

NC = FB.NC
KS = (3, 5)
HALF = 0.5


def auc(pos, neg):
    a = np.array(list(neg) + list(pos))
    r = a.argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def z(x):
    return (x - x.mean()) / max(1e-12, x.std())


class Spec:
    def __init__(self, bd, R, K):
        self.bd, self.R, self.K = bd, R, K
        self.attrs = sorted(bd.shapes)                         # (帯, 属性)
        self.ai = {a: i for i, a in enumerate(self.attrs)}
        self.band_of = [FB.BANDS.index(b) for b, _ in self.attrs]
        self.upos = {w: next(iter(bd.att[w]["品詞"])) for w in bd.lemmas}
        # 候補ごとの属性の重み（帯ごとに長さ 1）
        M = np.zeros((len(bd.lemmas), len(self.attrs)))
        for k, w in enumerate(bd.lemmas):
            for b, A in bd.att[w].items():
                n = np.sqrt(sum(v * v for v in A.values()))
                for a, v in A.items():
                    M[k, self.ai[(b, a)]] = v / n
        self.M = M
        self.top_cache, self.spec_cache = {}, {}

    def top(self, b, d):
        key = (b, d)
        if key not in self.top_cache:
            Q = np.conj(self.R[b]) * self.bd.ph(d)
            r = np.array([np.real(np.sum(self.bd.shapes[a] * Q[FB.BANDS.index(a[0]) * NC:(FB.BANDS.index(a[0]) + 1) * NC])) for a in self.attrs])
            keep = set()
            for bi in range(len(FB.BANDS)):
                idx = [i for i in range(len(self.attrs)) if self.band_of[i] == bi]
                if idx:
                    idx = np.array(idx)
                    keep |= set(idx[np.argsort(-r[idx])[: self.K]].tolist())
            self.top_cache[key] = keep
        return self.top_cache[key]

    def spec(self, b, d):
        key = (b, d)
        if key not in self.spec_cache:
            w = self.bd.lemmas[b]
            same = [i for i, x in enumerate(self.bd.lemmas) if self.upos[x] == self.upos[w] and i != b]
            cnt = {}
            for i in same:
                for a in self.top(i, d):
                    cnt[a] = cnt.get(a, 0) + 1
            self.spec_cache[key] = {a for a in self.top(b, d) if cnt.get(a, 0) < HALF * max(1, len(same))}
        return self.spec_cache[key]

    def score(self, ids, p):
        s = np.zeros(len(self.bd.lemmas))
        for q, b in enumerate(ids):
            if q != p:
                sp = list(self.spec(b, p - q))
                if sp:
                    s += self.M[:, sp].sum(1)
        return s


def evaluate(bd, R, sp, lines, kind):
    h = h5 = n = 0
    rows = []
    for ln in lines:
        ids = [bd.li[w] for w in ln]
        if len(ids) < 2:
            continue
        zt = []
        for p, t in enumerate(ids):
            a = bd.scores(R, ids, p)
            if kind == "そのまま":
                sc = a
            elif kind == "特有だけ":
                sc = sp.score(ids, p)
            else:
                sc = z(a) + z(sp.score(ids, p))
            h += int((sc > sc[t]).sum() == 0)
            h5 += int((sc > sc[t]).sum() < 5)
            n += 1
            zt.append(z(sc)[t])
        rows.append((float(np.mean(zt)), float(np.min(zt))))
    return dict(一位=round(h / n, 4), 上位5=round(h5 / n, 4)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    cb_all = {}
    for w in json.load(open(a.content, encoding="utf-8")):
        if w["name"] not in cb_all or w.get("freq", 0) > cb_all[w["name"]].get("freq", 0):
            cb_all[w["name"]] = w
    train = [s.split() for _, s in FB.LINES]
    ok = [s.split() for _, s in FB.UNSEEN_OK]
    bad = [s.split() for _, s in FB.UNSEEN_BAD]
    lemmas = sorted({w for s in train for w in s})
    bd = FB.Board(cb_all, lemmas, "乱数", None, 1.0)
    R = bd.learn(train)
    res = dict(語=len(lemmas), 学習の文=len(train), 除く境=HALF, 型={})
    for K in KS:
        sp = Spec(bd, R, K)
        for kind in ("そのまま", "特有だけ", "そのまま ＋ 特有"):
            if kind == "そのまま" and K != KS[0]:
                continue
            r = {}
            for tag, L in (("学習した文", train), ("学習に無い大体正しい文", ok), ("おかしな文", bad)):
                r[tag], rows = evaluate(bd, R, sp, L, kind)
                r[tag + "_行"] = rows
            ro, rb = r.pop("学習に無い大体正しい文_行"), r.pop("おかしな文_行")
            r.pop("学習した文_行")
            r["おかしな文を見分けるAUC"] = dict(行の平均=round(auc([x[0] for x in ro], [x[0] for x in rb]), 4),
                                         行の最小=round(auc([x[1] for x in ro], [x[1] for x in rb]), 4))
            if kind != "そのまま":
                sizes = [len(sp.spec(bd.li[s[q]], p - q)) for s in train for p in range(len(s)) for q in range(len(s)) if p != q]
                tops = [len(sp.top(bd.li[s[q]], p - q)) for s in train for p in range(len(s)) for q in range(len(s)) if p != q]
                r["残る属性の数_平均"] = round(float(np.mean(tops)), 2)
                r["特有の属性の数_平均"] = round(float(np.mean(sizes)), 2)
            nm = kind if kind == "そのまま" else f"{kind}・K {K}"
            res["型"][nm] = r
            print(nm, json.dumps(r, ensure_ascii=False), flush=True)
    # 例: 食べる・飲む・読む の 1 つ前の特有の属性（K 3）
    sp = Spec(bd, R, 3)
    res["1つ前の特有の属性_K3"] = {v: ["・".join([b, x.split(":")[-1]]) for b, x in sorted(sp.attrs[i] for i in sp.spec(bd.li[v], -1)) if b != "品詞"] for v in ("食べる", "飲む", "読む") if v in bd.li}
    print(json.dumps(res["1つ前の特有の属性_K3"], ensure_ascii=False), flush=True)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
