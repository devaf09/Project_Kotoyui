"""下調べ A: 隠した語の属性を当てる（数えだけ、新しい学びなし）。

知っている名詞を「知らない語」として隠し、本体が学んでいない文の中での出会い（1・2・4・8 回）から、その語の語板の属性を当てる。
  (a) 前後の語と詰め物から（板の逆読み）: 学びの文で、文脈の特徴（後ろの詰め物・前の詰め物・右のいちばん近い述語・左右の語の
      いちばん重いクラス）ごとに、そこに来た語の属性の平均を持つ（隠す語の分は引く）。出会いの特徴の平均を出す。
  (b) 文全体の波から: 文の中のほかの語の属性のクラスの和（文の波から隠した語を抜いた物）が近い学びの文の出会いを、近い順に
      上位 TOPN 件取り、そこに来た語の属性の平均を出す（隠す語の出会いは除く）。
  (c) 両方: (a) と (b) の平均。
  比べ: いつも同じ答え（学びの文の名詞の属性の、出た回数で重みをつけた平均）。
  近さ: 当てた属性と本当の属性（語板の B・C・D・E・G 帯）のコサイン。語の種類ごと（話題寄り＝G 帯の重さが半分以上、人＝D 帯の
  いちばん重いクラスが 個人・人 系）と、出会いの回数ごとに出す。
手で置く物（A2 も同じ）: TOPN（20、(b) で使う近い出会いの数）、隠す語の数（300）と、その選び方（学んでいない文に 8 回以上出る名詞から、名前の crc32 で選ぶ）。
python tools/past/i48_hidden_word.py --conllu ... --content ... --out out.json
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
import i48ab_long_grain as G  # noqa: E402

TOPN = 20
N_WORDS = 300
KS = (1, 2, 4, 8)
HUMAN = ("個人", "人間", "成人", "女", "男性", "肉縁", "祖", "吾れ")


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    a = ap.parse_args()
    cb = json.load(open(a.content, encoding="utf-8"))
    board = {}
    for w in cb:
        if w["name"] not in board or w.get("freq", 0) > board[w["name"]].get("freq", 0):
            board[w["name"]] = w
    del cb
    names = sorted({f"{b}|{x}" for w in board.values() for b in "BCDEG" for x in w["attrs"].get(b, {})})
    ni = {n: i for i, n in enumerate(names)}
    clsn = sorted({x for w in board.values() for x in w["attrs"].get("D", {})})
    ci = {n: i for i, n in enumerate(clsn)}

    vcache = {}

    def vec(lemma):
        if lemma in vcache:
            return vcache[lemma]
        vcache[lemma] = v = _vec(lemma)
        return v

    def _vec(lemma):
        w = board.get(lemma)
        if w is None:
            return None
        v = np.zeros(len(names), dtype=np.float32)
        for b in "BCDEG":
            for x, y in w["attrs"].get(b, {}).items():
                v[ni[f"{b}|{x}"]] = y
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def top_cls(lemma):
        w = board.get(lemma)
        D = w["attrs"].get("D", {}) if w else {}
        return max(D.items(), key=lambda z: z[1])[0] if D else None

    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    tr, ref = tr_all[:-2000][: a.train], tr_all[:-2000][a.train:]

    def occurrences(s):
        cont, gaps, head = G.split(s)
        out = []
        for k, (l, f_, u) in enumerate(cont):
            if u != "NOUN":
                continue
            pred = next((c[0] for c in cont[k + 1:] if c[2] in ("VERB", "ADJ")), None)
            feats = [("後", gaps[k]), ("前", head if k == 0 else gaps[k - 1]), ("述", pred),
                     ("左", top_cls(cont[k - 1][0]) if k > 0 else "頭"), ("右", top_cls(cont[k + 1][0]) if k + 1 < len(cont) else "末")]
            bag = np.zeros(len(clsn), dtype=np.float32)
            for j, c in enumerate(cont):
                if j != k:
                    t = top_cls(c[0])
                    if t is not None:
                        bag[ci[t]] += 1
            out.append((l, feats, bag))
        return out

    # 隠す語: 学んでいない文に 8 回以上出る名詞（語板あり）から、名前の crc32 の小さい順に N_WORDS 語
    seen = defaultdict(list)
    for s in ref:
        for l, feats, bag in occurrences(s):
            if l in board and len(seen[l]) < max(KS):
                seen[l].append((feats, bag))
    cand = sorted((w for w, o in seen.items() if len(o) >= max(KS) and vec(w) is not None), key=lambda w: zlib.crc32(w.encode()))[:N_WORDS]
    cset = set(cand)
    # 学びの文: 文脈の特徴ごとの属性の和（隠す語の分は別に持って、隠す時に引く）、文の波の鍵
    fsum, fcnt = defaultdict(lambda: np.zeros(len(names), dtype=np.float32)), Counter()
    fsum_w = defaultdict(lambda: defaultdict(lambda: np.zeros(len(names), dtype=np.float32)))
    fcnt_w = defaultdict(Counter)
    keys, kwords = [], []
    mean_sum, mean_n = np.zeros(len(names), dtype=np.float64), 0
    for s in tr:
        for l, feats, bag in occurrences(s):
            v = vec(l)
            if v is None:
                continue
            mean_sum += v
            mean_n += 1
            for f in feats:
                fsum[f] += v
                fcnt[f] += 1
                if l in cset:
                    fsum_w[l][f] += v
                    fcnt_w[l][f] += 1
            nb = float(np.linalg.norm(bag))
            if nb > 0:
                keys.append(bag / nb)
                kwords.append(l)
    K = np.stack(keys)
    kwords = np.array(kwords)
    base = (mean_sum / mean_n).astype(np.float32)
    base /= max(1e-9, float(np.linalg.norm(base)))
    def guess_a(l, occ):
        acc, n = np.zeros(len(names), dtype=np.float32), 0
        for feats, _ in occ:
            for f in feats:
                c = fcnt[f] - fcnt_w[l][f]
                if c > 0:
                    acc += (fsum[f] - fsum_w[l][f]) / c
                    n += 1
        return acc / n if n else None

    def guess_b(l, occ):
        acc, n = np.zeros(len(names), dtype=np.float32), 0
        mask = kwords != l
        for _, bag in occ:
            nb = float(np.linalg.norm(bag))
            if nb == 0:
                continue
            sim = (K @ (bag / nb))
            sim[~mask] = -9
            top = np.argpartition(-sim, TOPN)[:TOPN]
            for t in top:
                acc += vec(kwords[t])
                n += 1
        return acc / n if n else None

    def cos(a_, b_):
        if a_ is None:
            return None
        na, nb = float(np.linalg.norm(a_)), float(np.linalg.norm(b_))
        return float(a_ @ b_ / (na * nb)) if na > 0 and nb > 0 else None

    mu = (mean_sum / mean_n).astype(np.float32)               # 共通の分（学びの名詞の平均）
    P = mu / max(1e-9, float(mu.sum()))
    band_idx = {b: np.array([i for i, n in enumerate(names) if n.startswith(b + "|")]) for b in "BCDEG"}

    def ppmi(x):
        q = x / max(1e-9, float(x.sum()))
        out = np.zeros_like(q)
        ok = (P > 0) & (q > 0)
        out[ok] = np.maximum(0.0, np.log(q[ok] / P[ok]))
        return out

    def metrics(g, t):
        if g is None:
            return None
        gt, tt = ppmi(g), ppmi(t)
        top_g, top_t = set(np.argsort(-gt)[:5]), set(np.argsort(-tt)[:5])
        d = dict(そのまま=cos(g, t), 共通を引く=cos(g - mu, t - mu), 差の重み=cos(gt, tt), 上位5の重なり=len(top_g & top_t) / 5)
        for b, ix in band_idx.items():
            d["帯" + b] = cos((g - mu)[ix], (t - mu)[ix])
        return d

    MK = ("そのまま", "共通を引く", "差の重み", "上位5の重なり") + tuple("帯" + b for b in "BCDEG")
    rows = []
    for w in cand:
        truth = vec(w)
        bw = board[w]["attrs"]
        g_mass = sum(bw.get("G", {}).values())
        tot = sum(sum(bw.get(b, {}).values()) for b in "BCDEG")
        tc = top_cls(w) or ""
        kind = "人" if any(h in tc for h in HUMAN) else ("話題寄り" if tot > 0 and g_mass / tot >= 0.5 else "その他")
        tt = ppmi(truth)
        r = dict(語=w, 種類=kind, 当てずっぽう=dict(そのまま=cos(base, truth), 共通を引く=0.0, 差の重み=0.0,
                                               上位5の重なり=len(set(np.argsort(-mu)[:5]) & set(np.argsort(-tt)[:5])) / 5,
                                               **{"帯" + b: 0.0 for b in "BCDEG"}))
        for k in KS:
            occ = seen[w][:k]
            ga, gb = guess_a(w, occ), guess_b(w, occ)
            gc = None if ga is None or gb is None else (ga + gb) / 2
            r[f"a_{k}"], r[f"b_{k}"], r[f"c_{k}"] = metrics(ga, truth), metrics(gb, truth), metrics(gc, truth)
        rows.append(r)
    res = dict(隠した語=len(rows), TOPN=TOPN, 種類ごとの数=Counter(r["種類"] for r in rows),
               注=dict(当てずっぽう="学びの名詞の平均。共通を引く・差の重み・帯ごとでは、平均そのものは 0 になる（平均からのずれで比べるため）",
                       上位5の重なり="差の重み（PPMI）の上位 5 属性のうち、本当の上位 5 に入った数 ÷ 5。当てずっぽうは平均の上位 5"))

    def mean(key, mk, sel):
        v = [r[key][mk] for r in rows if sel(r) and r[key] is not None and r[key][mk] is not None]
        return round(float(np.mean(v)), 4) if v else None
    for kind in ("全部", "人", "話題寄り", "その他"):
        sel = (lambda r: True) if kind == "全部" else (lambda r, kind=kind: r["種類"] == kind)
        res[kind] = {mk: dict(当てずっぽう=mean("当てずっぽう", mk, sel), **{f"{m_}_{k}": mean(f"{m_}_{k}", mk, sel) for m_ in "abc" for k in KS}) for mk in MK}
    res["例"] = [dict(語=r["語"], 種類=r["種類"], 上位5の重なり_c8=r["c_8"]["上位5の重なり"] if r["c_8"] else None) for r in rows[:10]]
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "例"}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
