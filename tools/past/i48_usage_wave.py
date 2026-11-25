"""下調べ G: 使われ方の波（数えだけ、新しい学びなし。持ち主の考え〔案、未確定〕: 語板に使われ方や状況の波が無い）。

語ごとに、文の中での使われ方（前の詰め物・後ろの詰め物・右のいちばん近い述語・文の中の位置・意味の印）を数えて「使われ方の波」にする。
隠した語（下調べ A と同じ 300 語）の「次の使われ方」（後ろの詰め物・述語）を、学んでいない文での出会い 1・2・4・8 回から当てる。
  (a) 今の語板の属性から: 語板の属性が近い学びの語（上位 TOPN）の使われ方を合わせる。
  (b) 使われ方の波から: 出会いの使われ方の波が近い学びの語（上位 TOPN）の使われ方を合わせる。
  (c) 両方: (a) と (b) を半々に合わせる。
  当てずっぽう: 学びの名詞全部の使われ方。
  当てる物: 出会いの 9 回目以降（最大 16 回）の後ろの詰め物と述語。測り: 上位 1・5 に入る割合（述語は上位 5・20）。
  語の頻度の帯: 学びの文での回数の三分位で 多い・中・少ない。前処理（平均を引く・重み付け）は入れない。
意味の印は今回は入れていない（前・後ろの詰め物・述語・位置の 4 つ）。
手で置く物: TOPN（20）、出会いの回数 1・2・4・8、当てる出会いの数の上限 16。
python tools/past/i48_usage_wave.py --conllu ... --content ... --out out.json
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
KS = (1, 2, 4, 8)
N_WORDS = 300
MAX_TEST = 16


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
    vcache = {}

    def bvec(lemma):
        if lemma not in vcache:
            w = board.get(lemma)
            v = None
            if w is not None:
                v = np.zeros(len(names), dtype=np.float32)
                for b in "BCDEG":
                    for x, y in w["attrs"].get(b, {}).items():
                        v[ni[f"{b}|{x}"]] = y
                n = float(np.linalg.norm(v))
                v = v / n if n > 0 else None
            vcache[lemma] = v
        return vcache[lemma]

    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    tr, ref = tr_all[:-2000][: a.train], tr_all[:-2000][a.train:]

    def uses(s):
        cont, gaps, head = G.split(s)
        n = len(cont)
        out = []
        for k, (l, f_, u) in enumerate(cont):
            if u != "NOUN":
                continue
            pred = next((c[0] for c in cont[k + 1:] if c[2] in ("VERB", "ADJ")), "無し")
            pos = "頭" if k == 0 else ("末" if k == n - 1 else "中")
            feats = [("前", head if k == 0 else gaps[k - 1]), ("後", gaps[k]), ("述", pred), ("位置", pos)]
            out.append((l, feats, gaps[k], pred))
        return out

    # 学びの文: 語ごとの使われ方の数え
    use = defaultdict(Counter)
    after, predc = defaultdict(Counter), defaultdict(Counter)
    g_after, g_pred = Counter(), Counter()
    freq = Counter()
    for s in tr:
        for l, feats, aft, pr in uses(s):
            freq[l] += 1
            for f in feats:
                use[l][f] += 1
            after[l][aft] += 1
            predc[l][pr] += 1
            g_after[aft] += 1
            g_pred[pr] += 1
    words = [w for w in use if bvec(w) is not None]
    inv = defaultdict(list)                                     # 使われ方の波（長さ 1 にならす）を、特徴ごとの引き先で持つ
    for i, w in enumerate(words):
        nrm = float(np.sqrt(sum(c * c for c in use[w].values())))
        for f, c in use[w].items():
            inv[f].append((i, c / nrm))
    fi = set(inv)
    Bm = np.stack([bvec(w) for w in words])
    widx = {w: i for i, w in enumerate(words)}
    # 隠す語（A と同じ選び方）
    seen = defaultdict(list)
    for s in ref:
        for l, feats, aft, pr in uses(s):
            if l in board and len(seen[l]) < max(KS) + MAX_TEST:
                seen[l].append((feats, aft, pr))
    cand = sorted((w for w, o in seen.items() if len(o) >= max(KS) and bvec(w) is not None), key=lambda w: zlib.crc32(w.encode()))[:N_WORDS]
    fr = sorted(freq.get(w, 0) for w in cand)
    q1, q2 = fr[len(fr) // 3], fr[2 * len(fr) // 3]
    band = lambda w: "少ない" if freq.get(w, 0) <= q1 else ("中" if freq.get(w, 0) <= q2 else "多い")

    def neigh(sim, w):
        if w in widx:
            sim = sim.copy()
            sim[widx[w]] = -9
        top = np.argpartition(-sim, TOPN)[:TOPN]
        A, P = Counter(), Counter()
        for t in top:
            ww = words[t]
            na, npr = sum(after[ww].values()), sum(predc[ww].values())
            for k_, c in after[ww].items():
                A[k_] += c / na
            for k_, c in predc[ww].items():
                P[k_] += c / npr
        return A, P

    def mix(x, y):
        out = Counter()
        for d in (x, y):
            t = sum(d.values())
            for k_, c in d.items():
                out[k_] += c / t / 2
        return out

    def hit(dist, truth, n):
        return truth in [k_ for k_, _ in dist.most_common(n)]

    rows = []
    for w in cand:
        occ = seen[w]
        test = occ[max(KS):][:MAX_TEST]
        if not test:
            continue
        ga = neigh(Bm @ bvec(w), w)
        r = dict(語=w, 帯=band(w), 当てる出会い=len(test))
        for k in KS:
            u = Counter()
            for feats, _, _ in occ[:k]:
                for f in feats:
                    if f in fi:
                        u[f] += 1
            nu = float(np.sqrt(sum(c * c for c in u.values())))
            sim = np.zeros(len(words), dtype=np.float32)
            for f, c in u.items():
                for i, val in inv[f]:
                    sim[i] += c / nu * val
            gb = neigh(sim, w) if nu > 0 else (Counter(), Counter())
            gc = (mix(ga[0], gb[0]), mix(ga[1], gb[1]))
            for nm, (A_, P_) in (("a", ga), ("b", gb), ("c", gc)):
                r[f"{nm}_{k}"] = dict(後ろ上位1=np.mean([hit(A_, t[1], 1) for t in test]), 後ろ上位5=np.mean([hit(A_, t[1], 5) for t in test]),
                                      述語上位5=np.mean([hit(P_, t[2], 5) for t in test]), 述語上位20=np.mean([hit(P_, t[2], 20) for t in test]))
        r["当てずっぽう"] = dict(後ろ上位1=np.mean([hit(g_after, t[1], 1) for t in test]), 後ろ上位5=np.mean([hit(g_after, t[1], 5) for t in test]),
                             述語上位5=np.mean([hit(g_pred, t[2], 5) for t in test]), 述語上位20=np.mean([hit(g_pred, t[2], 20) for t in test]))
        rows.append(r)
    MK = ("後ろ上位1", "後ろ上位5", "述語上位5", "述語上位20")
    res = dict(隠した語=len(rows), TOPN=TOPN, 帯の境=dict(少ない以下=q1, 中以下=q2), 帯ごとの数=Counter(r["帯"] for r in rows))
    for bnd in ("全部", "多い", "中", "少ない"):
        sel = [r for r in rows if bnd == "全部" or r["帯"] == bnd]
        res[bnd] = {mk: dict(当てずっぽう=round(float(np.mean([r["当てずっぽう"][mk] for r in sel])), 4),
                             **{f"{m_}_{k}": round(float(np.mean([r[f"{m_}_{k}"][mk] for r in sel])), 4) for m_ in "abc" for k in KS}) for mk in MK} if sel else None
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
