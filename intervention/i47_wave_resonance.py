"""介入の研究 I47a: 会話を意味の波にして重ね合わせ、共鳴した場面の次の行から返事の中身をすくい上げる（試し）

持ち主（2026-11-12）:「似た会話や似た状況を同じような周波数にすることで、それを重ね合わせて答えを導く」
  「会話を意味の周波数として、単語と流れの似たものを集めて重ね合わせ、すくいあげて 1 つの会話にし、それをまた重ねて…」
  「語板は基本単独の形。これに関係性を持たせてうねる板の上に広げて、会話資料で学習する」

作り:
  語の波  : 語ごとに D 本の位相（単位の複素数）。始めは乱数。流れから学ぶ: 同じ台詞・隣の台詞に出た語の波を少しずつ足して
            位相を寄せる（似た流れに出る語ほど、位相がそろう = 周波数が近い）。値の名前（性質:〜）は使わない。
  行の波  : 行の中身の語の波の和（強さは珍しさ）。前の行の波を少し重ねて「流れ」を入れる。
  場面    : (行の波 → 次の行の語)。似た場面どうしを重ねて上の段（場面のまとまり）を作る（--levels 2）。
  共鳴    : 今の行の波と、しまってある場面の波の重なりの大きさ |<q, k>|。強く共鳴した場面の次の行を、共鳴の強さで重ねてすくい上げる。
比べる物差し（同じ分け方、10 作品を外す）:
  語そのまま（波を学ばない、乱数の位相のまま = 同じ語が出た場面だけが共鳴）、本の似た行（中身の語の重なり、TF-IDF の余弦）、
  隣の行での語の移り（前の行の語 → 次の行の語の数）、ふだんの多さ。
測り方: 外した作品の台詞の組 (A, B) で、A から B の中身の語を当てる。上位 10 語に B の中身の語がいくつ入るか（再現率@10）と、
  B の中身の語のうち 1 つでも上位 10 に入った割合（当たり@10）。

  python intervention/i47_wave_resonance.py --cache <wn50x5_parsed.json>
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np

from i47_gpu import Scenes, flow_sums

CONTENT = ("NOUN", "PROPN", "VERB", "ADJ")
STOP = {"する", "いる", "ある", "なる", "言う", "思う", "こと", "もの", "よう", "それ", "これ", "の", "ん", "方", "人", "今", "何", "自分", "私", "俺", "僕", "君", "あなた"}


def content(toks):
    return [t["lemma"] for t in toks if t["pos"] in CONTENT and t["lemma"] not in STOP and len(t["lemma"]) >= 1]


def pairs(works):
    """作品ごとの台詞の組 (前の行の語, A の語, B の語)。地の文を挟んだら組にしない。"""
    out = []
    for wk in works:
        L = wk["lines"]
        for i in range(1, len(L)):
            if L[i - 1]["text"].startswith("「") and L[i]["text"].startswith("「"):
                prev = content(L[i - 2]["toks"]) if i >= 2 and L[i - 2]["text"].startswith("「") else []
                a, b = content(L[i - 1]["toks"]), content(L[i]["toks"])
                if a and b:
                    out.append((prev, a, b))
    return out


class Waves:
    def __init__(self, vocab, D=512, seed=0):
        rng = np.random.default_rng(seed)
        self.ix = {w: i for i, w in enumerate(vocab)}
        self.base = np.exp(1j * rng.uniform(0, 2 * np.pi, (len(vocab), D)))
        self.W = self.base.copy()

    def learn(self, lines_seq, rounds=2, mix=0.5):
        """流れから学ぶ: 語の波 = 自分の乱数の波 + mix × (同じ行・隣の行に出た語の波の平均)。回すほど、似た流れの語の位相がそろう。"""
        for _ in range(rounds):
            acc, n = flow_sums(lines_seq, self.ix, self.W)  # 1 回目は乱数の波（random indexing）、2 回目からは学んだ波の和
            m = n > 0
            ctxw = np.zeros_like(self.W)
            ctxw[m] = acc[m] / n[m, None]
            ctxw[m] -= ctxw[m].mean(0)                      # どの語にも共通の向き（搬送波）を引く。引かないと、全部の行が同じ向きに共鳴する
            ctxw[m] /= (np.linalg.norm(ctxw[m], axis=1, keepdims=True) / math.sqrt(self.W.shape[1]) + 1e-9)
            W = (1 - mix) * self.base + mix * ctxw
            self.W = W / np.maximum(np.abs(W), 1e-9)       # 位相だけ残す（強さは 1）

    def line(self, ws, idf):
        ids = [(self.ix[w], idf.get(w, 1.0)) for w in ws if w in self.ix]
        if not ids:
            return None
        v = sum(self.W[j] * a for j, a in ids)
        return v / (np.linalg.norm(v) + 1e-9)


COMMON = set()


def evaluate(pred, test, k=10):
    """(再現率@k, 当たり@k, ふだんの多い語を除いた再現率@k)。最後のが「場面から来た中身」を当てたか。"""
    rec, hit, inf, ni = [], 0, [], 0
    for (_, a, b), ranked in zip(test, pred):
        top = set(ranked[:k])
        bs = set(b)
        got = len(bs & top)
        rec.append(got / len(bs))
        hit += got > 0
        bi = bs - COMMON
        if bi:
            inf.append(len(bi & top) / len(bi))
    return round(sum(rec) / len(rec), 4), round(hit / len(test), 4), round(sum(inf) / max(1, len(inf)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--test_works", type=int, default=10)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--top", type=int, default=30, help="すくい上げる場面の数")
    ap.add_argument("--flow", type=float, default=0.3, help="前の行の波を重ねる重さ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.5, help="ふだんの多さを混ぜる重さ")
    ap.add_argument("--mix", type=float, default=0.5, help="語の波を流れで寄せる重さ")
    ap.add_argument("--out")
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    random.Random(a.seed).shuffle(works)
    tr_w, te_w = works[a.test_works:], works[:a.test_works]
    train, test = pairs(tr_w), pairs(te_w)
    seq = [content(ln["toks"]) for wk in tr_w for ln in wk["lines"] if ln["text"].startswith("「")]
    seq_all = [content(ln["toks"]) for wk in tr_w for ln in wk["lines"]]   # 語の波は地の文も含めた全部の行の流れから学ぶ
    df = Counter(w for wk in tr_w for ln in wk["lines"] for w in set(content(ln["toks"])))
    N = sum(len(wk["lines"]) for wk in tr_w)
    idf = {w: math.log((N + 1) / (c + 0.5)) for w, c in df.items()}
    vocab = sorted(df)
    freq = Counter(w for _, _, b in train for w in b)
    prior = [w for w, _ in freq.most_common(200)]
    COMMON.update(prior[:50])
    pf = {w: n / sum(freq.values()) for w, n in freq.items()}

    def mixed(c, lam=a.lam):
        """すくい上げた語の分布と、ふだんの多さを混ぜる。"""
        z = sum(c.values()) or 1.0
        sc = Counter({w: (1 - lam) * v / z for w, v in c.items()})
        for w in prior:
            sc[w] += lam * pf[w]
        return [w for w, _ in sc.most_common(60)]
    res = dict(train=len(train), test=len(test), vocab=len(vocab))

    # 物差し 1: ふだんの多さ
    res["ふだんの多さ"] = evaluate([prior] * len(test), test)
    # 物差し 2: 隣の行での語の移り
    tr = defaultdict(Counter)
    for _, x, b in train:
        for w in set(x):
            tr[w].update(set(b))
    pred = []
    for _, x, _ in test:
        c = Counter()
        for w in set(x):
            for v, n in tr[w].items():
                c[v] += n / max(1, sum(tr[w].values()))
        pred.append(mixed(c))
    res["隣の行での語の移り"] = evaluate(pred, test)

    def scoop(keys, query_vecs):
        """共鳴の強い場面の次の行の語を、共鳴の強さで重ねる。"""
        K = Scenes(np.stack(keys))
        out = []
        for q in query_vecs:
            if q is None:
                out.append(prior)
                continue
            top, rt = K.top(q, a.top)
            c = Counter()
            for j, rj in zip(top, rt):
                for w in set(train[j][2]):
                    c[w] += rj ** 2
            out.append(mixed(c))
        return out

    def run(name, wv):
        keys, qs, keep = [], [], []
        for j, (p, x, _) in enumerate(train):
            v = wv.line(x, idf)
            pv = wv.line(p, idf) if p else None
            if v is None:
                v = np.zeros(a.D, complex)
            keys.append(v + (a.flow * pv if pv is not None else 0))
        for p, x, _ in test:
            v = wv.line(x, idf)
            pv = wv.line(p, idf) if p else None
            qs.append(None if v is None else v + (a.flow * pv if pv is not None else 0))
        res[name] = evaluate(scoop(keys, qs), test)
        return keys

    def wave_pick(name, wv, keys):
        """波で測る: すくい上げた次の行の波（共鳴の強い場面の B の波の重ね）が、本当の B と、ほかの 99 行の B のどれと
        いちばん共鳴するか。本当の B が上位 10 に入る割合（偶然なら 0.10）。語を当てるより「向き」が合うかを見る。"""
        K = Scenes(np.stack(keys))
        BV = [wv.line(b, idf) for _, _, b in train]
        BV = np.stack([v if v is not None else np.zeros(a.D, complex) for v in BV])
        TB = [wv.line(b, idf) for _, _, b in test]
        rr = np.random.default_rng(1)
        hit, n = 0, 0
        for i, (p, x, _) in enumerate(test):
            v = wv.line(x, idf)
            if v is None or TB[i] is None:
                continue
            pv = wv.line(p, idf) if p else None
            q = v + (a.flow * pv if pv is not None else 0)
            top, rt = K.top(q, a.top)
            pw = (BV[top] * (rt ** 2)[:, None]).sum(0)
            others = [j for j in rr.choice(len(test), 120, replace=False) if j != i and TB[j] is not None][:99]
            cand = [TB[i]] + [TB[j] for j in others]
            s_ = [abs(np.vdot(c, pw)) for c in cand]
            rank = sum(1 for z in s_[1:] if z >= s_[0])
            hit += rank < 10
            n += 1
        res[name + "・向き（本当の次の行が 100 行中の上位 10）"] = round(hit / max(1, n), 4)

    wv = Waves(vocab, D=a.D, seed=a.seed)
    k0 = run("語そのまま（乱数の波、学ばない）", wv)
    wave_pick("語そのまま", wv, k0)
    wv.learn(seq_all, rounds=2, mix=a.mix)
    keys = run("流れで学んだ波（I47a）", wv)
    wave_pick("流れで学んだ波", wv, keys)
    # 段: 似た場面を重ねて、まとまりの波にする（共鳴でまとまりを選び、その中の場面からすくう）
    K = np.stack(keys)
    rng = np.random.default_rng(a.seed)
    C = K[rng.choice(len(K), 200, replace=False)]
    for _ in range(5):                                      # 共鳴の強いまとまりに入れ、まとまりの波 = 中の場面の波の重ね合わせ
        lab = np.argmax(np.abs(K @ C.conj().T), 1)
        for c in range(len(C)):
            m = lab == c
            if m.any():
                s = K[m].sum(0)
                C[c] = s / (np.linalg.norm(s) + 1e-9)
    groups = defaultdict(list)
    for j, c in enumerate(lab):
        groups[c].append(j)
    pred = []
    for p, x, _ in test:
        v = wv.line(x, idf)
        pv = wv.line(p, idf) if p else None
        if v is None:
            pred.append(prior)
            continue
        q = v + (a.flow * pv if pv is not None else 0)
        rc = np.abs(C @ q.conj())
        cs = np.argsort(-rc)[:3]
        cand = [j for c in cs for j in groups[c]]
        r = np.abs(K[cand] @ q.conj())
        c = Counter()
        for jj in np.argsort(-r)[: a.top]:
            for w in set(train[cand[jj]][2]):
                c[w] += r[jj] ** 2
        pred.append(mixed(c))
    res["流れで学んだ波 + まとまりの段（I47a）"] = evaluate(pred, test)
    # 物差し 3: 本の似た行（TF-IDF の余弦、波を使わない）
    def tfidf(ws):
        c = Counter(ws)
        return {w: n * idf.get(w, 1.0) for w, n in c.items()}
    inv = defaultdict(list)
    tv = []
    for j, (_, x, _) in enumerate(train):
        v = tfidf(x)
        nv = math.sqrt(sum(s * s for s in v.values())) or 1
        tv.append((v, nv))
        for w in v:
            inv[w].append(j)
    pred = []
    for _, x, _ in test:
        q = tfidf(x)
        nq = math.sqrt(sum(s * s for s in q.values())) or 1
        sc = Counter()
        for w, s in q.items():
            for j in inv.get(w, []):
                sc[j] += s * tv[j][0][w] / (nq * tv[j][1])
        c = Counter()
        for j, r in sc.most_common(a.top):
            for w in set(train[j][2]):
                c[w] += r ** 2
        pred.append(mixed(c))
    res["本の似た行（語の重なり）"] = evaluate(pred, test)
    # 似た語の例（流れで学んだ波で、位相がそろった語）
    ex = {}
    for w in ("海", "猫", "雨", "仕事", "電気", "誕生日", "学校", "嬉しい"):
        if w in wv.ix:
            s = np.abs(wv.W.conj() @ wv.W[wv.ix[w]]) / a.D
            ex[w] = [vocab[j] for j in np.argsort(-s)[1:9]]
    res["似た波の語"] = ex
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
