"""I49e: 「2 つから残りの 1 つを問う」聞き方（持ち主の案「算数で言うなら A＋B＝？みたいな関係性を聞く」、未確定）の一歩目。数えだけ。

思考のループも先生も使わない。青空文庫の作品（3 人で読む試しと同じ選び方の --books 番目、1 ページ 10 文）の文から、
GiNZA の文の形で 3 つ組（だれが = が格の名詞・何を = を格の名詞・どうした = 述語の動詞の原形）を取る。
  述語 = 品詞 VERB の語。だれが = その語に nsubj でかかり、格の助詞「が」を持つ NOUN／PROPN。何を = obj でかかり、「を」を持つ NOUN／PROPN。
  3 つともベクトルを持つ物だけ使う。取れなかった理由を数える（が格が無い・を格が無い・ベクトルが無い（どこが））。
1 つを隠し、残り 2 つのベクトルを長さ 1 にして足した向き（重みなし）に近い順（|cos|）に候補を並べ、隠した語の順位を見る。隠す場所 3 通りを別々に。
候補: ア そのページの内容語（見せた 2 語を除く） ／ イ その作品全体の内容語（同じく除く）。内容語 = ベクトルを持つ (原形, 品詞)。
数える: 1 位の当たり、3 位以内、順位の中央値、候補の数の中央値。
比べ: ① でたらめ（候補の数から計算: 1/n、min(3,n)/n、(n+1)/2）② そのページでいちばん多く出る語（見せた 2 語を除く。同じ数なら先に出た語）
  ③ 関係の無い 2 語（別の文の 3 つ組の、同じ場所の 2 語。seed 固定）を足した向き ④ 見せた 2 語のうち 1 語だけの向き（前の語・後の語）
  ⑤ 候補を、隠した語と同じ品詞だけに絞る。
ベクトル: 波の板の語のベクトル（tools/i49c_read_words.WordVecs、ことゆいが使っている物）。--attrs を付けると属性のベクトル（Assoc.from_attrs）でも数える。
  前処理（平均を引く・重み）はしない。
手で置く物: 作品 1〜8、1 ページ 10 文、seed 7（③）、|cos|、品詞の決まり。
python tools/i49e_triple_count.py --src ... --meta ... --wave <波の板> --out <stats.json> [--books 1,...,8] [--attrs --content ... --func ...]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

SLOTS = ("だれが", "何を", "どうした")
NOUNS = ("NOUN", "PROPN")


def has_case(tok, mark):
    return any(c.dep_ == "case" and c.lemma_ == mark for c in tok.children)


def triples_of(doc):
    """文の 3 つ組 [(だれが, 何を, どうした)]。各要素は (原形, 品詞)。取れなかった理由も返す。"""
    out, why = [], Counter()
    for t in doc:
        if t.pos_ != "VERB":
            continue
        subj = [c for c in t.children if c.dep_ == "nsubj" and c.pos_ in NOUNS and has_case(c, "が")]
        obj = [c for c in t.children if c.dep_ == "obj" and c.pos_ in NOUNS and has_case(c, "を")]
        if not subj and not obj:
            why["が格もを格も無い"] += 1
        elif not subj:
            why["が格が無い"] += 1
        elif not obj:
            why["を格が無い"] += 1
        else:
            out.append(((subj[0].lemma_, subj[0].pos_), (obj[0].lemma_, obj[0].pos_), (t.lemma_, t.pos_)))
    return out, why


def unit(v):
    v = np.asarray(v)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class Table:
    """語 → 行の番号と、共役を取ったベクトルの行列（近さをまとめて計算するため）。"""

    def __init__(self, vec):
        self.names = list(vec)
        self.ix = {w: i for i, w in enumerate(self.names)}
        self.Vc = np.conj(np.stack([unit(vec[w]) for w in self.names]))

    def rank_of(self, q, hidden, cands):
        """q の向きに近い順（|cos|）に候補を並べた時の、隠した語の順位（1 から）。同じ値の語は前に置く（甘めに数えない）。"""
        idx = np.array([self.ix[w] for w in cands])
        s = np.abs(self.Vc[idx] @ unit(q))
        h = s[cands.index(hidden)]
        return int(1 + np.sum(s >= h) - 1)


def summ(ranks, ns):
    r, n = np.array(ranks), np.array(ns)
    if not len(r):
        return dict(数=0)
    return dict(数=len(r), 一位=round(float(np.mean(r == 1)), 4), 三位以内=round(float(np.mean(r <= 3)), 4),
                順位の中央値=float(np.median(r)), 候補の数の中央値=float(np.median(n)))


def chance(ns):
    n = np.array(ns, dtype=float)
    return dict(一位=round(float(np.mean(1 / n)), 4), 三位以内=round(float(np.mean(np.minimum(3, n) / n)), 4),
                順位の中央値=float(np.median((n + 1) / 2)))


def count(items, vec, seed=7):
    """items: [dict(3つ組, ページの語の並び, 作品の語の集まり)]。vec: (原形, 品詞) → ベクトル。"""
    rng = random.Random(seed)
    tb = Table(vec)
    rank_of = lambda q, hid, cands, _vec: tb.rank_of(q, hid, cands)
    res = {}
    usable = [it for it in items if all(w in vec for w in it["組"])]
    for si, slot in enumerate(SLOTS):
        shown = [j for j in range(3) if j != si]
        for cname in ("ア_ページ", "イ_作品"):
            R = {k: [] for k in ("足した向き", "関係の無い2語", "前の語だけ", "後の語だけ", "同じ品詞だけ")}
            N, Npos, freq_hit1, freq_hit3 = [], [], [], []
            for it in usable:
                tri = it["組"]
                hid = tri[si]
                show = {tri[j] for j in shown}
                pool = it["ページ"] if cname == "ア_ページ" else it["作品"]
                cands = sorted({w for w in pool if w in vec and w not in show} | {hid}, key=str)
                if len(cands) < 2:
                    continue
                N.append(len(cands))
                q = unit(vec[tri[shown[0]]]) + unit(vec[tri[shown[1]]])
                R["足した向き"].append(rank_of(q, hid, cands, vec))
                other = rng.choice([o for o in usable if o is not it and o["文"] != it["文"]])["組"]
                R["関係の無い2語"].append(rank_of(unit(vec[other[shown[0]]]) + unit(vec[other[shown[1]]]), hid, cands, vec))
                R["前の語だけ"].append(rank_of(vec[tri[shown[0]]], hid, cands, vec))
                R["後の語だけ"].append(rank_of(vec[tri[shown[1]]], hid, cands, vec))
                same = [w for w in cands if w[1] == hid[1]]
                Npos.append(len(same))
                R["同じ品詞だけ"].append(rank_of(q, hid, same, vec) if len(same) >= 1 else None)
                if cname == "ア_ページ":
                    fc = Counter(w for w in it["ページの並び"] if w in vec and w not in show)
                    order = sorted(fc, key=lambda w: (-fc[w], it["ページの並び"].index(w)))
                    freq_hit1.append(bool(order) and order[0] == hid)
                    freq_hit3.append(hid in order[:3])
            o = {k: summ([x for x in v if x is not None], N if k != "同じ品詞だけ" else [n for n, x in zip(Npos, v) if x is not None])
                 for k, v in R.items()}
            o["でたらめ"] = chance(N)
            o["同じ品詞だけのでたらめ"] = chance([n for n in Npos if n > 0])
            if freq_hit1:
                o["いちばん多い語"] = dict(一位=round(float(np.mean(freq_hit1)), 4), 三位以内=round(float(np.mean(freq_hit3)), 4))
            res[f"{slot}を隠す・{cname}"] = o
    return res, len(usable)


def main():
    import spacy
    import i48_wave_board as W48
    import i49c_read_words as Rw
    import i49e_read_together as RT
    ap = argparse.ArgumentParser()
    for k in ("src", "meta", "wave", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--books", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--attrs", action="store_true")
    ap.add_argument("--content", default=None)
    ap.add_argument("--func", default=None)
    a = ap.parse_args()
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    items, why, n_pred, n_tri = [], Counter(), 0, 0
    miss = Counter()
    for rk in map(int, a.books.split(",")):
        _, S = RT.pick_book(a.src, a.meta, rank=rk)
        docs = list(nlp.pipe(S, batch_size=128))
        page_seq = {}
        for si, d in enumerate(docs):
            page_seq.setdefault(si // RT.PAGE_LEN, []).extend(
                (t.lemma_, t.pos_) for t in d if t.pos_ in W48.CONTENT and t.lemma_ not in W48.STOP)
        work = {w for seq in page_seq.values() for w in seq}
        for si, d in enumerate(docs):
            n_pred += sum(1 for t in d if t.pos_ == "VERB")
            tris, w_ = triples_of(d)
            why.update(w_)
            for tri in tris:
                n_tri += 1
                lack = [s for s, w in zip(SLOTS, tri) if wv.of(*w) is None]
                if lack:
                    miss["ベクトルが無い: " + "・".join(lack)] += 1
                    continue
                items.append(dict(組=tri, 文=(rk, si), ページの並び=page_seq[si // RT.PAGE_LEN],
                                  ページ=set(page_seq[si // RT.PAGE_LEN]), 作品=work))
    vec = {}
    for it in items:
        for w in set(it["ページ"]) | set(it["作品"]):
            if w not in vec:
                v = wv.of(*w)
                if v is not None:
                    vec[w] = v
    res = dict(作品=a.books, 述語の数=n_pred, 三つ組=n_tri, 使えた三つ組=len(items), 取れなかった理由=dict(why), 落とした理由=dict(miss))
    res["波の語のベクトル"], _ = count(items, vec)
    if a.attrs:
        import i48ab_long_grain as G
        import i49d_assoc as AS
        cb = json.load(open(a.content, encoding="utf-8"))
        A = AS.Assoc.from_attrs(G.AttrSpace(cb, G.Func(json.load(open(a.func, encoding="utf-8"))), 2048), cb)
        avec = {w: A.vec(w[0]) for w in vec if A.vec(w[0]) is not None}
        items_a = [dict(it, ページ={w for w in it["ページ"] if w in avec}, 作品={w for w in it["作品"] if w in avec}) for it in items]
        res["属性のベクトル"], res["属性で使えた三つ組"] = count(items_a, avec)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
