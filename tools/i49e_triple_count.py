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
    ap.add_argument("--conllu", default=None, help="二歩目: 覚える側に使う青空文庫の conllu（試しの作品は除く）")
    ap.add_argument("--cap", type=int, default=2000, help="二歩目: 1 作品から読む文の上限")
    ap.add_argument("--pick_words", default=None, help="三歩目: 作り手に渡す語を選んで、この jsonl に書く")
    ap.add_argument("--n_words", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--talks", default=None, help="三歩目: 作った会話 <名前>=<会話.jsonl>,... を覚える側に足して数える")
    ap.add_argument("--pick_words4", default=None, help="四歩目: 覚える側と試し用の語を選んで、この jsonl に書く")
    ap.add_argument("--keep", default=None, help="四歩目: 前に作った会話（渡した語を新しく選ぶ語から除く）")
    ap.add_argument("--n_learn", type=int, default=450)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--learn_talks", default=None, help="四歩目: 覚える側の会話の jsonl（, 区切り、並べた順に 100・250・500 本で切る）")
    ap.add_argument("--test_talks", default=None, help="四歩目: 試し用の会話の jsonl")
    ap.add_argument("--tests5", default=None, help="五歩目: 試し用の会話 <名前>=<会話.jsonl>,...（作り手ごと）。覚える側は --learn_talks")
    ap.add_argument("--share_tests", default=None, help="六歩目: 会話の試し <名前>=<会話.jsonl>,...（童話の試しは --books から）。覚える側の会話は --learn_talks")
    ap.add_argument("--chain_tests", default=None, help="七歩目: 会話の試し <名前>=<会話.jsonl>,...（童話の試しは --books から）")
    a = ap.parse_args()
    if a.chain_tests:
        return main7(a)
    if a.share_tests:
        return main6(a)
    if a.tests5:
        return main5(a)
    if a.pick_words4:
        return pick_words4(a)
    if a.test_talks:
        return main4(a)
    if a.pick_words:
        return pick_words(a)
    if a.talks:
        return main3(a)
    if a.conllu:
        return main2(a)
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


# ---- 二歩目: 3 つ組を増やし、数えて覚える土台と比べる ----
SUBJ_CASE = ("が", "は")
ARG_CASE = (("obj", "を"), ("obl", "に"), ("obl", "で"))


def toks_of_doc(doc):
    """GiNZA の文 → [dict(i, l = 原形, p = 品詞, d = 係り, h = 親)]（conllu を読んだ物と同じ形）。"""
    return [dict(i=t.i - doc[0].i, l=t.lemma_, p=t.pos_, d=t.dep_, h=t.head.i - doc[0].i) for t in doc]


def triples2(toks):
    """3 つ組（主語の格 が・は、目的の席の格 を・に・で）と、そろわない時の 2 つ組。返す: [(種類, 主語の格, 目的の格, 語の並び)]。"""
    kids = {}
    for t in toks:
        if t["h"] != t["i"]:
            kids.setdefault(t["h"], []).append(t)

    def cases(t):
        return {c["l"] for c in kids.get(t["i"], []) if c["d"] == "case"}

    out = []
    for t in toks:
        if t["p"] != "VERB":
            continue
        ch = kids.get(t["i"], [])
        subj = next(((c, k) for c in ch if c["d"] == "nsubj" and c["p"] in NOUNS for k in SUBJ_CASE if k in cases(c)), None)
        args = []
        for dep, k in ARG_CASE:
            x = next((c for c in ch if c["d"] == dep and c["p"] in NOUNS and k in cases(c)), None)
            if x is not None:
                args.append((x, k))
        v = (t["l"], t["p"])
        if subj and args:
            for x, k in args:
                out.append(("三つ組", subj[1], k, ((subj[0]["l"], subj[0]["p"]), (x["l"], x["p"]), v)))
        elif subj:
            out.append(("主語と述語", subj[1], None, ((subj[0]["l"], subj[0]["p"]), v)))
        else:
            for x, k in args:
                out.append(("目的と述語", None, k, ((x["l"], x["p"]), v)))
    return out


def read_conllu(path, skip_ids, cap=2000):
    """conllu を作品ごとに読む（skip_ids の作品は読まない）。返す: {作品 ID: [文の toks]}（1 作品 cap 文まで）。"""
    works, cur, sent = {}, None, []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("# newdoc id = "):
                wid = ln.split("= ")[1].strip().split("-")[1]
                cur = None if wid in skip_ids else works.setdefault(wid, [])
                continue
            if ln.startswith("#"):
                continue
            if not ln.strip():
                if sent and cur is not None and len(cur) < cap:
                    cur.append(sent)
                sent = []
                continue
            c = ln.rstrip("\n").split("\t")
            if len(c) < 8 or "-" in c[0] or "." in c[0]:
                continue
            i = int(c[0]) - 1
            sent.append(dict(i=i, l=c[2], p=c[3], d=c[7], h=(int(c[6]) - 1) if c[6] != "0" else i))
    return works


class Tables:
    """覚える側の 3 つ組から、隠す席ごとに (見せた 2 語) → 隠した語、(述語側の 1 語) → 隠した語、(もう片方の 1 語) → 隠した語 の数。"""

    NEAR = {0: 2, 1: 2, 2: 1}                             # 隠す席 → 述語側の見せた語の席（だれが・何を → どうした、どうした → 何を）

    def __init__(self):
        self.t2 = [Counter() for _ in SLOTS]
        self.t1n = [Counter() for _ in SLOTS]
        self.t1o = [Counter() for _ in SLOTS]

    def add(self, tri):
        for si in range(3):
            sh = [j for j in range(3) if j != si]
            near = self.NEAR[si]
            other = next(j for j in sh if j != near)
            self.t2[si][(tri[sh[0]], tri[sh[1]], tri[si])] += 1
            self.t1n[si][(tri[near], tri[si])] += 1
            self.t1o[si][(tri[other], tri[si])] += 1

    def scores(self, si, tri, cands, backoff):
        """候補ごとの数と、使った表の段。上の表に 1 つでも数があればそれを使う。どれにも無ければ (None, 答えられない)。"""
        sh = [j for j in range(3) if j != si]
        near = self.NEAR[si]
        other = next(j for j in sh if j != near)
        levels = [("2語", lambda c: self.t2[si].get((tri[sh[0]], tri[sh[1]], c), 0))]
        if backoff:
            levels += [("述語側の1語", lambda c: self.t1n[si].get((tri[near], c), 0)),
                       ("もう片方の1語", lambda c: self.t1o[si].get((tri[other], c), 0))]
        for name, f in levels:
            s = {c: f(c) for c in cands}
            if any(v > 0 for v in s.values()):
                return s, name
        return None, "答えられない"


def rank_by(score, hidden):
    """数の多い順の順位（同じ数は前に置く）。"""
    h = score[hidden]
    return int(sum(1 for v in score.values() if v >= h))


def count2(items, vec, tables, seed=7):
    """二歩目の並べる数。目的の席の格ごと（を・に・で）と、まとめた物。"""
    rng = random.Random(seed)
    tb = Table(vec)
    out = {}
    groups = {"まとめ": items}
    for k in ("を", "に", "で"):
        groups[k] = [it for it in items if it["目的の格"] == k]
    for gname, its in groups.items():
        for si, slot in enumerate(SLOTS):
            shown = [j for j in range(3) if j != si]
            near = Tables.NEAR[si]
            for cname in ("ア_ページ", "イ_作品"):
                R = {k: [] for k in ("足した向き", "隣の1語の向き", "関係の無い2語", "数えた表_2語", "数えた表_下がる込み", "いちばん多い語")}
                N, level, no_ans = [], Counter(), 0
                for it in its:
                    tri, hid = it["組"], it["組"][si]
                    show = {tri[j] for j in shown}
                    pool = it["ページ"] if cname == "ア_ページ" else it["作品"]
                    cands = sorted({w for w in pool if w in vec and w not in show} | {hid}, key=str)
                    if len(cands) < 2:
                        continue
                    N.append(len(cands))
                    R["足した向き"].append(tb.rank_of(unit(vec[tri[shown[0]]]) + unit(vec[tri[shown[1]]]), hid, cands))
                    R["隣の1語の向き"].append(tb.rank_of(vec[tri[near]], hid, cands))
                    other = rng.choice([o for o in its if o is not it and o["文"] != it["文"]] or [it])["組"]
                    R["関係の無い2語"].append(tb.rank_of(unit(vec[other[shown[0]]]) + unit(vec[other[shown[1]]]), hid, cands))
                    for key, bo in (("数えた表_2語", False), ("数えた表_下がる込み", True)):
                        s, lv = tables.scores(si, tri, cands, bo)
                        if bo:
                            level[lv] += 1
                        elif s is None:
                            no_ans += 1
                        R[key].append(rank_by(s, hid) if s else len(cands))   # 答えられない時は全部同点 = いちばん後ろ
                    fc = Counter(w for w in it["ページの並び"] if w in vec and w not in show)
                    R["いちばん多い語"].append(rank_by({c: fc.get(c, 0) for c in cands}, hid))
                o = {k: summ(v, N) for k, v in R.items()}
                o["でたらめ"] = chance(N) if N else None
                o["2語の表で答えられなかった割合"] = round(no_ans / len(N), 4) if N else None
                o["下がる込みで使った表"] = dict(level)
                if len(N) < 100:
                    o["注意"] = "少ない（100 を切る）"
                out[f"{gname}・{slot}を隠す・{cname}"] = o
    return out


def prepare(a, nlp, wv, t0, read_learn=True):
    """試しの 3 つ組（作品 a.books）と、覚える側（conllu、試しの作品を除く）の 3 つ組。"""
    import i48_wave_board as W48
    import i49e_read_together as RT
    test_ids, items, kinds = set(), [], Counter()
    for rk in map(int, a.books.split(",")):
        info, S = RT.pick_book(a.src, a.meta, rank=rk)
        test_ids.add(info["作品ID"])
        docs = list(nlp.pipe(S, batch_size=128))
        page_seq = {}
        for si, d in enumerate(docs):
            page_seq.setdefault(si // RT.PAGE_LEN, []).extend(
                (t.lemma_, t.pos_) for t in d if t.pos_ in W48.CONTENT and t.lemma_ not in W48.STOP)
        work = {w for seq in page_seq.values() for w in seq}
        for si, d in enumerate(docs):
            for kind, sk, ak, ws in triples2(toks_of_doc(d)):
                kinds[f"試し・{kind}・{sk or '-'}・{ak or '-'}"] += 1
                if kind != "三つ組":
                    continue
                if any(wv.of(*w) is None for w in ws):
                    kinds["試し・三つ組・ベクトルが無い"] += 1
                    continue
                items.append(dict(組=ws, 主語の格=sk, 目的の格=ak, 文=(rk, si), ページの並び=page_seq[si // RT.PAGE_LEN],
                                  ページ=set(page_seq[si // RT.PAGE_LEN]), 作品=work))
    print("試し", len(items), round(time.time() - t0), flush=True)
    if not read_learn:
        return items, {}, kinds
    works = read_conllu(a.conllu, test_ids, cap=a.cap)
    learn = {}
    for wid, sents in works.items():
        L = []
        for s in sents:
            for kind, sk, ak, ws in triples2(s):
                kinds[f"覚える側・{kind}・{sk or '-'}・{ak or '-'}"] += 1
                if kind == "三つ組":
                    L.append(ws)
        learn[wid] = L
    print("覚える側", len(works), "作品", sum(len(v) for v in learn.values()), "組", round(time.time() - t0), flush=True)
    return items, learn, kinds


def vec_of_items(items, wv):
    vec = {}
    for it in items:
        for w in set(it["ページ"]) | set(it["作品"]):
            if w not in vec and wv.of(*w) is not None:
                vec[w] = wv.of(*w)
    return vec


def main2(a):
    import spacy
    import i49c_read_words as Rw
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    items, learn, kinds = prepare(a, nlp, wv, t0)
    vec = vec_of_items(items, wv)
    res = dict(試しの作品=a.books, 覚える側の作品=len(learn), 試しの三つ組=len(items), 取れた数=dict(sorted(kinds.items())))
    ids = sorted(learn)
    for size, sel in (("1/10", ids[::10]), ("全部", ids)):
        tables = Tables()
        for wid in sel:
            for tri in learn[wid]:
                tables.add(tri)
        res[f"覚える側 {size}"] = dict(作品=len(sel), 三つ組=sum(len(learn[w]) for w in sel), 数え=count2(items, vec, tables))
        print(size, round(time.time() - t0), flush=True)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---- 三歩目: 作った会話の 3 つ組を覚える側に足す ----
FIRST_HALF = 4                                            # 試しの作品の前半 = 並びの 1〜4 番目（--books の初めの 4 つ）


def pick_words(a):
    """作り手に渡す語を選ぶ。あ = 試しの前半の 3 つ組の、だれが・何を の名詞（どうした の問いで見せる側の語。答えを見て選ばない）。
    い = 語のベクトルを持つ内容語から seed 固定ででたらめに。各 a.n_words 語。"""
    import spacy
    import i48_wave_board as W48
    import i49c_read_words as Rw
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    items, _, _ = prepare(a, nlp, wv, t0, read_learn=False)
    first = {r for r in list(map(int, a.books.split(",")))[:FIRST_HALF]}
    seen, nouns = set(), []
    for it in items:
        if it["文"][0] in first:
            for w in it["組"][:2]:
                if w not in seen:
                    seen.add(w)
                    nouns.append(w)
    rng = random.Random(a.seed)
    rng.shuffle(nouns)
    names = [w for w, i in sorted(wv.ix.items(), key=lambda x: x[1])]
    pool = [w for w in names if isinstance(w, str)]
    rng2 = random.Random(a.seed)
    rand = []
    for nm in rng2.sample(pool, len(pool)):
        d = nlp(nm)
        if len(d) == 1 and d[0].pos_ in W48.CONTENT and d[0].lemma_ not in W48.STOP and wv.of(d[0].lemma_, d[0].pos_) is not None:
            rand.append((d[0].lemma_, d[0].pos_))
        if len(rand) >= a.n_words:
            break
    with open(a.pick_words, "w", encoding="utf-8") as f:
        for w in nouns[: a.n_words]:
            f.write(json.dumps({"原形": w[0], "品詞": w[1], "選び方": "あ"}, ensure_ascii=False) + "\n")
        for w in rand:
            f.write(json.dumps({"原形": w[0], "品詞": w[1], "選び方": "い"}, ensure_ascii=False) + "\n")
    print("あ", min(len(nouns), a.n_words), "（候補", len(nouns), "）い", len(rand))


def talk_triples(nlp, path, wv):
    """作った会話の jsonl → 選び方ごとの 3 つ組と、数え。"""
    import i48_wave_board as W48
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    by, stat = {}, {}
    for r in rows:
        g = r["語"]["選び方"]
        s = stat.setdefault(g, Counter())
        s["会話"] += 1
        s["発言"] += len(r["発言"])
        s["答えでない物"] += not r["発言"]
        for d in nlp.pipe(r["発言"], batch_size=64):
            s["述語"] += sum(1 for t in d if t.pos_ == "VERB")
            for t in d:
                if t.pos_ in W48.CONTENT and t.lemma_ not in W48.STOP:
                    s["内容語"] += 1
                    s["ベクトルを持つ内容語"] += wv.of(t.lemma_, t.pos_) is not None
                    by.setdefault(g + "_語", set()).add(t.lemma_)
            for kind, sk, ak, ws in triples2(toks_of_doc(d)):
                s[kind] += 1
                if kind == "三つ組":
                    by.setdefault(g, []).append(ws)
    return by, stat


def main3(a):
    import spacy
    import i49c_read_words as Rw
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    items, learn, kinds = prepare(a, nlp, wv, t0)
    vec = vec_of_items(items, wv)
    first = {r for r in list(map(int, a.books.split(",")))[:FIRST_HALF]}
    halves = {"前半": [it for it in items if it["文"][0] in first], "後半": [it for it in items if it["文"][0] not in first]}
    res = dict(試しの三つ組={k: len(v) for k, v in halves.items()}, 会話={})
    talk = {}
    for spec in a.talks.split(","):
        label, path = spec.split("=", 1)
        by, stat = talk_triples(nlp, path, wv)
        for g, st in stat.items():
            st = dict(st)
            st["1会話あたりの三つ組"] = round(st.get("三つ組", 0) / max(1, st["会話"]), 2)
            st["述語のうち三つ組の割合"] = round(st.get("三つ組", 0) / max(1, st["述語"]), 4)
            st["内容語のうちベクトルを持つ割合"] = round(st["ベクトルを持つ内容語"] / max(1, st["内容語"]), 4)
            words = by.get(g + "_語", set())
            st["漏れ"] = {h: {slot: round(float(np.mean([it["組"][si][0] in words for it in its])), 4) for si, slot in enumerate(SLOTS)}
                         for h, its in halves.items()}             # 試しの答えの語が、そのまま会話に出た割合
            res["会話"][f"{label}・{g}"] = st
            talk[f"{label}・{g}"] = by.get(g, [])
    base = Tables()
    for L in learn.values():
        for tri in L:
            base.add(tri)

    def brief(tables):
        out = {}
        for h, its in halves.items():
            c = count2(its, vec, tables)
            out[h] = {k.replace("まとめ・", ""): dict(一位=v["数えた表_下がる込み"].get("一位"), 三位以内=v["数えた表_下がる込み"].get("三位以内"),
                                                    二語の表だけ_一位=v["数えた表_2語"].get("一位"), 二語の表で答えられない=v["2語の表で答えられなかった割合"],
                                                    数=v["数えた表_下がる込み"].get("数"))
                      for k, v in c.items() if k.startswith("まとめ") and k.endswith("ア_ページ")}
        return out

    res["足す前（青空だけ）"] = brief(base)
    for name, tris in talk.items():
        plus = copy_tables(base)
        only = Tables()
        for tri in tris:
            plus.add(tri)
            only.add(tri)
        res[f"足した後（青空＋{name}）"] = brief(plus)
        res[f"作った会話だけ（{name}）"] = brief(only)
        res[f"作った会話だけ（{name}）"]["三つ組"] = len(tris)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


def copy_tables(t):
    c = Tables()
    c.t2 = [Counter(x) for x in t.t2]
    c.t1n = [Counter(x) for x in t.t1n]
    c.t1o = [Counter(x) for x in t.t1o]
    return c


# ---- 四歩目: 試しの側にも作った会話を置く ----
def random_words(nlp, wv, n, seed, exclude=()):
    """語のベクトルを持つ内容語（1 語で GiNZA が同じ原形・品詞を返す物）を seed 固定ででたらめに n 個。exclude の原形は除く。"""
    import i48_wave_board as W48
    pool = [w for w, _ in sorted(wv.ix.items(), key=lambda x: x[1]) if isinstance(w, str) and w not in exclude]
    out = []
    for nm in random.Random(seed).sample(pool, len(pool)):
        d = nlp(nm)
        if len(d) == 1 and d[0].pos_ in W48.CONTENT and d[0].lemma_ not in W48.STOP and wv.of(d[0].lemma_, d[0].pos_) is not None:
            out.append((d[0].lemma_, d[0].pos_))
        if len(out) >= n:
            break
    return out


def pick_words4(a):
    """覚える側 a.n_learn 語（seed a.seed）と、試し用 a.n_test 語（seed a.seed + 1、覚える側の語と、--keep の会話で渡した語を除く）。"""
    import spacy
    import i49c_read_words as Rw
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    keep = [json.loads(l) for l in open(a.keep, encoding="utf-8")] if a.keep else []
    used = {r["語"]["原形"] for r in keep}
    learn = random_words(nlp, wv, a.n_learn, a.seed, exclude=used)
    test = random_words(nlp, wv, a.n_test, a.seed + 1, exclude=used | {w for w, _ in learn})
    with open(a.pick_words4, "w", encoding="utf-8") as f:
        for g, ws in (("覚える側", learn), ("試し用", test)):
            for w in ws:
                f.write(json.dumps({"原形": w[0], "品詞": w[1], "選び方": g}, ensure_ascii=False) + "\n")
    print("覚える側", len(learn), "試し用", len(test))


def talk_items(nlp, rows, wv):
    """作った会話（1 本 = 1 ページ）→ 試しの 3 つ組（候補 = その会話の内容語）と、落ちた数。"""
    import i48_wave_board as W48
    items, kinds = [], Counter()
    for ti, r in enumerate(rows):
        docs = list(nlp.pipe(r["発言"], batch_size=64))
        seq = [(t.lemma_, t.pos_) for d in docs for t in d if t.pos_ in W48.CONTENT and t.lemma_ not in W48.STOP]
        for d in docs:
            for kind, sk, ak, ws in triples2(toks_of_doc(d)):
                if kind != "三つ組":
                    continue
                kinds["三つ組"] += 1
                if any(wv.of(*w) is None for w in ws):
                    kinds["ベクトルが無い"] += 1
                    continue
                items.append(dict(組=ws, 主語の格=sk, 目的の格=ak, 文=(ti, len(items)), ページの並び=seq, ページ=set(seq), 作品=set(seq)))
    return items, kinds


def main4(a):
    import spacy
    import i49c_read_words as Rw
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    dou, learn, _ = prepare(a, nlp, wv, t0)
    rows = [json.loads(l) for f in a.learn_talks.split(",") for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r["語"]["選び方"] in ("い", "覚える側")]          # 三歩目の あ（試しの童話から選んだ語）は使わない
    test_rows = [json.loads(l) for l in open(a.test_talks, encoding="utf-8")]
    talk_tris = []
    for r in rows:
        tris = []
        for d in nlp.pipe(r["発言"], batch_size=64):
            tris += [ws for kind, _, _, ws in triples2(toks_of_doc(d)) if kind == "三つ組"]
        talk_tris.append(tris)
    kai, kai_kinds = talk_items(nlp, test_rows, wv)
    print("童", len(dou), "会", len(kai), "覚える会話", len(rows), round(time.time() - t0), flush=True)
    vec = vec_of_items(dou, wv)
    vec.update(vec_of_items(kai, wv))
    learned = [tri for tris in talk_tris for tri in tris]
    pairs = {frozenset([(i, tri[i]), (j, tri[j])]) for tri in learned for i in range(3) for j in range(i + 1, 3)}
    full = set(learned)
    res = dict(覚える会話=len(rows), 覚える会話の三つ組=len(learned), 試し用の会話=len(test_rows), 童=len(dou), 会=len(kai),
               会の取れた数=dict(kai_kinds), 会のベクトルが無くて落ちた割合=round(kai_kinds["ベクトルが無い"] / max(1, kai_kinds["三つ組"]), 4),
               漏れ=dict(同じ三つ組=round(float(np.mean([it["組"] in full for it in kai])), 4) if kai else None,
                       同じ2語の組=round(float(np.mean([any(frozenset([(i, it["組"][i]), (j, it["組"][j])]) in pairs
                                                       for i in range(3) for j in range(i + 1, 3)) for it in kai])), 4) if kai else None))
    aoz = Tables()
    for L in learn.values():
        for tri in L:
            aoz.add(tri)

    def brief(tables, items):
        c = count2(items, vec, tables)
        keep = ("足した向き", "隣の1語の向き", "関係の無い2語", "数えた表_2語", "数えた表_下がる込み", "いちばん多い語")
        return {k.replace("まとめ・", ""): dict({m: (v[m].get("一位"), v[m].get("三位以内")) for m in keep},
                                              でたらめ=(v["でたらめ"] or {}).get("一位"), 数=v["足した向き"].get("数"),
                                              二語の表で答えられない=v["2語の表で答えられなかった割合"], 使った表=v["下がる込みで使った表"],
                                              注意=v.get("注意", ""))
                for k, v in c.items() if k.startswith("まとめ") and k.endswith("ア_ページ")}

    for tname, items in (("童", dou), ("会", kai)):
        res[f"{tname}・青空だけ"] = brief(aoz, items)
        for n in (100, 250, 500):
            only, both = Tables(), copy_tables(aoz)
            for tris in talk_tris[:n]:
                for tri in tris:
                    only.add(tri)
                    both.add(tri)
            res[f"{tname}・会話だけ {n} 本"] = brief(only, items)
            res[f"{tname}・両方 {n} 本"] = brief(both, items)
        print(tname, round(time.time() - t0), flush=True)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---- 五歩目: 作り手の癖を覚えただけかを見る ----
def hits_by_pair(items, tables, pairs, vec):
    """席ごとに、見せた 2 語の組が覚える側にあった問い・無かった問いで分けた、数えた表（下がる込み）の 1 位の割合。"""
    out = {}
    for si, slot in enumerate(SLOTS):
        shown = [j for j in range(3) if j != si]
        g = {"組があった": [], "組が無かった": []}
        for it in items:
            tri, hid = it["組"], it["組"][si]
            show = {tri[j] for j in shown}
            cands = sorted({w for w in it["ページ"] if w in vec and w not in show} | {hid}, key=str)
            if len(cands) < 2:
                continue
            s, _ = tables.scores(si, tri, cands, True)
            hit = bool(s) and rank_by(s, hid) == 1
            key = frozenset([(shown[0], tri[shown[0]]), (shown[1], tri[shown[1]])])
            g["組があった" if key in pairs else "組が無かった"].append(hit)
        out[slot] = {k: dict(問い=len(v), 一位=round(float(np.mean(v)), 4) if v else None) for k, v in g.items()}
    return out


def main5(a):
    import spacy
    import i49c_read_words as Rw
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    dou, learn, _ = prepare(a, nlp, wv, t0)
    rows = [json.loads(l) for f in a.learn_talks.split(",") for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r["語"]["選び方"] in ("い", "覚える側")]
    learned = []
    for r in rows:
        for d in nlp.pipe(r["発言"], batch_size=64):
            learned += [ws for kind, _, _, ws in triples2(toks_of_doc(d)) if kind == "三つ組"]
    pairs = {frozenset([(i, tri[i]), (j, tri[j])]) for tri in learned for i in range(3) for j in range(i + 1, 3)}
    aoz, only = Tables(), Tables()
    for L in learn.values():
        for tri in L:
            aoz.add(tri)
    for tri in learned:
        only.add(tri)
    both = copy_tables(aoz)
    for tri in learned:
        both.add(tri)
    res = dict(覚える会話=len(rows), 覚える会話の三つ組=len(learned), 試し={})
    for spec in a.tests5.split(","):
        name, path = spec.split("=", 1)
        trows = [json.loads(l) for l in open(path, encoding="utf-8")]
        kai, kinds = talk_items(nlp, trows, wv)
        vec = vec_of_items(kai, wv)
        o = dict(会話=len(trows), 答えでない物=sum(1 for r in trows if not r["発言"]), 取れた数=dict(kinds), 使えた=len(kai),
                 ベクトルが無くて落ちた割合=round(kinds["ベクトルが無い"] / max(1, kinds["三つ組"]), 4))
        o["同じ2語の組が覚える側にある割合"] = round(float(np.mean([any(frozenset([(i, it["組"][i]), (j, it["組"][j])]) in pairs
                                                       for i in range(3) for j in range(i + 1, 3)) for it in kai])), 4) if kai else None
        o["同じ三つ組が覚える側にある割合"] = round(float(np.mean([it["組"] in set(learned) for it in kai])), 4) if kai else None
        for tname, tb in (("青空だけ", aoz), ("会話だけ", only), ("両方", both)):
            c = count2(kai, vec, tb)
            o[tname] = {k.split("・")[1]: dict(表下がる込み=(v["数えた表_下がる込み"].get("一位"), v["数えた表_下がる込み"].get("三位以内")),
                                               でたらめ=(v["でたらめ"] or {}).get("一位"),
                                               いちばん多い語=(v["いちばん多い語"].get("一位"), v["いちばん多い語"].get("三位以内")),
                                               二語の表で答えられた=round(1 - v["2語の表で答えられなかった割合"], 4) if v["2語の表で答えられなかった割合"] is not None else None,
                                               数=v["足した向き"].get("数"))
                        for k, v in c.items() if k.startswith("まとめ") and k.endswith("ア_ページ")}
            if tname != "青空だけ":
                o[tname + "_組の有無で分けた一位"] = hits_by_pair(kai, tb, pairs, vec)
        subj = Counter(it["組"][0][0] for it in kai)
        top = subj.most_common(5)
        o["だれがの答え"] = dict(種類=len(subj), 上位5語の割合=round(sum(n for _, n in top) / max(1, len(kai)), 4), 上位5語=top)
        res["試し"][name] = o
        print(name, round(time.time() - t0), flush=True)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---- 六歩目: 似た語どうしで数を分け合う（持ち主「学習を増やせないなぁ」） ----
class Share:
    """覚えた 3 つ組を、語の向きの近さで分け合って使う口。
    b: 点(候補) = Σ_t 近さ(見せた語1, t の語1) × 近さ(見せた語2, t の語2) × [t の答え = 候補]
    c: 点(候補) = Σ_t 近さ1 × 近さ2 × 近さ(候補, t の答え)
    d: 波の板。席ごとに 板 = Σ_t 結び(t の語1, t の語2) ⊙ t の答え（結び = 席ごとの並べ替えを掛けた向きの要素ごとの積）。
       問いは 板 ⊙ conj(結び(見せた語1, 見せた語2)) に近い候補を選ぶ。近さはどれも |cos|。切る線・上位 k 個・前処理は置かない。"""

    def __init__(self, tris, vec, ask_words, seed=7):
        words = sorted({w for t in tris for w in t if w in vec}, key=str)
        self.L = {w: i for i, w in enumerate(words)}
        VL = np.stack([unit(vec[w]) for w in words]).astype(np.complex64)
        self.tris = [t for t in tris if all(w in self.L for w in t)]
        self.dropped = len(tris) - len(self.tris)
        self.T = np.array([[self.L[w] for w in t] for t in self.tris], dtype=np.int64).reshape(-1, 3)
        ask = sorted({w for w in ask_words if w in vec}, key=str)
        self.U = {w: i for i, w in enumerate(ask)}
        VU = np.stack([unit(vec[w]) for w in ask]).astype(np.complex64)
        self.S = np.zeros((len(ask), len(words)), dtype=np.float32)       # 近さ（問いの側の語 × 覚えた語）
        for s in range(0, len(ask), 512):
            self.S[s:s + 512] = np.abs(np.conj(VU[s:s + 512]) @ VL.T)
        dim = VL.shape[1]
        rng = np.random.default_rng(seed)
        self.perm = [rng.permutation(dim) for _ in range(3)]               # 席ごとの並べ替え
        self.VU = VU
        self.board = []
        for si in range(3):
            j, k = [x for x in range(3) if x != si]
            B = np.zeros(dim, dtype=np.complex64)
            for s in range(0, len(self.T), 4096):
                t = self.T[s:s + 4096]
                B += (VL[t[:, j]][:, self.perm[j]] * VL[t[:, k]][:, self.perm[k]] * VL[t[:, si]]).sum(0)
            self.board.append(B)

    def scores(self, kind, si, tri, cands):
        j, k = [x for x in range(3) if x != si]
        if tri[j] not in self.U or tri[k] not in self.U:
            return None
        if kind == "d":
            q = self.VU[self.U[tri[j]]][self.perm[j]] * self.VU[self.U[tri[k]]][self.perm[k]]
            r = self.board[si] * np.conj(q)
            return {c: float(np.abs(np.vdot(self.VU[self.U[c]], r))) if c in self.U else 0.0 for c in cands}
        w = self.S[self.U[tri[j]], self.T[:, j]] * self.S[self.U[tri[k]], self.T[:, k]]
        agg = np.bincount(self.T[:, si], weights=w, minlength=len(self.L))
        if kind == "b":
            return {c: float(agg[self.L[c]]) if c in self.L else 0.0 for c in cands}
        return {c: float(self.S[self.U[c]] @ agg) if c in self.U else 0.0 for c in cands}


def share_count(items, vec, tables, share, seed=7):
    """席ごとに、a 今の表（下がる込み）・b・c（2 語の表に組があればそれを使い、無ければ分け合う）・d 波の板、でたらめ、いちばん多い語。
    2 語の表で答えられなかった問いだけの 1 位も。"""
    out = {}
    for si, slot in enumerate(SLOTS):
        shown = [j for j in range(3) if j != si]
        R = {k: [] for k in ("a_今の表", "b_分け合う", "c_答えも分け合う", "d_波の板", "いちばん多い語")}
        miss = {k: [] for k in ("a_今の表", "b_分け合う", "c_答えも分け合う", "d_波の板", "いちばん多い語")}
        N, Nm, cannot = [], [], 0
        for it in items:
            tri, hid = it["組"], it["組"][si]
            show = {tri[j] for j in shown}
            cands = sorted({w for w in it["ページ"] if w in vec and w not in show} | {hid}, key=str)
            if len(cands) < 2:
                continue
            N.append(len(cands))
            s2, _ = tables.scores(si, tri, cands, False)
            sa, _ = tables.scores(si, tri, cands, True)
            ranks = {"a_今の表": rank_by(sa, hid) if sa else len(cands)}
            for key, kind in (("b_分け合う", "b"), ("c_答えも分け合う", "c"), ("d_波の板", "d")):
                s = s2 if (s2 and kind != "d") else share.scores(kind, si, tri, cands)
                if s is None:
                    cannot += kind == "b"
                ranks[key] = rank_by(s, hid) if s else len(cands)
            fc = Counter(w for w in it["ページの並び"] if w in vec and w not in show)
            ranks["いちばん多い語"] = rank_by({c: fc.get(c, 0) for c in cands}, hid)
            for key, r in ranks.items():
                R[key].append(r)
                if s2 is None:
                    miss[key].append(r)
            if s2 is None:
                Nm.append(len(cands))
        o = {k: summ(v, N) for k, v in R.items()}
        o["でたらめ"] = chance(N) if N else None
        o["2語の表で答えられなかった問いだけ"] = dict({k: summ(v, Nm) for k, v in miss.items()}, でたらめ=chance(Nm) if Nm else None)
        o["見せた語にベクトルが無くて分け合えなかった問い"] = cannot
        out[slot] = o
    return out


def main6(a):
    import spacy
    import i49c_read_words as Rw
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    dou, learn, _ = prepare(a, nlp, wv, t0)
    rows = [json.loads(l) for f in a.learn_talks.split(",") for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r["語"]["選び方"] in ("い", "覚える側")]
    talk_tris = []
    for r in rows:
        tris = []
        for d in nlp.pipe(r["発言"], batch_size=64):
            tris += [ws for kind, _, _, ws in triples2(toks_of_doc(d)) if kind == "三つ組"]
        talk_tris.append(tris)
    tests = {"童": dou}
    for spec in a.share_tests.split(","):
        name, path = spec.split("=", 1)
        tests[name], _ = talk_items(nlp, [json.loads(l) for l in open(path, encoding="utf-8")], wv)
    vec = {}
    for its in tests.values():
        vec.update(vec_of_items(its, wv))
    ids = sorted(learn)
    res = dict(試し={k: len(v) for k, v in tests.items()})
    for size, sel, n_talk in (("1/10", ids[::10], 50), ("全部", ids, len(talk_tris))):
        aoz_tris = [tri for w in sel for tri in learn[w]]
        talk = [tri for tris in talk_tris[:n_talk] for tri in tris]
        for w in {w for t in aoz_tris + talk for w in t}:
            if w not in vec and wv.of(*w) is not None:
                vec[w] = wv.of(*w)
        ask_words = {w for its in tests.values() for it in its for w in it["ページ"]}
        for lname, tris, tnames in (("青空", aoz_tris, ["童"]), ("青空＋会話", aoz_tris + talk, [k for k in tests if k != "童"])):
            tables = Tables()
            for tri in tris:
                tables.add(tri)
            share = Share(tris, vec, ask_words, seed=a.seed)
            print(size, lname, "三つ組", len(tris), "分け合えない（ベクトル無し）", share.dropped, round(time.time() - t0), flush=True)
            for tn in tnames:
                res[f"{size}・覚える側 {lname}・試し {tn}"] = dict(覚える三つ組=len(tris), ベクトルが無くて分け合えない三つ組=share.dropped,
                                                             数え=share_count(tests[tn], vec, tables, share))
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---- 七歩目: 借りる段を、今の表の下がり方のどこに入れるか（切る線・上位 k 個は置かない） ----
def inv_rank(x):
    """近さの並び → 近い順に 1, 1/2, 1/3 … の重み（同じ値は前に置く順位）。"""
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1)
    return 1.0 / r


def borrow(share, si, tri, cands, how="both", weight="cos"):
    """借りる段。how = both（見せた 2 語とも似た語で分け合う）／ one（述語側の語は同じ物に限り、もう片方だけ分け合う）。
    weight = cos（|cos| を掛ける）／ rank（近さの順位 1, 1/2, … を掛ける）。数が全部 0 なら None。"""
    j, k = [x for x in range(3) if x != si]
    near = Tables.NEAR[si]
    other = k if near == j else j
    if tri[j] not in share.U or tri[k] not in share.U:
        return None
    sim = {p: share.S[share.U[tri[p]]].astype(np.float64) for p in (j, k)}
    if weight == "rank":
        sim = {p: inv_rank(v) for p, v in sim.items()}
    if how == "both":
        w = sim[j][share.T[:, j]] * sim[k][share.T[:, k]]
    else:
        if tri[near] not in share.L:
            return None
        w = (share.T[:, near] == share.L[tri[near]]) * sim[other][share.T[:, other]]
    agg = np.bincount(share.T[:, si], weights=w, minlength=len(share.L))
    s = {c: float(agg[share.L[c]]) if c in share.L else 0.0 for c in cands}
    return s if any(v > 0 for v in s.values()) else None


CHAINS = {
    "a_今の表": ("2語", "述語側", "もう片方"),
    "e_2語の次に借りる": ("2語", "借りる", "述語側", "もう片方"),
    "e_順位で借りる": ("2語", "借りる_順位", "述語側", "もう片方"),
    "f_白紙の時だけ借りる": ("2語", "述語側", "もう片方", "借りる"),
    "g_片方だけ借りる": ("2語", "片方だけ借りる", "述語側", "もう片方", "借りる"),
    "h_いつも借りる": ("2語", "借りる"),
}


def chain_scores(tables, share, si, tri, cands, chain):
    sh = [j for j in range(3) if j != si]
    near = Tables.NEAR[si]
    other = next(j for j in sh if j != near)
    for lv in chain:
        if lv == "2語":
            s = {c: tables.t2[si].get((tri[sh[0]], tri[sh[1]], c), 0) for c in cands}
        elif lv == "述語側":
            s = {c: tables.t1n[si].get((tri[near], c), 0) for c in cands}
        elif lv == "もう片方":
            s = {c: tables.t1o[si].get((tri[other], c), 0) for c in cands}
        elif lv == "借りる":
            s = borrow(share, si, tri, cands, "both", "cos")
        elif lv == "借りる_順位":
            s = borrow(share, si, tri, cands, "both", "rank")
        else:
            s = borrow(share, si, tri, cands, "one", "cos")
        if s and any(v > 0 for v in s.values()):
            return s, lv
    return None, "答えられない"


def chain_count(items, vec, tables, share):
    out = {}
    for si, slot in enumerate(SLOTS):
        shown = [j for j in range(3) if j != si]
        o = {}
        N, freq = [], []
        per = {k: dict(r=[], lv=Counter(), hit=Counter()) for k in CHAINS}
        for it in items:
            tri, hid = it["組"], it["組"][si]
            show = {tri[j] for j in shown}
            cands = sorted({w for w in it["ページ"] if w in vec and w not in show} | {hid}, key=str)
            if len(cands) < 2:
                continue
            N.append(len(cands))
            for name, chain in CHAINS.items():
                s, lv = chain_scores(tables, share, si, tri, cands, chain)
                r = rank_by(s, hid) if s else len(cands)
                per[name]["r"].append(r)
                per[name]["lv"][lv] += 1
                per[name]["hit"][lv] += r == 1
            fc = Counter(w for w in it["ページの並び"] if w in vec and w not in show)
            freq.append(rank_by({c: fc.get(c, 0) for c in cands}, hid))
        for name, p in per.items():
            o[name] = dict(summ(p["r"], N), 答えた段=dict(p["lv"]),
                           段ごとの一位={lv: round(p["hit"][lv] / n, 4) for lv, n in p["lv"].items() if n})
        o["いちばん多い語"] = summ(freq, N)
        o["でたらめ"] = chance(N) if N else None
        out[slot] = o
    return out


def main7(a):
    import spacy
    import i49c_read_words as Rw
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    dou, learn, _ = prepare(a, nlp, wv, t0)
    rows = [json.loads(l) for f in a.learn_talks.split(",") for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r["語"]["選び方"] in ("い", "覚える側")]
    talk_tris = []
    for r in rows:
        tris = []
        for d in nlp.pipe(r["発言"], batch_size=64):
            tris += [ws for kind, _, _, ws in triples2(toks_of_doc(d)) if kind == "三つ組"]
        talk_tris.append(tris)
    tests = {"童": dou}
    for spec in a.chain_tests.split(","):
        name, path = spec.split("=", 1)
        tests[name], _ = talk_items(nlp, [json.loads(l) for l in open(path, encoding="utf-8")], wv)
    vec = {}
    for its in tests.values():
        vec.update(vec_of_items(its, wv))
    ids = sorted(learn)
    all_aoz = [tri for w in ids for tri in learn[w]]
    novec = Counter()
    for tri in all_aoz:                                      # ベクトルの無い語の品詞の内訳（語そのものは書かない）
        for w in tri:
            if wv.of(*w) is None:
                novec[w] += 1
    by_pos = Counter()
    for (l, p), n in novec.items():
        by_pos[p] += n
    top = novec.most_common(20)
    res = dict(試し={k: len(v) for k, v in tests.items()},
               ベクトルの無い語=dict(異なり=len(novec), 延べ=sum(novec.values()), 品詞ごとの延べ=dict(by_pos),
                                    品詞ごとの異なり=dict(Counter(p for _, p in novec)),
                                    上位20語の品詞と数=[(w[1], n) for w, n in top],
                                    上位20語が延べに占める割合=round(sum(n for _, n in top) / max(1, sum(novec.values())), 4),
                                    ベクトルの無い語を含む三つ組=sum(1 for t in all_aoz if any(wv.of(*w) is None for w in t))))
    for size, sel, n_talk in (("1/10", ids[::10], 50), ("全部", ids, len(talk_tris))):
        aoz_tris = [tri for w in sel for tri in learn[w]]
        talk = [tri for tris in talk_tris[:n_talk] for tri in tris]
        for w in {w for t in aoz_tris + talk for w in t}:
            if w not in vec and wv.of(*w) is not None:
                vec[w] = wv.of(*w)
        ask_words = {w for its in tests.values() for it in its for w in it["ページ"]}
        for lname, tris, tnames in (("青空", aoz_tris, ["童"]), ("青空＋会話", aoz_tris + talk, [k for k in tests if k != "童"])):
            tables = Tables()
            for tri in tris:
                tables.add(tri)
            share = Share(tris, vec, ask_words, seed=a.seed)
            for tn in tnames:
                res[f"{size}・覚える側 {lname}・試し {tn}"] = chain_count(tests[tn], vec, tables, share)
            print(size, lname, round(time.time() - t0), flush=True)
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
