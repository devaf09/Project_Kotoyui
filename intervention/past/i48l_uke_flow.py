"""I48l: 受けの一言（相手の話への相槌・共感の、行の始めの一言）を、相手の前の行から波で当てる試し（語板 v3 の L3 の続き）。

I48k の (b): 先生の言い方の指摘の 25% が「受け・共感が無い」、12% が「返事の始めの相槌の選び方」だった。
今の bot は、問いの型と返し方（相槌・肯定・否定）ごとに決まった感動詞を 1 つだけ選ぶ（chat_bot2.intj）。相手の行の中身を見ない。

受けの一言の取り方（一覧は手で置かない）:
  行の始めから最初の区切り（、。！？!?…）までのかたまりで、区切りの後にも行が続き、4 語以下で、名詞・代名詞を含まない物。
  学びの資料で min_count 回以上出たかたまりを候補にし、それより少ない物は「ほかの受け」、無い行は「受け無し」とする。
  かたまりは語の形のまま（そうなんですね と そうなんだ は別）。丁寧さも受けの選び方の一部なので。
相手の前の行の手がかり（鍵）:
  尾   : 相手の行の終わりの述語の後ろのつなぎ（語尾。I48e の「述語の後ろ」と同じ取り方）
  中   : 相手の行の中身の語（名詞・動詞・形容詞など）
  印   : 問いか、感嘆（！）か、形容詞があるか、否定があるか、相手の行に受けがあったか、自分の前の行に受けがあったか、相手の行の長さ（中身の語の数の段）
RealPersonaChat は 2 人の対話で行が交互なので、1 つ前の行 = 相手、2 つ前 = 自分。

比べる物: いちばん多い受け、同じ手がかりの数える表、相手の行の語尾だけ、相手の行の中身の語だけ。

実行: python intervention/past/i48l_uke_flow.py --cache <parsed.json> --D 512 --seed 0 --out i48l.json [--device cuda:0]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i48_wave_board import Waves  # noqa: E402
from i48e_relation_split import CONTENT, CUE, FUNC, NEG, NOUNS, NONE, PREDS, CountTable, Relation, chain, is_question, mark  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

log = logging.getLogger("i48l")
BREAK = ("、", "。", "！", "？", "!", "?", "…", "‥", "，", ",")
NO_UKE = "受け無し"
OTHER = "ほかの受け"
MARKS = ("問い", "感嘆", "形容詞", "否定", "相手の受け", "自分の受け", "長さ")


def body(toks):
    return [t for t in toks if t["text"] not in ("「", "」", "『", "』")]


def uke(toks, max_len=4):
    """行の始めの受けの一言（無ければ None）。"""
    T = body(toks)
    for j, t in enumerate(T):
        if t["text"] in BREAK:
            head = T[:j]
            rest = [x for x in T[j + 1:] if x["text"] not in BREAK]
            if head and rest and len(head) <= max_len and not any(x["pos"] in NOUNS for x in head):
                return "".join(x["text"] for x in head)
            return None
    return None


def tail(toks):
    """行の終わりの述語の後ろのつなぎ（語尾）。述語が無ければ ∅。"""
    T = body(toks)
    idx = [i for i, t in enumerate(T) if t["pos"] in PREDS]
    if not idx:
        return NONE
    i = idx[-1]
    j = i + 1
    while j < len(T) and T[j]["pos"] in FUNC:
        j += 1
    return chain(T[i + 1:j])


def samples(work):
    """対話 1 本 → 行ごとの場面（2 行目から）。post = その行の受け（候補に寄せる前）。"""
    L = work["lines"]
    uk = [uke(ln["toks"]) for ln in L]
    out = []
    for li in range(1, len(L)):
        P = L[li - 1]
        T = body(P["toks"])
        cont = [t["lemma"] for t in T if t["pos"] in CUE]
        n = sum(t["pos"] in CONTENT for t in T)
        marks = {"問い": is_question(P), "感嘆": any(t["text"] in ("！", "!") for t in T),
                 "形容詞": any(t["pos"] == "ADJ" for t in T),
                 "否定": any(t["lemma"] in NEG and t["pos"] in ("AUX", "ADJ", "PART") for t in T),
                 "相手の受け": uk[li - 1] is not None, "自分の受け": li >= 2 and uk[li - 2] is not None,
                 "長さ": "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else "6+"}
        out.append(dict(w=NONE, pos="受け", host="受け", post=uk[li] or NO_UKE, tail=tail(P["toks"]), cues=cont, marks=marks,
                        flags={}, head=NONE, dep="?", nxt=NONE, npost=NONE, li=li))
    return out


def to_cands(S, cands):
    cs = set(cands)
    for s in S:
        if s["post"] not in cs:
            s["post"] = OTHER
    return S


class UkeRelation(Relation):
    """鍵は相手の前の行だけで作る（尾・中・印、と 尾 × 印 の対）。どの手がかりも 1 つ分の重さ。"""

    ROLES = Relation.ROLES + ("尾", "中", "受印", "尾印", "中印")

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.mk = {}

    def m(self, x):
        if x not in self.mk:
            self.mk[x] = mark(x, self.D)
        return self.mk[x]

    def key(self, s):
        k, R, parts = self.keys, self.role, []
        tv = self.m("尾:" + s["tail"])
        mv = None
        if "印" in k:
            mv = sum(self.m(f"受印:{f}:{s['marks'][f]}") for f in MARKS)
            mv = mv / (np.linalg.norm(mv) / math.sqrt(self.D) + 1e-9)
            parts.append(mv * R["受印"])
        if "尾" in k:
            parts.append(tv * R["尾"])
            if mv is not None:
                parts.append(tv * mv * R["尾印"])
        if "中" in k:
            b = self.bag(s["cues"])
            if b is not None:
                parts.append(b * R["中"])
                if mv is not None:
                    parts.append(b * mv * R["中印"])
        if not parts:
            parts.append(self.m("空"))
        v = sum(parts)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.complex64)


class UkeTable(CountTable):
    def fit(self, S):
        super().fit(S)
        self.pr = np.maximum(self.pr, 1e-9)                          # 2 段目（受けのある場面だけ）は 受け無し・ほかの受け が 0 になるので、0 割りを避ける
        self.pr = self.pr / self.pr.sum()
        return self

    def feats(self, s):
        k, f = self.keys, []
        if "尾" in k:
            f.append("尾:" + s["tail"])
        if "中" in k:
            f += ["中:" + c for c in set(s["cues"])]
        if "印" in k:
            f += [f"印:{x}:{s['marks'][x]}" for x in MARKS]
        return f


def auroc(score, y):
    """y = 1 の方が score が高い割合（同点は半分）。"""
    o = np.argsort(score, kind="mergesort")
    r = np.empty(len(score))
    r[o] = np.arange(1, len(score) + 1)
    s_ = np.asarray(score)
    for v in np.unique(s_):                                          # 同点は平均の順位
        m = s_ == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = int(y.sum()), int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)


def measure(SC, te, cands, SCw=None):
    """2 つに分けて測る。受けを付けるか（受け無しの見込みの低さで、付ける行を見分けられるか = AUROC）と、
    受けのある行で、どの受けか（受け無しを除いた候補の中の 1 番目・3 番目まで）。"""
    cix = {c: i for i, c in enumerate(cands)}
    t = np.array([cix[s["post"]] for s in te])
    P = SC / SC.sum(1, keepdims=True)
    none = cix[NO_UKE]
    has = t != none
    r = dict(top1=round(float((P.argmax(1) == t).mean()), 4))
    r["有無 AUROC"] = round(float(auroc(1 - P[:, none], has)), 4)
    real = has & (t != cix[OTHER])                                    # 候補にある受けの行だけ（ほかの受けを除く）
    Q = (SCw / SCw.sum(1, keepdims=True)) if SCw is not None else P.copy()   # どの受けか: 受けのある場面だけで作った板（2 段目）があればそちらで
    Q[:, none] = -1
    Q[:, cix[OTHER]] = -1                                             # どの受けかは、名前のある候補の中から選ぶ
    o = np.argsort(-Q, 1)[:, :3]
    r["どの受けか 1 番目"] = round(float((o[real, 0] == t[real]).mean()), 4)
    r["どの受けか 3 番目まで"] = round(float((o[real] == t[real, None]).any(1).mean()), 4)
    r["受けの行の数"] = int(real.sum())
    return r


VARIANTS = (
    ("相手の行の語尾だけ", ("尾",)),
    ("相手の行の中身の語だけ", ("中",)),
    ("印だけ", ("印",)),
    ("語尾・印", ("尾", "印")),
    ("中身の語・印", ("中", "印")),
    ("語尾・中身の語・印", ("尾", "中", "印")),
)
TABLES = (
    ("数える表 語尾", ("尾",)),
    ("数える表 中身の語", ("中",)),
    ("数える表 印", ("印",)),
    ("数える表 語尾・印", ("尾", "印")),
    ("数える表 語尾・中身の語・印", ("尾", "中", "印")),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="解析済みの会話の資料（works: [{name, lines: [{text, toks}]}]）")
    ap.add_argument("--test_works", type=int, default=500)
    ap.add_argument("--train_works", type=int, default=0)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--min_count", type=int, default=20, help="受けの候補にする回数（学び）")
    ap.add_argument("--max_scenes", type=int, default=200000)
    ap.add_argument("--test_n", type=int, default=4000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--board", help="受けの板を、資料の全部で作って、この名前（拡張子なし）で書き出す（測りはしない）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    if a.board:
        dev = torch.device(a.device) if (a.device and torch is not None) else None
        bd = UkeBoard.build(works, a.D, a.seed, a.min_count, a.max_scenes, dev)
        bd.save(a.board)
        log.info("受けの板: 候補 %d、1 段目 %d 場面、2 段目 %d 場面、%.0f 秒", len(bd.cands), len(bd.r1.P), len(bd.r2.P), time.time() - t0)
        return
    random.Random(a.seed).shuffle(works)
    te_w, tr_w = works[: a.test_works], works[a.test_works:]
    if a.train_works:
        tr_w = tr_w[: a.train_works]
    tr = [s for w in tr_w for s in samples(w)]
    cnt = Counter(s["post"] for s in tr)
    cands = [c for c, n in cnt.most_common() if n >= a.min_count and c != NO_UKE] + [OTHER, NO_UKE]
    tr = to_cands(tr, cands)
    te = to_cands([s for w in te_w for s in samples(w)], cands)
    te = random.Random(a.seed + 1).sample(te, min(a.test_n, len(te)))
    share = Counter(s["post"] for s in te)
    log.info("場面: 学び %d（%d 対話）、試し %d。候補 %d（受け %d ＋ほかの受け・受け無し）。試しで受け無し %.3f、ほかの受け %.3f",
             len(tr), len(tr_w), len(te), len(cands), len(cands) - 2, share[NO_UKE] / len(te), share[OTHER] / len(te))
    log.info("多い受け: %s", " ".join(f"{c}({cnt[c]})" for c in cands[:30]))
    vocab = sorted({t["lemma"] for w in tr_w for ln in w["lines"] for t in ln["toks"] if t["pos"] in CUE})
    seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in CUE] for w in tr_w for ln in w["lines"] + [dict(toks=[])]]
    wv = Waves(vocab, a.D, a.seed)
    dev = torch.device(a.device) if (a.device and torch is not None) else None
    wv.learn(seq, dev=dev)
    wv.W, wv.base = wv.W.astype(np.complex64), None
    tr_k = tr[:: max(1, math.ceil(len(tr) / a.max_scenes))]
    tr_has = [x for x in tr if x["post"] not in (NO_UKE, OTHER)]      # 2 段目（どの受けか）: 名前のある受けの場面だけ
    tr_has = tr_has[:: max(1, math.ceil(len(tr_has) / a.max_scenes))]
    rows = {}

    def row(label, SC, SCw=None):
        rows[label] = r = measure(SC, te, cands, SCw)
        log.info("%-24s 付けるか AUROC %.3f　どの受けか 1 番目 %.3f（3 番まで %.3f、%d 行）", label, r["有無 AUROC"],
                 r["どの受けか 1 番目"], r["どの受けか 3 番目まで"], r["受けの行の数"])

    pr = np.array([cnt.get(c, 0) if c != OTHER else sum(v for k, v in cnt.items() if k not in cands) for c in cands], float)
    row("いちばん多い受け", np.tile(pr, (len(te), 1)))
    log.info("2 段目（どの受けか）の学び %d 場面", len(tr_has))
    for label, keys in TABLES:
        ct = UkeTable(cands, keys).fit(tr)
        cw = UkeTable(cands, keys).fit(tr_has)
        row(label, np.stack([ct.scores(s) for s in te]), np.stack([cw.scores(s) for s in te]))
    B = 256
    for label, keys in VARIANTS:
        rel = UkeRelation(wv, cands, a.D, keys, device=dev).fit(tr_k)
        rw = UkeRelation(wv, cands, a.D, keys, device=dev).fit(tr_has)
        row("波: " + label, np.concatenate([rel.scoop_scores(te[i:i + B]) for i in range(0, len(te), B)]),
            np.concatenate([rw.scoop_scores(te[i:i + B]) for i in range(0, len(te), B)]))
        del rel, rw
        if dev is not None:
            torch.cuda.empty_cache()
    res = dict(cache=os.path.basename(a.cache), seed=a.seed, D=a.D, n_train_works=len(tr_w), n_test_works=len(te_w), n_train=len(tr),
               n_scenes=len(tr_k), n_test=len(te), n_cands=len(cands), top_cands=[(c, cnt.get(c, 0)) for c in cands[:40]],
               test_share={c: round(share[c] / len(te), 4) for c in (NO_UKE, OTHER)}, rows=rows, seconds=round(time.time() - t0))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.info("済み %.0f 秒", time.time() - t0)


# ---------------------------------------------------------------- bot につなぐ板（I48l の 2 段）
class _Wv:
    """読み込んだ語の波（Relation が使う ix・W だけ）。"""

    def __init__(self, vocab, W):
        self.vocab, self.W = list(vocab), W
        self.ix = {w: i for i, w in enumerate(self.vocab)}


def query(prev_toks, own_toks=None):
    """相手の前の行（と自分の前の行）から、1 つの場面（鍵を作る形）。"""
    work = dict(lines=([dict(toks=own_toks)] if own_toks is not None else []) + [dict(toks=prev_toks), dict(toks=[])])
    return samples(work)[-1]


class UkeBoard:
    """受けの板。1 段目（付けるか: 受け無しの見込み）と 2 段目（どの受けか: 受けのある場面だけ）を持つ。
    build で資料から作り、save・load で書き出す・読み込む（I48f の関係の板と同じ作り）。"""

    KEYS = ("尾", "中", "印")

    def __init__(self, wv, cands, D, K1, P1, K2, P2):
        self.cands, self.D = list(cands), D
        self.cix = {c: i for i, c in enumerate(self.cands)}
        self.r1 = UkeRelation(wv, self.cands, D, self.KEYS)
        self.r2 = UkeRelation(wv, self.cands, D, self.KEYS)
        for r, K, P in ((self.r1, K1, P1), (self.r2, K2, P2)):
            r.K, r.P = K, P
            r.prior = np.bincount(P, minlength=len(self.cands)).astype(float)

    @classmethod
    def build(cls, works, D=512, seed=0, min_count=20, max_scenes=200000, dev=None):
        tr = [s for w in works for s in samples(w)]
        cnt = Counter(s["post"] for s in tr)
        cands = [c for c, n in cnt.most_common() if n >= min_count and c != NO_UKE] + [OTHER, NO_UKE]
        tr = to_cands(tr, cands)
        vocab = sorted({t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] in CUE})
        seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in CUE] for w in works for ln in w["lines"] + [dict(toks=[])]]
        wv = Waves(vocab, D, seed)
        wv.learn(seq, dev=dev)
        wv = _Wv(vocab, wv.W.astype(np.complex64))
        has = [x for x in tr if x["post"] not in (NO_UKE, OTHER)]
        r1 = UkeRelation(wv, cands, D, cls.KEYS).fit(tr[:: max(1, math.ceil(len(tr) / max_scenes))])
        r2 = UkeRelation(wv, cands, D, cls.KEYS).fit(has[:: max(1, math.ceil(len(has) / max_scenes))])
        return cls(wv, cands, D, r1.K, r1.P, r2.K, r2.P)

    def save(self, path):
        wv = self.r1.wv
        np.savez(path + ".npz", W=wv.W, K1=self.r1.K, P1=self.r1.P, K2=self.r2.K, P2=self.r2.P)
        json.dump(dict(D=self.D, cands=self.cands, vocab=wv.vocab), open(path + ".json", "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        m = json.load(open(path + ".json", encoding="utf-8"))
        z = np.load(path + ".npz")
        return cls(_Wv(m["vocab"], z["W"]), m["cands"], m["D"], z["K1"], z["P1"], z["K2"], z["P2"])

    def open_prob(self, s):
        """1 段目: 受けを付ける見込み（1 − 受け無しの明るさの割合）。"""
        p = self.r1.scoop_scores([s])[0]
        return float(1 - p[self.cix[NO_UKE]] / p.sum())

    def brightness(self, s):
        """2 段目: 名前のある受けごとの明るさの割合（受け無し・ほかの受け は 0）。"""
        q = self.r2.scoop_scores([s])[0].copy()
        q[self.cix[NO_UKE]] = q[self.cix[OTHER]] = 0
        return q / q.sum()

    def choose(self, s, used, rng):
        """明るさの比で選ぶ（1 番目に固定しない）。この会話で使った受けは、薄れた記憶として明るさを下げる:
        使った割合が、その場の見込み（明るさの割合）を越えている分だけ、見込み ÷ 使った割合 を掛ける（消さない。手で置く値は無い）。"""
        q = self.brightness(s)
        n = sum(used.values())
        if n:
            for u, k in used.items():
                j = self.cix.get(u)
                if j is not None and q[j] > 0:
                    q[j] *= min(1.0, q[j] / (k / n))
        q = q / q.sum()
        j = int(rng.choices(range(len(q)), weights=q)[0])
        return self.cands[j], float(q[j])


if __name__ == "__main__":
    main()
