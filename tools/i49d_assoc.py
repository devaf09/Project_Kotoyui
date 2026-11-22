"""I49d の下ごしらえ: 連想の口と、語ごとの印（持ち主の案「連想は指揮者の前で、属性の向きが近い物同士をもつ」「入れ物に過去形・現在進行形等の印をいれる」、未確定）。

ループ本体（intervention/i49d_*、クラウド側で作る）は、ここを import して使う。

連想の口:
  Assoc(names, M, pos=None)   M の行が語ごとの向き。行は長さ 1 にそろえる。
  near(vec, k)                → [(語の名前, ベクトル（長さ 1）, 近さ |cos|)] を近い順に k 個。自分自身は除かない（除くのは呼ぶ側）。
  Assoc.from_wordvecs(wv)     波の板の語の波 W（i49c_read_words.WordVecs と同じ。複素数）で引く。
  Assoc.from_attrs(S, cb)     語板の属性（I48 の AttrSpace の帯の上位、float32）の向きで引く。語板の内容語だけ。
印（語ごと。GiNZA の形態の情報 Inflection と、語の並びから）:
  過去   Inflection が 助動詞-タ（た・だ）
  丁寧   Inflection が 助動詞-マス・助動詞-デス
  否定   Inflection が 助動詞-ナイ・助動詞-ヌ
  進行   接続助詞「て・で」のすぐ後の「いる」（非自立の動詞。います・いない・いた を含む）
  印は助動詞など、ベクトルを持たない語に付く。読み手に渡す時の移し方は 2 通り（tools/i49c_read_words.py の mark_mode）:
    prev  その文の中で、直前のベクトルを持つ語に移す（無ければ取りこぼし）
    sent  文の印としてまとめ、その文の初めのベクトルを持つ語に付ける（文にベクトルを持つ語が無ければ、次の文へ持ち越す）
手で置く物: 印の決まり（上の 4 つ。Inflection の名前と「て・で ＋ いる」）、数えの語の数 1,000、k = 5。
python tools/i49d_assoc.py --src <元の本文の置き場> --conllu <青空文庫の conllu> --content <content_boards.json> --func <func_boards.json> \
    --wave <波の板の名前> --out <stats.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

MARKS = ("過去", "丁寧", "否定", "進行")
JA2UPOS = {"名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ"}


class Assoc:
    """連想の口: 向きの近い語を引く。"""

    def __init__(self, names, M, pos=None):
        M = np.asarray(M)
        n = np.linalg.norm(M, axis=1, keepdims=True)
        self.M = (M / np.where(n > 0, n, 1.0)).astype(M.dtype)
        self.names = list(names)
        self.pos = list(pos) if pos is not None else None
        self.ix = {w: i for i, w in enumerate(self.names)}

    def near(self, vec, k=5):
        v = np.asarray(vec).reshape(-1)
        nv = float(np.linalg.norm(v))
        if nv == 0:
            return []
        c = np.abs(np.conj(self.M) @ (v / nv)) if np.iscomplexobj(self.M) else np.abs(self.M @ (v / nv).real)
        k = min(int(k), len(c))
        idx = np.argpartition(-c, k - 1)[:k]
        idx = idx[np.argsort(-c[idx])]
        return [(self.names[i], self.M[i], float(c[i])) for i in idx]

    def vec(self, name):
        i = self.ix.get(name)
        return None if i is None else self.M[i]

    @classmethod
    def from_wordvecs(cls, wv):
        names = [None] * (max(wv.ix.values()) + 1)
        for w, i in wv.ix.items():
            names[i] = w
        keep = [i for i, w in enumerate(names) if w is not None]
        return cls([names[i] for i in keep], wv.W[keep])

    @classmethod
    def from_attrs(cls, S, cb):
        """語板の内容語ごとに、帯の上位の属性を足した向き（float32、AttrSpace の D）。"""
        names, rows, pos = [], [], []
        seen = set()
        for w in cb:
            up = JA2UPOS.get(w.get("pos"))
            if up is None or w["name"] in seen:
                continue
            f, v = S.attrs(w["name"], up)
            y = np.zeros(S.D, dtype=np.float32)
            np.add.at(y, f, v)
            seen.add(w["name"])
            names.append(w["name"])
            rows.append(y)
            pos.append(up)
        return cls(names, np.stack(rows), pos)


def token_marks(toks):
    """toks: GiNZA の語の並び（lemma_・pos_・morph を持つ物）→ 語ごとの印の集合の並び。"""
    out = [set() for _ in toks]
    for i, t in enumerate(toks):
        inf = ",".join(t.morph.get("Inflection") or []) if hasattr(t, "morph") else ""
        if "助動詞-タ" in inf:
            out[i].add("過去")
        if "助動詞-マス" in inf or "助動詞-デス" in inf:
            out[i].add("丁寧")
        if "助動詞-ナイ" in inf or "助動詞-ヌ" in inf:
            out[i].add("否定")
        if t.lemma_ == "いる" and i > 0 and toks[i - 1].lemma_ in ("て", "で") and toks[i - 1].pos_ == "SCONJ":
            out[i].add("進行")
    return out


def main():
    import i48ab_long_grain as G
    import i49b_read_long as RL
    import i49c_read_words as R
    ap = argparse.ArgumentParser()
    for k in ("src", "conllu", "content", "func", "wave", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    RL.MAX_SENT = 3000
    w = RL.pick_works(a.src, a.conllu, n=1)[0]
    import spacy
    nlp = spacy.load("ja_ginza")
    docs = list(nlp.pipe([s for s, _ in w], batch_size=256))
    wv = R.WordVecs.load(a.wave)
    res = dict(数えの語=a.n, k=a.k)
    # ---- 1. 連想の口 ----
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, 2048)
    assocs = dict(波=Assoc.from_wordvecs(wv), 属性=Assoc.from_attrs(S, cb))
    pos_of = {}
    for x in cb:
        pos_of.setdefault(x["name"], JA2UPOS.get(x.get("pos")))
    read = []                                             # 作品 1 を読んだ順の、ベクトルを持つ語（波の板にあり、語板にもある語）
    for d in docs:
        for t in d:
            if wv.of(t.lemma_, t.pos_) is not None and t.lemma_ in assocs["属性"].ix:
                read.append((t.lemma_, t.pos_))
            if len(read) >= a.n:
                break
        if len(read) >= a.n:
            break
    for nm, A in assocs.items():
        sims, same, secs, self_top = [], [], [], 0
        for lem, p in read:
            v = wv.of(lem, p) if nm == "波" else A.vec(lem)
            t1 = time.perf_counter()
            r = A.near(v, a.k + 1)
            secs.append(time.perf_counter() - t1)
            self_top += r[0][0] == lem
            r = [x for x in r if x[0] != lem][: a.k]
            sims.append(float(np.median([c for _, _, c in r])))
            same.append(pos_of.get(r[0][0]) == p)
        res[f"連想_{nm}"] = dict(語の数=len(A.names), 引いた語=len(read), 上位5の近さの中央値_自分を除く=round(float(np.median(sims)), 4),
                               上位1が同じ品詞の割合_自分を除く=round(float(np.mean(same)), 4), 上位1が自分自身の割合=round(self_top / max(1, len(read)), 4),
                               一回のミリ秒_中央値=round(float(np.median(secs)) * 1000, 2))
    # ---- 2. 印 ----
    cnt = {m: 0 for m in MARKS}
    prev_ok = {m: 0 for m in MARKS}
    sent_ok = {m: 0 for m in MARKS}
    for d in docs:
        toks = [t for t in d if t.pos_ != "PUNCT"]
        mk = token_marks(toks)
        has_vec = [wv.of(t.lemma_, t.pos_) is not None for t in toks]
        any_vec = any(has_vec)
        for i, ms in enumerate(mk):
            for m in ms:
                cnt[m] += 1
                prev_ok[m] += any(has_vec[:i])
                sent_ok[m] += any_vec
    res["印"] = {m: dict(数=cnt[m], 直前の語へ移せた割合=round(prev_ok[m] / max(1, cnt[m]), 4), 文の印にできた割合=round(sent_ok[m] / max(1, cnt[m]), 4))
                for m in MARKS}
    res["印_文の数"] = len(docs)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
