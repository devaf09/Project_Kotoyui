"""I49c の読み手: 文章を 1 語ずつ渡す（持ち主の案「文章の渡しを単語ごとに」「ループが次の単語を自発的に読み込む」、未確定）。

ループ本体（intervention/i49c_think_loop.py の Loop、クラウド側で作る）は、この読み手を import して使う。口は 2 つ:
  has_next()  まだ読む語があるか
  next()      次の 1 語を (語の名前, ベクトル または None, 印の dict) で返す。読み手は呼ばれた時だけ進む（自分では進まない）。
ベクトル: 波の板（I48 の WaveBoard を保存した物）の語の波 W の 1 行を、長さ 1 にした物（complex64）。前処理（平均を引く・idf を掛ける など）はしない。
  ベクトルを持つのは、内容語（I48 の CONTENT: NOUN・PROPN・VERB・ADJ）で、STOP に無く、板の語にある語だけ。
  それ以外の語（機能語・記号・板に無い語）の扱いは 2 通り:
    skip  渡さない（読み手から消える）
    none  ベクトル None で渡す（語があった事と印だけ渡す）
印: 文頭（その文の初めの語）、文の終わり（その文の最後の語）、見出し・空き行・改ページ（その文の直前にあった時、文の初めの語に付ける）。
  skip の時、文の初めや終わりの語が飛ばされたら、印は残る語のうち初め・終わりの語に移す。
作品の読み直しは tools/i49b_read_long.py と同じ（元の本文から区切りの印つきで読み直す）。
python tools/i49c_read_words.py --src <元の本文の置き場> --conllu <青空文庫の conllu> --wave <波の板（.npz と .json の名前）> --out <stats.json>
  [--works 1] [--max_sent 3000]
数えるのは: 語の数、ベクトルを持つ語の割合、1 文あたりの語の数（全部・ベクトルを持つ語）、ベクトルを持つ語が 0 の文の数（2 通りの扱いごと）。
手で置く物: 作品の選び方（I49b と同じ、種 0）、1 作品の文の上限 3,000、内容語の品詞（I48 の CONTENT）と STOP。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48_wave_board as W48  # noqa: E402
import i49b_read_long as RL  # noqa: E402

MODES = ("skip", "none")


class WordVecs:
    """波の板の語の波 W（行を長さ 1 にした物）と、語 → 行の番号。"""

    def __init__(self, W, vocab):
        n = np.linalg.norm(W, axis=1, keepdims=True)
        self.W = (W / np.where(n > 0, n, 1.0)).astype(np.complex64)
        self.ix = {w: i for i, w in enumerate(vocab) if w is not None}

    @classmethod
    def load(cls, path):
        z = np.load(path + ".npz")
        m = json.load(open(path + ".json", encoding="utf-8"))
        return cls(z["W"], m["vocab"])

    def of(self, lemma, pos):
        if pos not in W48.CONTENT or lemma in W48.STOP:
            return None
        i = self.ix.get(lemma)
        return None if i is None else self.W[i]


class WordReader:
    """1 語ずつ渡す読み手。items は (語の名前, ベクトル か None, 印の dict) の並び。"""

    def __init__(self, items):
        self.items = list(items)
        self.i = 0

    def has_next(self):
        return self.i < len(self.items)

    def next(self):
        if not self.has_next():
            raise StopIteration("読む語が無い")
        x = self.items[self.i]
        self.i += 1
        return x

    def __len__(self):
        return len(self.items)

    @classmethod
    def from_tokens(cls, sents, vec_of, mode="skip"):
        """sents: [(語の並び [(原形, 品詞)], 区切りの印の dict)]。vec_of(原形, 品詞) → ベクトル か None。mode: skip か none。"""
        assert mode in MODES
        out, carry = [], {}
        for toks, marks in sents:
            row = [(w, vec_of(w, p)) for w, p in toks]
            if mode == "skip":
                row = [(w, v) for w, v in row if v is not None]
            here = dict(carry)
            here.update({kk: True for kk, vv in marks.items() if vv})
            if not row:                                   # 語が全部飛ばされた文の区切りの印は、次の文の初めの語へ持ち越す
                carry = here
                continue
            carry = {}
            for k, (w, v) in enumerate(row):
                m = {}
                if k == 0:
                    m.update(here)
                    m["文頭"] = True
                if k == len(row) - 1:
                    m["文の終わり"] = True
                out.append((w, v, m))
        return cls(out)


def counts(sents, vec_of):
    n_tok = n_vec = 0
    per_all, per_vec, zero = [], [], 0
    for toks, _ in sents:
        k = sum(vec_of(w, p) is not None for w, p in toks)
        n_tok += len(toks)
        n_vec += k
        per_all.append(len(toks))
        per_vec.append(k)
        zero += k == 0
    return dict(文=len(sents), 語=n_tok, ベクトルを持つ語=n_vec, ベクトルを持つ語の割合=round(n_vec / max(1, n_tok), 4),
                一文あたりの語_平均=round(float(np.mean(per_all)), 2), 一文あたりの語_中央値=float(np.median(per_all)),
                一文あたりのベクトルを持つ語_平均=round(float(np.mean(per_vec)), 2), 一文あたりのベクトルを持つ語_中央値=float(np.median(per_vec)),
                ベクトルを持つ語が0の文=zero)


AFTER = 500
NEAR = 3


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    x = np.array(list(neg) + list(pos), dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=x.argsort().argsort() + 1.0) / cnt)[inv]
    return round(float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


class Trace:
    """試しの観察: 歩ごとの項目・読んだ語・印。出力ごとに、いちばん近い読んだ語（何語前か・近さ・品詞）。"""

    def __init__(self, pos_of, n_max, dim):
        self.pos_of = pos_of
        self.items, self.marks, self.word_step = [], [], []
        self.V = np.zeros((n_max, dim), dtype=np.complex64)   # 読んだ語の向き（毎回作り直さないよう、先に場所を取る）
        self.nv = 0
        self.names, self.vec_step = [], []
        self.said = []                                    # (歩, 何語前か（ベクトルを持つ語で数える）, 近さ, 品詞)

    def after(self, loop, rec):
        self.items.append(rec["項目"])
        self.marks.append(rec["印"] if rec["項目"] == "次へ" else None)
        if rec["項目"] == "次へ" and rec["ベクトル"] is not None:
            v = np.asarray(rec["ベクトル"], dtype=np.complex64)
            self.V[self.nv] = v / np.linalg.norm(v)
            self.nv += 1
            self.names.append(rec["語"])
            self.vec_step.append(rec["歩"])
        if rec["項目"] == "出力" and self.nv:
            c = np.abs(np.conj(self.V[: self.nv]) @ rec["出した向き"])
            k = int(c.argmax())
            self.said.append((rec["歩"], self.nv - 1 - k, float(c[k]), self.pos_of.get(self.names[k], "?")))


def trial(sents, wv, mode):
    """読み手を Loop に渡し、読み終わるまで回し、その後 AFTER 歩回す。数えだけ返す。"""
    import i49c_think_loop as TL
    from i49a_inner_vessel import Vessel
    pos_of = {}
    for toks, _ in sents:
        for w, p in toks:
            pos_of.setdefault(w, p)
    rd = WordReader.from_tokens(sents, wv.of, mode)
    loop = TL.Loop(Vessel(2048, bands=20, longest=1024), rd)
    tr = loop.attach(Trace(pos_of, sum(v is not None for _, v, _ in rd.items), 2048))
    t0 = time.time()
    cap = 20 * len(rd) + 1000                             # 止め役: 読み終わらなくても必ず止まる
    while rd.has_next() and loop.n < cap:
        loop.step()
    read_steps = loop.n
    loop.run(AFTER)
    sec = time.time() - t0
    a = loop.arrays()
    items = tr.items
    # 語と語の間に入った「待つ」（ほかの項目は数えない）と、待った直前の語（ベクトルを持つ語だけ）の外れ・動きの量
    nxt = [i for i, it in enumerate(items[:read_steps]) if it == "次へ"]
    gaps = [sum(items[j] == "待つ" for j in range(p + 1, q)) for p, q in zip(nxt, nxt[1:])]
    sur, act = a["sur"].mean(1), (a["act"] / float(loop.v.cap)).mean(1)
    heard = set(tr.vec_step)
    w_pos = [p for p, g in zip(nxt, gaps) if p in heard and g > 0]
    w_neg = [p for p, g in zip(nxt, gaps) if p in heard and g == 0]
    hist = {"0": 0, "1": 0, "2": 0, "3-5": 0, "6-10": 0, "11-": 0}
    for g in gaps:
        hist["0" if g == 0 else "1" if g == 1 else "2" if g == 2 else "3-5" if g <= 5 else "6-10" if g <= 10 else "11-"] += 1
    # 出力: 区切りの直後（その区切りの語から NEAR 語以内）か。でたらめに置いた時の割合は、読み終わるまでの全部の歩の中でその窓に入る割合
    def windows(kind):
        inwin, last, cnt = np.zeros(len(items), dtype=bool), None, 10 ** 9
        for i, (it, m) in enumerate(zip(items, tr.marks)):
            if it == "次へ":
                cnt += 1
                if m and m.get(kind):
                    cnt = 0
            inwin[i] = cnt <= NEAR
        return inwin
    outs = [s for s, _, _, _ in tr.said]
    out_read = [s for s in outs if s < read_steps]
    near = {}
    for kind in ("文の終わり", "見出し", "空き行"):
        w = windows(kind)
        near[kind] = dict(出力の割合=round(float(np.mean([w[s] for s in out_read])), 4) if out_read else None,
                          でたらめの割合=round(float(w[:read_steps].mean()), 4))
    back = np.array([b for _, b, _, _ in tr.said]) if tr.said else np.array([])
    sim = np.array([c for _, _, c, _ in tr.said]) if tr.said else np.array([])
    pos_c = {}
    for _, _, _, p in tr.said:
        pos_c[p] = pos_c.get(p, 0) + 1
    g = a["gain"]
    return dict(
        扱い=mode, 渡す語=len(rd), 歩=int(loop.n), 読み終わるまでの歩=int(read_steps), 秒=round(sec, 1), 内訳=loop.counts(),
        内訳_読み終わるまで={k: sum(it == k for it in items[:read_steps]) for k in TL.ITEMS},
        待つ_語と語の間の分布=hist, 待つ_語と語の間の平均=round(float(np.mean(gaps)), 3) if gaps else None,
        待った直前の語=dict(数=len(w_pos), 待たなかった語=len(w_neg), 外れ_AUC=auc(sur[w_pos], sur[w_neg]), 動きの量_AUC=auc(act[w_pos], act[w_neg]),
                     外れ_平均=[round(float(sur[w_pos].mean()), 4) if w_pos else None, round(float(sur[w_neg].mean()), 4) if w_neg else None],
                     動きの量_平均=[round(float(act[w_pos].mean()), 4) if w_pos else None, round(float(act[w_neg].mean()), 4) if w_neg else None]),
        出力=dict(回数=len(outs), 読み終わるまで=len(out_read), 読み終わった後=len(outs) - len(out_read), 区切りの直後=near,
                直前に読んだ語がいちばん近い割合=round(float((back == 0).mean()), 4) if len(back) else None,
                五語以内がいちばん近い割合=round(float((back < 5).mean()), 4) if len(back) else None,
                近さの中央値=round(float(np.median(sim)), 4) if len(sim) else None,
                いちばん近い語の品詞の割合={p: round(n / len(tr.said), 4) for p, n in sorted(pos_c.items(), key=lambda z: -z[1])} if tr.said else {}),
        見張り=dict(張り付き=loop.v.watch["張り付き"].tolist(), 死んだ刻み=int(loop.v.watch["死んだ"]),
                  感度の中央値=np.round(np.median(g, 0), 3).tolist(), 感度の上限の割合=np.round((g >= loop.v.gmax * 0.999).mean(0), 4).tolist()))


def main():
    ap = argparse.ArgumentParser()
    for k in ("src", "conllu", "wave", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--works", type=int, default=1)
    ap.add_argument("--max_sent", type=int, default=3000)
    ap.add_argument("--trial", action="store_true", help="思考ループ（i49c_think_loop.Loop）に 1 語ずつ読ませる試し（skip と none）")
    a = ap.parse_args()
    if a.trial:
        t0 = time.time()
        RL.MAX_SENT = a.max_sent
        w = RL.pick_works(a.src, a.conllu, n=1)[0]
        import spacy
        nlp = spacy.load("ja_ginza")
        wv = WordVecs.load(a.wave)
        docs = list(nlp.pipe([s for s, _ in w], batch_size=256))
        sents = [([(t.lemma_, t.pos_) for t in d if t.pos_ != "PUNCT"], m) for d, (_, m) in zip(docs, w)]
        res = dict(読み終わった後の歩=AFTER, 区切りの窓=NEAR, 試し=[])
        for mode in MODES:
            r = trial(sents, wv, mode)
            res["試し"].append(r)
            print(mode, json.dumps(r, ensure_ascii=False)[:600], flush=True)
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        res["秒"] = round(time.time() - t0)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    t0 = time.time()
    RL.MAX_SENT = a.max_sent
    works = RL.pick_works(a.src, a.conllu, n=a.works)
    import spacy
    nlp = spacy.load("ja_ginza")
    wv = WordVecs.load(a.wave)
    res = dict(作品=[], 秒=None)
    for w in works:
        docs = list(nlp.pipe([s for s, _ in w], batch_size=256))
        sents = [([(t.lemma_, t.pos_) for t in d if t.pos_ != "PUNCT"], m) for d, (_, m) in zip(docs, w)]
        r = counts(sents, wv.of)
        for mode in MODES:
            rd = WordReader.from_tokens(sents, wv.of, mode)
            marks = [m for _, _, m in rd.items]
            r[f"読み手_{mode}"] = dict(渡す語=len(rd), ベクトル無しで渡す語=sum(v is None for _, v, _ in rd.items),
                                     文の終わりの印=sum("文の終わり" in m for m in marks), 見出しの印=sum("見出し" in m for m in marks),
                                     空き行の印=sum("空き行" in m for m in marks))
        res["作品"].append(r)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
