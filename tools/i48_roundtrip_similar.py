"""波にしてから戻す試し（本物の語板）の続き: 似て非なる語と、全体の波との合わなさ（持ち主の案、未確定）。

置き方は拡散、候補は語板の語 3,000（tools/i48_roundtrip_board.py と同じ選び方）、文は自作の 10 文 ＋ 試しの文 1,000 文の内容語の並び。
1. 似て非なる語: 語ごとに、候補のうちいちばん属性の近いほかの語との属性のコサイン（近い語の近さ）を出し、その帯
   （〜.5・〜.8・〜.95・〜1 未満・1＝属性が全く同じ）ごとに、語を戻せた割合を見る。
   あわせて、語の波に「語の名前だけの符号」（語ごとの乱数の位相。属性と関係なし）を重み ε で混ぜる:
   語の波 = (1 − ε) × 属性の波 ＋ ε × 名前の符号（どちらも長さ 1）。ε = 0・0.1・0.2・0.3・0.5。
   名前の符号が増えると似て非なる語は分かれるが、属性で読む力（語を当てずに属性を読む）は下がるはず。その釣り合いを見る。
2. 全体の波との合わなさ: 読んだ語の並びから波を作り直し、元の文の波との差を見る。
   文の合わなさ = |元の波 − 作り直した波|² / |元の波|²。文の中に外れがある時に大きくなるか（外れのある文を見分ける AUC）。
   位置の合わなさ: その位置で選んだ語の近さ（低いほど怪しい）と、1 位と 2 位の近さの差（小さいほど怪しい）で、外れた位置を見分ける AUC。
   本当の並びで作り直した時（合わなさの床。0 になるはず）も並べる。
手で置く物: ε の並び、近い語の近さの帯の境、D（512・2,048）。
python tools/i48_roundtrip_similar.py --conllu ... --content ... --func ... --out out.json [--test 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48ab_long_grain as G  # noqa: E402
import i48ab_place as PL  # noqa: E402
import i48_roundtrip_board as RB  # noqa: E402

EPS = (0.0, 0.1, 0.2, 0.3, 0.5)
DS = (512, 2048)
EDGES = (0.5, 0.8, 0.95, 0.9999)
BIN_NAMES = ("〜.5", "〜.8", "〜.95", "〜1 未満", "1（全く同じ）")


def bin_of(c):
    for e, nm in zip(EDGES, BIN_NAMES):
        if c <= e:
            return nm
    return BIN_NAMES[-1]


def auc(pos, neg):
    """pos（外れ）の値が neg（当たり）より大きい確からしさ。"""
    if not pos or not neg:
        return None
    a = np.array(neg + pos)
    r = a.argsort().argsort() + 1.0
    rp = r[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def name_code(lemma, D):
    r = np.random.default_rng(zlib.crc32(("名:" + lemma).encode()))
    c = np.exp(2j * np.pi * r.random(D)).astype(np.complex64)
    return c / np.linalg.norm(c)


class LexE(RB.Lex):
    def __init__(self, waves, keys):
        self.keys = keys
        self.idx = {k: i for i, k in enumerate(keys)}
        W = np.stack(waves)
        self.norm = np.maximum(1e-9, np.linalg.norm(W, axis=1)).astype(np.float32)
        self.Ln = (W / self.norm[:, None]).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--test", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    own = RB.tag(RB.SENTS)
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    del sents
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    board_toks = [dict(lemma=w["name"], upos=RB.UPOS.get(w.get("pos"), "NOUN"), xpos="", form=w["name"])
                  for w in sorted(cb, key=lambda w: -w.get("freq", 0))]
    seqs = [[t for t in s if G.is_content(t)] for s in own] + [[t for t in s["toks"] if G.is_content(t)] for s in te]
    seqs = [s for s in seqs if s]
    cand = list({RB.key_of(t): t for s in seqs for t in s}.values())
    seen = {RB.key_of(t) for t in cand}
    for t in board_toks:
        if len(cand) >= RB.N_CAND:
            break
        if RB.key_of(t) not in seen:
            cand.append(t)
            seen.add(RB.key_of(t))
    keys = [RB.key_of(t) for t in cand]
    res = dict(文=len(seqs), 語=sum(len(s) for s in seqs), 候補=len(cand), ε=list(EPS), 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for D in DS:
        S = G.AttrSpace(cb, func, D, bands=("B", "C", "D", "E", "G"))
        pl = PL.Placer(RB.Sp(S), "拡散")
        # 属性の近さ（語板の属性の名前 → 重み、長さ 1）
        names = sorted({nm for t in cand for nm, _ in RB.attr_names(S, t)})
        ni = {nm: i for i, nm in enumerate(names)}
        A = np.zeros((len(cand), len(names)), dtype=np.float32)
        for i, t in enumerate(cand):
            for (nm, _), v in zip(RB.attr_names(S, t), S.attrs(t["lemma"], t["upos"])[1]):
                A[i, ni[nm]] = v
        A /= np.maximum(1e-9, np.linalg.norm(A, axis=1, keepdims=True))
        Cm = A @ A.T
        np.fill_diagonal(Cm, -1)
        near = Cm.max(1)
        del Cm
        kidx = {k: i for i, k in enumerate(keys)}
        bins_word = {RB.key_of(t): bin_of(near[kidx[RB.key_of(t)]]) for s_ in seqs for t in s_}
        res.setdefault("近い語の近さの帯ごとの語の数", {})[f"D {D}"] = {b: sum(1 for s in seqs for t in s if bins_word[RB.key_of(t)] == b) for b in BIN_NAMES}
        Aw = np.stack([pl.word(np.array([S.f_of(nm)]), np.array([1.0], dtype=np.float32)) for nm in names])
        Aw /= np.maximum(1e-9, np.linalg.norm(Aw, axis=1, keepdims=True))
        attr_w = {}
        for t in cand:
            w = pl.word(*S.tok(t))
            attr_w[RB.key_of(t)] = w / max(1e-9, float(np.linalg.norm(w)))
        for eps in EPS:
            t1 = time.time()
            wv = {k: (1 - eps) * attr_w[k] + eps * name_code(k[0] + "|" + k[1], D) for k in keys}
            lex = LexE([wv[k] for k in keys], keys)
            ok_bin, n_bin = {b: 0 for b in BIN_NAMES}, {b: 0 for b in BIN_NAMES}
            hit = n = 0
            sent_res, sent_res_true, sent_wrong = [], [], []
            pos_score = {"当たり": [], "外れ": []}
            pos_margin = {"当たり": [], "外れ": []}
            got = tot = 0
            for s in seqs:
                ids = [lex.idx[RB.key_of(t)] for t in s]
                X = pl.encode([wv[keys[i]] for i in ids])
                g, sc = RB.decode(pl, X, len(s), [lex] * len(s))
                wrong = False
                for k, ti in enumerate(ids):
                    ok = g[k] == ti
                    hit += ok
                    n += 1
                    b = bins_word[keys[ti]]
                    n_bin[b] += 1
                    ok_bin[b] += ok
                    wrong |= not ok
                    srt = np.partition(sc[k], -2)[-2:]
                    lab = "当たり" if ok else "外れ"
                    pos_score[lab].append(-float(sc[k][g[k]]))          # 低いほど怪しい → 符号を返して「大きいほど怪しい」
                    pos_margin[lab].append(-float(srt[1] - srt[0]))
                Xg = pl.encode([lex.L(np.array([i]))[0] for i in g])
                nx = float(np.vdot(X, X).real)
                sent_res.append(float(np.vdot(X - Xg, X - Xg).real) / nx)
                Xt = pl.encode([lex.L(np.array([i]))[0] for i in ids])
                sent_res_true.append(float(np.vdot(X - Xt, X - Xt).real) / nx)
                sent_wrong.append(wrong)
                # 属性で読む（語を当てずに）
                cen = RB.centers(pl, len(s))
                Y = np.stack([pl.take(X, c) for c in cen])
                sa = np.real(Y @ np.conj(Aw).T)
                for k, t in enumerate(s):
                    tn = [ni[nm] for nm, _ in RB.attr_names(S, t) if nm in ni]
                    top = set(np.argpartition(-sa[k], len(tn))[:len(tn)].tolist())
                    got += sum(i in top for i in tn)
                    tot += len(tn)
            rw = [r_ for r_, w_ in zip(sent_res, sent_wrong) if w_]
            rr = [r_ for r_, w_ in zip(sent_res, sent_wrong) if not w_]
            res["型"][f"D {D}・ε {eps}"] = dict(
                語_上位1=round(hit / n, 4),
                近い語の近さの帯ごとの語の戻り={b: (round(ok_bin[b] / n_bin[b], 4) if n_bin[b] else None) for b in BIN_NAMES},
                属性で読む_全部=round(got / max(1, tot), 4),
                文の合わなさ=dict(外れのある文の数=len(rw), 外れのある文_平均=round(float(np.mean(rw)), 4) if rw else None,
                             外れの無い文_平均=round(float(np.mean(rr)), 4) if rr else None,
                             本当の並びで作り直した時_平均=round(float(np.mean(sent_res_true)), 4),
                             見分けのAUC=round(auc(rw, rr), 4) if rw and rr else None),
                位置の怪しさ=dict(選んだ語の近さ_AUC=round(auc(pos_score["外れ"], pos_score["当たり"]), 4) if pos_score["外れ"] else None,
                             一位と二位の差_AUC=round(auc(pos_margin["外れ"], pos_margin["当たり"]), 4) if pos_margin["外れ"] else None),
                秒=round(time.time() - t1))
            print(f"D {D}・ε {eps}", json.dumps(res["型"][f"D {D}・ε {eps}"], ensure_ascii=False), flush=True)
            save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
