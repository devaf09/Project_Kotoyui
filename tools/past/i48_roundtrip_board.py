"""文を波に変えて戻す試し（本物の語板。tools/past/i48_roundtrip.py の続き）。

語 = 語板の属性の符号の重みつきの和（i48core.space.AttrSpace の属性 → 周波数、i48ab_place.Placer の置き方。似た語は似た波）。
置き方: 頭から・位相の板・拡散。D = 512・2,048・8,192。文 = 自作の 10 文（GiNZA で原形・品詞を付ける）＋ 試しの文 1,000 文（本体の試しの文と同じ選び方）。
1 と 2 と 3 は内容語の並び（要点）、4 は内容語と詰め物を交互に並べた物。
  1. 語の戻り: 位置ごとに、候補の語の波との近さで読む（読めた語を引いて読み直す 2 回）。上位 1・上位 5。
     候補: 語板の語 3,000（文の語 ＋ 語板の頻度の上位で 3,000 まで）と、全部（語板の 45,577 語 ＋ 文の語。重いので D 512・2,048、
     自作の 10 文 ＋ 試しの文 200 文だけ）。取り違えた相手が似た語か（属性のコサイン）も出す。
  2. 属性で戻す: 位置ごとに、語を当てずに、属性の符号との近さで属性を読む。本当の属性 K 個のうち、読んだ上位 K に入った数（帯ごと）。
  3. 全体に光を当てる:
     (a) 全部の位置を一度に読む形（今の読み方）と、1 つずつ読む形（いちばん近さの大きい位置から 1 つ読み、その語を引いて次へ）の当たりと秒。
     (b) 位置を使わず、文の語の波をそのまま足した波から: 入っている語（上位 n・2n に入った割合、n = 文の語の数）、強い属性（上位 10 の重なり）。
     (c) 10 文（自作）・100 文・1,000 文を、文ごとの鍵（乱数の位相の板）を掛けて 1 本に重ね、鍵で戻して読む。
  4. 詰め物も入れた形: 内容語と、その後ろの詰め物（間の機能語の並び。属性は機能語の板）を交互に置いた波から、詰め物の位置を読む。
     候補: 学びの文の詰め物の上位 300 ＋ 試しの文の詰め物。
  当てずっぽう: 上位 1 は 1/候補の数、上位 5 は 5/候補の数。詰め物はいつも同じ答え（学びの文でいちばん多い詰め物）も。
図の材料: 自作の 1 文「友達 と 映画 を 見 ました」（全部の語。機能語は機能語の板）を D 256・拡散で波にした物（実部・虚部）と、
  3 番目の位置を読んだ時（読み直し無し）の候補（語 3,000 ＋ 10 文の機能語）との近さの上位 300 と、正解の値・順位。
手で置く物: 候補の数 3,000、全部で使う文の数 200、読み直しの回数 2、鍵の種、詰め物の候補 300、強い属性の上位 10。
python tools/past/i48_roundtrip_board.py --conllu ... --content ... --func ... --out out.json [--test 1000]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
import i48ab_long_grain as G  # noqa: E402
import i48ab_place as PL  # noqa: E402

SENTS = ["今日 は 天気 が いい です", "友達 と 映画 を 見 ました", "猫 が 窓 の そば で 寝て います",
         "駅 まで 歩いて 十分 かかり ます", "明日 は 早く 起き ない と いけ ない", "この 本 は とても 面白 かった",
         "雨 が 降っ たら 傘 を 持って 行く", "先生 に 質問 を して も いい です か", "昨日 の 夜 は よく 眠れ なかった",
         "コーヒー を 飲み ながら 音楽 を 聞く"]
MODES = ("頭から", "位相の板", "拡散")
DS = (512, 2048, 8192)
N_CAND = 3000
N_ALL_TEST = 200
ROUNDS = 2
N_FILL = 300
TOP_ATTR = 10
UPOS = {"名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ"}


def tag(sents):
    """自作の文の語（空白で区切った物）に、GiNZA で原形・品詞を付ける（語の頭の字に当たる GiNZA の語の物）。"""
    import spacy
    nlp = spacy.load("ja_ginza")
    out = []
    for s in sents:
        ws = s.split()
        doc = nlp("".join(ws))
        start = {t.idx: t for t in doc}
        toks, pos = [], 0
        for w in ws:
            t = start.get(pos)
            if t is None:
                t = nlp(w)[0]
            toks.append(dict(lemma=t.lemma_, upos=t.pos_, xpos=t.tag_, form=w, dep=t.dep_))
            pos += len(w)
        out.append(toks)
    return out


class Sp:
    """Placer が要る物だけ（D と SHIFT）。"""

    def __init__(self, S):
        self.D, self.shift = S.D, S.shift


def attr_names(S, t):
    """AttrSpace.attrs と同じ順の (属性の名前, 帯) の並び（内容語だけ）。"""
    out = [("品詞:" + t["upos"], "品詞")]
    w = S.cb.get(t["lemma"]) if t["upos"] in G.CONTENT else None
    if w is not None:
        for b in S.bands:
            A = w["attrs"].get(b, {})
            for a, _ in sorted(A.items(), key=lambda z: -z[1])[: S.topk]:
                out.append((b + "|" + a, b))
    return out


def key_of(t):
    return (t["lemma"], t["upos"])


class Lex:
    """候補の語の波（行を長さ 1 にした物と長さ）。"""

    def __init__(self, pl, S, toks):
        self.keys = [key_of(t) for t in toks]
        self.idx = {k: i for i, k in enumerate(self.keys)}
        W = np.stack([pl.word(*S.tok(t)) for t in toks])
        self.norm = np.maximum(1e-9, np.linalg.norm(W, axis=1)).astype(np.float32)
        self.Ln = (W / self.norm[:, None]).astype(np.complex64)
        del W

    def L(self, g):
        return self.Ln[g] * self.norm[g][:, None]

    def score(self, Y):
        return np.real(Y @ np.conj(self.Ln).T)


def centers(pl, n):
    return [pl.shifts(k, n)[0][0] for k in range(n)]


def decode(pl, X, n, lexes, rounds=ROUNDS):
    """位置ごとに候補の組（lexes[k]）で読む。読めた語の波をほかの位置から引いて読み直す。返り値: 選び・最後の近さ。"""
    cen = centers(pl, n)
    Y0 = np.stack([pl.take(X, c) for c in cen])
    sc = [lexes[k].score(Y0[k:k + 1])[0] for k in range(n)]
    g = [int(np.argmax(s)) for s in sc]
    for _ in range(rounds):
        ws = [lexes[k].L(np.array([g[k]]))[0] for k in range(n)]
        Xr = pl.encode(ws)
        Y = Y0 - np.stack([pl.take(Xr, c) for c in cen]) + np.stack(ws)
        sc = [lexes[k].score(Y[k:k + 1])[0] for k in range(n)]
        g = [int(np.argmax(s)) for s in sc]
    return g, sc


def decode_one_by_one(pl, X, n, lex):
    """1 つずつ読む: まだ読んでいない位置のうち、いちばん近さの大きい位置を 1 つ決め、その語の波を引いてから次を読む。"""
    cen = centers(pl, n)
    R = X.copy()
    g = [None] * n
    for _ in range(n):
        Y = np.stack([pl.take(R, c) for c in cen])
        sc = lex.score(Y)
        best = max((k for k in range(n) if g[k] is None), key=lambda k: sc[k].max())
        g[best] = int(np.argmax(sc[best]))
        R = R - pl.put(lex.L(np.array([g[best]]))[0], cen[best])
    return g


def acos(a, b):
    """属性の名前 → 重み の辞書どうしのコサイン。"""
    num = sum(v * b.get(k, 0.0) for k, v in a.items())
    na, nb = np.sqrt(sum(v * v for v in a.values())), np.sqrt(sum(v * v for v in b.values()))
    return float(num / (na * nb)) if na > 0 and nb > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--ds", nargs="*", type=int, default=list(DS))
    a = ap.parse_args()
    t0 = time.time()
    own = tag(SENTS)
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0][:-2000][: a.train]
    del sents
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    board_toks = [dict(lemma=w["name"], upos=UPOS.get(w.get("pos"), "NOUN"), xpos="", form=w["name"])
                  for w in sorted(cb, key=lambda w: -w.get("freq", 0))]
    # 要点（内容語の並び）
    own_c = [[t for t in s if G.is_content(t)] for s in own]
    te_c = [[t for t in s["toks"] if G.is_content(t)] for s in te]
    te_c = [s for s in te_c if s]
    seqs = own_c + te_c
    sent_toks = {key_of(t): t for s in seqs for t in s}
    cand3 = list(sent_toks.values())
    seen = set(sent_toks)
    for t in board_toks:
        if len(cand3) >= N_CAND:
            break
        if key_of(t) not in seen:
            cand3.append(t)
            seen.add(key_of(t))
    sub = own_c + te_c[:N_ALL_TEST]
    cand_all = list({key_of(t): t for t in board_toks + [t for s in sub for t in s]}.values())
    # 詰め物の並び（内容語 i の後ろの機能語の並び）
    def fill_seq(s):
        T = s["toks"] if isinstance(s, dict) else s
        ci = [i for i, t in enumerate(T) if G.is_content(t)]
        out = []
        for k, i in enumerate(ci):
            j = ci[k + 1] if k + 1 < len(ci) else len(T)
            out.append((T[i], tuple(T[i + 1:j])))
        return out
    fc = Counter()
    for s in tr:
        for _, g in fill_seq(s):
            fc["".join(t["form"] for t in g)] += 1
    fill_rep = {}
    for s in tr + te:
        for _, g in fill_seq(s):
            fill_rep.setdefault("".join(t["form"] for t in g), g)
    fseqs = [fill_seq(s) for s in own] + [fill_seq(s) for s in te]
    fseqs = [x for x in fseqs if x]
    f_names = [f for f, _ in fc.most_common(N_FILL)]
    for x in fseqs:
        for _, g in x:
            f = "".join(t["form"] for t in g)
            if f not in f_names:
                f_names.append(f)
    most_fill = fc.most_common(1)[0][0]
    res = dict(自作の文=[" ".join(t["form"] for t in s) for s in own], 自作の文の原形と品詞=[[(t["lemma"], t["upos"]) for t in s] for s in own],
               要点の文=len(seqs), 要点の語=sum(len(s) for s in seqs), 候補_3000=len(cand3), 候補_全部=len(cand_all), 全部で使う文=len(sub),
               詰め物の候補=len(f_names), 詰め物の文=len(fseqs), 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("用意", round(time.time() - t0), "秒", json.dumps({k: v for k, v in res.items() if k not in ("自作の文", "自作の文の原形と品詞", "型")}, ensure_ascii=False), flush=True)

    for D in a.ds:
        S = G.AttrSpace(cb, func, D, bands=("B", "C", "D", "E", "G"))
        ad = {}                                                # 語 → {属性の名前: 重み}（取り違えのコサイン用）

        def adict(tt):
            if key_of(tt) not in ad:
                ad[key_of(tt)] = {nm: float(v) for (nm, _), v in zip(attr_names(S, tt), S.attrs(tt["lemma"], tt["upos"])[1])}
            return ad[key_of(tt)]
        for mode in MODES:
            t1 = time.time()
            pl = PL.Placer(Sp(S), mode)
            lex = Lex(pl, S, cand3)
            enc = lambda s: pl.encode([pl.word(*S.tok(t)) for t in s])
            r = {}
            # 1. 語の戻り（候補 3,000）＋ 3(a) 一度に読む
            hit1 = hit5 = n = 0
            wrong, wrong_cos, own_hit = Counter(), [], [0, 0]
            t_all = 0.0
            for si, s in enumerate(seqs):
                X = enc(s)
                t2 = time.time()
                g, sc = decode(pl, X, len(s), [lex] * len(s))
                t_all += time.time() - t2
                for k, t in enumerate(s):
                    ti = lex.idx[key_of(t)]
                    top5 = np.argpartition(-sc[k], 5)[:5]
                    ok = g[k] == ti
                    hit1 += ok
                    hit5 += ti in top5
                    n += 1
                    if si < len(own_c):
                        own_hit[0] += ok
                        own_hit[1] += 1
                    if not ok:
                        wrong[lex.keys[g[k]]] += 1
                        if len(wrong_cos) < 3000:
                            wrong_cos.append(acos(adict(t), adict(cand3[g[k]])))
            rng = np.random.default_rng(0)
            rand_cos = []
            for _ in range(500):
                i, j = rng.integers(0, len(cand3), 2)
                rand_cos.append(acos(adict(cand3[i]), adict(cand3[j])))
            top_wrong = wrong.most_common(5)
            r["1_語の戻り_候補3000"] = dict(上位1=round(hit1 / n, 4), 上位5=round(hit5 / n, 4), 当てずっぽう_上位1=round(1 / len(cand3), 5),
                                        当てずっぽう_上位5=round(5 / len(cand3), 5), 自作の10文_上位1=round(own_hit[0] / max(1, own_hit[1]), 4),
                                        取り違えの属性のコサイン=round(float(np.mean(wrong_cos)), 4) if wrong_cos else None,
                                        取り違えで属性が同じ割合=round(float(np.mean([c > 0.9999 for c in wrong_cos])), 4) if wrong_cos else None,
                                        でたらめな2語の属性のコサイン=round(float(np.mean(rand_cos)), 4),
                                        外れの多い相手=[dict(語=k[0], 品詞=k[1], 数=c) for k, c in top_wrong],
                                        外れの上位5の割合=round(sum(c for _, c in top_wrong) / max(1, n - hit1), 4))
            # 3(a) 1 つずつ読む
            h = n2 = 0
            t2 = time.time()
            for s in seqs:
                g = decode_one_by_one(pl, enc(s), len(s), lex)
                h += sum(g[k] == lex.idx[key_of(t)] for k, t in enumerate(s))
                n2 += len(s)
            r["3a_読み方"] = dict(一度に_上位1=round(hit1 / n, 4), 一度に_秒=round(t_all, 2), 一つずつ_上位1=round(h / n2, 4), 一つずつ_秒=round(time.time() - t2, 2))
            # 2. 属性で戻す（語を当てずに）
            names = sorted({nm for t in cand3 for nm, _ in attr_names(S, t)})
            band = {nm: (nm.split("|")[0] if "|" in nm else "品詞") for nm in names}
            ni = {nm: i for i, nm in enumerate(names)}
            Aw = np.stack([pl.word(np.array([S.f_of(nm)]), np.array([1.0], dtype=np.float32)) for nm in names])
            Aw /= np.maximum(1e-9, np.linalg.norm(Aw, axis=1, keepdims=True))
            got, tot = Counter(), Counter()
            for s in seqs:
                X = enc(s)
                cen = centers(pl, len(s))
                Y = np.stack([pl.take(X, c) for c in cen])
                sc = np.real(Y @ np.conj(Aw).T)
                for k, t in enumerate(s):
                    tn = [nm for nm, _ in attr_names(S, t) if nm in ni]
                    K = len(tn)
                    top = set(np.argpartition(-sc[k], K)[:K].tolist())
                    for nm in tn:
                        tot["全部"] += 1
                        tot[band[nm]] += 1
                        if ni[nm] in top:
                            got["全部"] += 1
                            got[band[nm]] += 1
            K_mean = tot["全部"] / max(1, n)
            r["2_属性で戻す"] = dict(属性の数=len(names), 一語の属性の数の平均=round(K_mean, 2), 当てずっぽう=round(K_mean / len(names), 5),
                                 **{b: round(got[b] / max(1, tot[b]), 4) for b in ("全部", "品詞", "B", "C", "D", "E", "G")})
            # 3(b) 位置を使わず文全体を照らす（語の袋・強い属性）
            in_n = in_2n = nn = 0
            ov = []
            for s in seqs:
                Xb = np.sum([pl.word(*S.tok(t)) for t in s], axis=0)
                sc = lex.score(Xb[None, :])[0]
                truth = {lex.idx[key_of(t)] for t in s}
                m_ = len(truth)
                o = np.argsort(-sc)
                in_n += len(truth & set(o[:m_].tolist()))
                in_2n += len(truth & set(o[:2 * m_].tolist()))
                nn += m_
                tw = Counter()
                for t in s:
                    for (nm, _), v in zip(attr_names(S, t), S.attrs(t["lemma"], t["upos"])[1]):
                        if nm in ni:
                            tw[nm] += float(v)
                sa = np.real(Aw @ np.conj(Xb))
                kk = min(TOP_ATTR, len(tw))
                ov.append(len({ni[nm] for nm, _ in tw.most_common(kk)} & set(np.argsort(-sa)[:kk].tolist())) / max(1, kk))
            r["3b_文全体"] = dict(語の袋_上位n=round(in_n / nn, 4), 語の袋_上位2n=round(in_2n / nn, 4), 当てずっぽう_上位n=round(np.mean([len(s) for s in seqs]) / len(cand3), 5),
                               強い属性_上位10の重なり=round(float(np.mean(ov)), 4), 強い属性_当てずっぽう=round(TOP_ATTR / len(names), 5))
            # 3(c) 重ねて鍵で戻す
            st = {}
            for nm_, group in (("10 文（自作）", own_c), ("100 文", te_c[:100]), ("1,000 文", te_c[:1000])):
                kr = np.random.default_rng(99)
                keys = np.exp(2j * np.pi * kr.random((len(group), D))).astype(np.complex64)
                X = np.zeros(D, dtype=np.complex64)
                for kk_, s in zip(keys, group):
                    X += kk_ * enc(s)
                h = n3 = 0
                for kk_, s in zip(keys, group):
                    g, _ = decode(pl, np.conj(kk_) * X, len(s), [lex] * len(s))
                    h += sum(g[k] == lex.idx[key_of(t)] for k, t in enumerate(s))
                    n3 += len(s)
                st[nm_] = dict(文=len(group), 上位1=round(h / max(1, n3), 4))
            r["3c_重ねる"] = st
            # 4. 詰め物も入れた形
            frep = [fill_rep.get(f, ()) for f in f_names]

            def fwave(g):
                if not g:
                    return pl.word(np.array([S.f_of("詰め物:空")]), np.array([1.0], dtype=np.float32))
                f = np.concatenate([S.tok(t)[0] for t in g])
                v = np.concatenate([S.tok(t)[1] for t in g])
                return pl.word(f, v / max(1e-9, float(np.linalg.norm(v))))

            class FLex(Lex):
                def __init__(self):
                    self.keys = f_names
                    self.idx = {k: i for i, k in enumerate(f_names)}
                    W = np.stack([fwave(g) for g in frep])
                    self.norm = np.maximum(1e-9, np.linalg.norm(W, axis=1)).astype(np.float32)
                    self.Ln = (W / self.norm[:, None]).astype(np.complex64)
            flex = FLex()
            lex_c = Lex(pl, S, list({key_of(t): t for t in cand3 + [c for x in fseqs for c, _ in x]}.values()))
            hf = hc = nf = nfc = hm = 0
            for x in fseqs:
                ws, lx = [], []
                for c, g in x:
                    ws += [pl.word(*S.tok(c)), fwave(g)]
                    lx += [lex_c, flex]
                g_, _ = decode(pl, pl.encode(ws), len(ws), lx)
                for k, (c, g) in enumerate(x):
                    f = "".join(t["form"] for t in g)
                    hf += g_[2 * k + 1] == flex.idx[f]
                    hm += f == most_fill
                    nf += 1
                    hc += g_[2 * k] == lex_c.idx[key_of(c)]
                    nfc += 1
            r["4_詰め物も入れた形"] = dict(詰め物_上位1=round(hf / nf, 4), 内容語_上位1=round(hc / nfc, 4), いつも同じ答え=round(hm / nf, 4),
                                     当てずっぽう=round(1 / len(f_names), 5), 詰め物の数=nf)
            del lex_c, flex
            # 1. 候補 全部（D 512・2,048 だけ、文を絞る）
            if D <= 2048:
                lexA = Lex(pl, S, cand_all)
                h1 = h5 = n4 = 0
                for s in sub:
                    g, sc = decode(pl, enc(s), len(s), [lexA] * len(s))
                    for k, t in enumerate(s):
                        ti = lexA.idx[key_of(t)]
                        h1 += g[k] == ti
                        h5 += ti in np.argpartition(-sc[k], 5)[:5]
                        n4 += 1
                r["1_語の戻り_候補全部"] = dict(上位1=round(h1 / n4, 4), 上位5=round(h5 / n4, 4), 当てずっぽう_上位1=round(1 / len(cand_all), 6), 語=n4)
                del lexA
            r["秒"] = round(time.time() - t1)
            res["型"][f"D {D}・{mode}"] = r
            print(f"D {D}・{mode}", json.dumps({k: v for k, v in r.items() if k != "1_語の戻り_候補3000"}, ensure_ascii=False)[:600],
                  json.dumps({k: r["1_語の戻り_候補3000"][k] for k in ("上位1", "上位5", "取り違えの属性のコサイン")}, ensure_ascii=False), flush=True)
            save()
            del lex
    # 図の材料: 「友達 と 映画 を 見 ました」、D 256・拡散、全部の語
    S = G.AttrSpace(cb, func, 256, bands=("B", "C", "D", "E", "G"))
    pl = PL.Placer(Sp(S), "拡散")
    s = own[1]
    fn_toks = list({key_of(t): t for x in own for t in x if not G.is_content(t)}.values())
    lex = Lex(pl, S, list({key_of(t): t for t in cand3 + fn_toks + s}.values()))
    X = pl.encode([pl.word(*S.tok(t)) for t in s])
    Y = pl.take(X, centers(pl, len(s))[2])
    sc = lex.score(Y[None, :])[0]
    o = np.argsort(-sc)
    ti = lex.idx[key_of(s[2])]
    res["図の材料"] = dict(文=" ".join(t["form"] for t in s), D=256, 置き方="拡散", 波_実部=[round(float(x), 3) for x in X.real], 波_虚部=[round(float(x), 3) for x in X.imag],
                       読んだ位置=3, 正解=s[2]["form"], 候補の数=len(lex.keys), 近さ_上位300=[round(float(sc[i]), 3) for i in o[:300]],
                       上位300の語=[lex.keys[i][0] for i in o[:300]], 正解の近さ=round(float(sc[ti]), 3), 正解の順位=int(np.nonzero(o == ti)[0][0]) + 1)
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
