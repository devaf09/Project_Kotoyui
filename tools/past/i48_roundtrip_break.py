"""波にしてから戻す試し（本物の語板）を、わざと壊す（人工の符号でやった tools/past/i48_roundtrip_fail.py の本物の語板版）。

置き方は拡散、候補は語板の語 3,000（tools/past/i48_roundtrip_board.py と同じ選び方）、文は自作の 10 文 ＋ 試しの文の内容語の並び。
D は 2,048 と 512。読み方は位置ごとに読んで、読めた語を引いて 2 回読み直す（i48_roundtrip_board と同じ）。
1. 点を欠けさせる: 文の波の点のうち 20・40・60・80% を 0 にした時の語の戻り（どの点を消すかは乱数）。
2. 雑音を足す: 雑音の強さを、文の波の強さの 1・2・3 倍にした時の語の戻り（複素の正規の雑音）。
3. 1 点ずらし: 位置の鍵（巡回のずれ）を δ = ±1・2 点ずらして読んだ時の語の戻り。あわせて、語板の語の波の自己相関
   |<w, roll(w, δ)>| / |w|²（候補 3,000 の平均。白い符号なら 1/√D くらい）と、人工の乱数の符号の同じ値。
4. 似て非なる語: 文の真ん中の位置の語を、次の語に差し替えた文と元の文で、
   (a) その位置を読んだ時の元の語への近さ（読み直しなし）と順位、(b) 文全体の波どうしの似具合（複素のコサインの実部）。
   差し替える語: ほぼ同じ（属性のコサイン .9 以上 1 未満でいちばん近い語）・半分ほど（.4〜.6）・無関係（.1 未満）・全く同じ属性（1）。
   ほかに、語順を 2 語入れ替えた文（真ん中と次の語）と、別の文（次の文）。
5. 想定外の反応: 外れた選びのうち、同じ文のほかの位置の語を選んだ割合。同じ文に属性の近い語（.9 以上）が 2 つある時に、
   片方の位置でもう片方が鳴る（選ばれる）割合。
手で置く物: 欠けさせる割合・雑音の倍率・δ の並び、似具合の帯（.9・.4〜.6・.1）、乱数の種。
python tools/past/i48_roundtrip_break.py --conllu ... --content ... --func ... --out out.json [--test 1000]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
import i48ab_long_grain as G  # noqa: E402
import i48ab_place as PL  # noqa: E402
import i48_roundtrip_board as RB  # noqa: E402

DS = (2048, 512)
DROPS = (0.2, 0.4, 0.6, 0.8)
NOISES = (1.0, 2.0, 3.0)
DELTAS = (1, -1, 2)


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
    kidx = {k: i for i, k in enumerate(keys)}
    res = dict(文=len(seqs), 語=sum(len(s) for s in seqs), 候補=len(cand), 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for D in DS:
        t1 = time.time()
        S = G.AttrSpace(cb, func, D, bands=("B", "C", "D", "E", "G"))
        pl = PL.Placer(RB.Sp(S), "拡散")
        lex = RB.Lex(pl, S, cand)
        rng = np.random.default_rng(0)
        enc_ids = lambda ids: pl.encode([lex.L(np.array([i]))[0] for i in ids])
        ids_of = [[kidx[RB.key_of(t)] for t in s] for s in seqs]

        def recover(transform, delta=0):
            h = n = 0
            for ids in ids_of:
                X = transform(enc_ids(ids))
                if delta:
                    cen = [c + delta for c in RB.centers(pl, len(ids))]
                    Y = np.stack([pl.take(X, c) for c in cen])
                    g = np.argmax(lex.score(Y), axis=1)
                else:
                    g, _ = RB.decode(pl, X, len(ids), [lex] * len(ids))
                h += sum(int(x == y) for x, y in zip(g, ids))
                n += len(ids)
            return round(h / n, 4)
        r = {}
        r["壊さない"] = recover(lambda X: X)
        # 1. 点を欠けさせる
        r["1_点を欠けさせる"] = {}
        for p in DROPS:
            def drop(X, p=p):
                m = rng.random(D) >= p
                return X * m
            r["1_点を欠けさせる"][f"{int(p * 100)}%"] = recover(drop)
        # 2. 雑音を足す
        r["2_雑音"] = {}
        for k in NOISES:
            def noisy(X, k=k):
                lvl = k * float(np.linalg.norm(X)) / np.sqrt(2 * D)
                return X + (lvl * (rng.normal(size=D) + 1j * rng.normal(size=D))).astype(np.complex64)
            r["2_雑音"][f"{k:g} 倍"] = recover(noisy)
        # 3. 1 点ずらし（読み直しなしで、ずらさない時も並べる）
        r["3_ずらして読む"] = {"ずらさない（読み直しなし）": recover(lambda X: X, delta=0 if False else 0)}
        h = n = 0
        for ids in ids_of:
            X = enc_ids(ids)
            Y = np.stack([pl.take(X, c) for c in RB.centers(pl, len(ids))])
            g = np.argmax(lex.score(Y), axis=1)
            h += sum(int(x == y) for x, y in zip(g, ids))
            n += len(ids)
        r["3_ずらして読む"]["ずらさない（読み直しなし）"] = round(h / n, 4)
        for d in DELTAS:
            r["3_ずらして読む"][f"δ {d:+d}"] = recover(lambda X: X, delta=d)
        ac = {}
        Ln = lex.Ln
        for d in DELTAS:
            ac[f"δ {d:+d}"] = round(float(np.mean(np.abs(np.sum(np.conj(Ln) * np.roll(Ln, d, axis=1), axis=1)))), 4)
        rc = np.exp(2j * np.pi * rng.random((len(cand), D))).astype(np.complex64) / np.sqrt(D)
        r["3_自己相関"] = dict(語板の語の波=ac, 人工の乱数の符号={f"δ {d:+d}": round(float(np.mean(np.abs(np.sum(np.conj(rc) * np.roll(rc, d, axis=1), axis=1)))), 4) for d in DELTAS},
                            目安_1割るルートD=round(1 / np.sqrt(D), 4))
        del rc
        # 4. 似て非なる語
        names = sorted({nm for t in cand for nm, _ in RB.attr_names(S, t)})
        ni = {nm: i for i, nm in enumerate(names)}
        A = np.zeros((len(cand), len(names)), dtype=np.float32)
        for i, t in enumerate(cand):
            for (nm, _), v in zip(RB.attr_names(S, t), S.attrs(t["lemma"], t["upos"])[1]):
                A[i, ni[nm]] = v
        A /= np.maximum(1e-9, np.linalg.norm(A, axis=1, keepdims=True))
        kinds = ("全く同じ属性", "ほぼ同じ", "半分ほど", "無関係", "語順の入れ替え", "別の文")
        acc = {k: dict(元の語への近さ=[], 元の語の順位=[], 元の語が一位=[], 文の波の似具合=[]) for k in kinds}
        for si, ids in enumerate(ids_of):
            if len(ids) < 2:
                continue
            k = len(ids) // 2
            w = ids[k]
            cs = A @ A[w]
            cs[w] = -9
            pick = {"全く同じ属性": np.nonzero(cs > 0.9999)[0],
                    "ほぼ同じ": np.nonzero((cs >= 0.9) & (cs <= 0.9999))[0],
                    "半分ほど": np.nonzero((cs >= 0.4) & (cs <= 0.6))[0],
                    "無関係": np.nonzero((cs < 0.1) & (cs > -9))[0]}
            X0 = enc_ids(ids)
            n0 = float(np.linalg.norm(X0))
            alts = {}
            for nm, c in pick.items():
                if len(c):
                    j = int(c[np.argmax(cs[c])]) if nm == "ほぼ同じ" else int(c[rng.integers(len(c))])
                    alts[nm] = ids[:k] + [j] + ids[k + 1:]
            if k + 1 < len(ids):
                sw = list(ids)
                sw[k], sw[k + 1] = sw[k + 1], sw[k]
                if sw[k] != ids[k]:
                    alts["語順の入れ替え"] = sw
            other = ids_of[(si + 1) % len(ids_of)]
            alts["別の文"] = other
            for nm, alt in alts.items():
                X1 = enc_ids(alt)
                acc[nm]["文の波の似具合"].append(float(np.real(np.vdot(X0, X1))) / (n0 * float(np.linalg.norm(X1))))
                if nm == "別の文":
                    continue
                Y = pl.take(X1, RB.centers(pl, len(alt))[k])
                sc = lex.score(Y[None, :])[0]
                acc[nm]["元の語への近さ"].append(float(sc[w]))
                acc[nm]["元の語の順位"].append(int((sc > sc[w]).sum()) + 1)
                acc[nm]["元の語が一位"].append(int(np.argmax(sc) == w))
        # 差し替えない時の同じ位置の近さ（比べの土台）
        base_sc = []
        for ids in ids_of:
            if len(ids) < 2:
                continue
            k = len(ids) // 2
            Y = pl.take(enc_ids(ids), RB.centers(pl, len(ids))[k])
            base_sc.append(float(lex.score(Y[None, :])[0][ids[k]]))
        r["4_似て非なる語"] = dict(差し替えない時の元の語への近さ=round(float(np.mean(base_sc)), 4),
                               **{nm: dict(文の数=len(v["文の波の似具合"]),
                                           元の語への近さ=round(float(np.mean(v["元の語への近さ"])), 4) if v["元の語への近さ"] else None,
                                           元の語の順位_中央値=float(np.median(v["元の語の順位"])) if v["元の語の順位"] else None,
                                           元の語が一位=round(float(np.mean(v["元の語が一位"])), 4) if v["元の語が一位"] else None,
                                           文の波の似具合=round(float(np.mean(v["文の波の似具合"])), 4) if v["文の波の似具合"] else None)
                                  for nm, v in acc.items()})
        # 5. 想定外の反応
        wrong = same_sent = 0
        pair_n = pair_ring = 0
        for ids in ids_of:
            g, _ = RB.decode(pl, enc_ids(ids), len(ids), [lex] * len(ids))
            for k, (x, y) in enumerate(zip(g, ids)):
                if x != y:
                    wrong += 1
                    same_sent += int(x in ids)
            for i in range(len(ids)):
                for j in range(len(ids)):
                    if i != j and ids[i] != ids[j] and float(A[ids[i]] @ A[ids[j]]) >= 0.9:
                        pair_n += 1
                        pair_ring += int(g[i] == ids[j])
        r["5_想定外の反応"] = dict(外れ=wrong, 外れのうち同じ文のほかの語を選んだ割合=round(same_sent / max(1, wrong), 4),
                               属性の近い語が同じ文に2つある位置=pair_n, そのうちもう片方が鳴った割合=round(pair_ring / max(1, pair_n), 4))
        r["秒"] = round(time.time() - t1)
        res["型"][f"D {D}・拡散"] = r
        print(f"D {D}", json.dumps(r, ensure_ascii=False), flush=True)
        save()
        del lex
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
