"""波にしてから戻す試し（本物の語板）: 符号の太さを変えて、ずれへの強さと混ざりにくさを両方測る（案、未確定）。

置き方は拡散。属性ごとの符号（D 全体の乱数の位相）を、幅 w の箱で巡回にならして「太く」する（隣の w 点ほどが似た値になる）。
語の波は今までどおり、太くした属性の符号の重みつきの和。w = 1 が今の形。
候補は語板の語 3,000（tools/past/i48_roundtrip_board.py と同じ選び方）、文は自作の 10 文 ＋ 試しの文の内容語の並び。
測る物（w = 1・2・3・4・6・8・16、D 2,048 と 512）:
  (a) 読む位置を δ = 0・1・2 点ずらした時の 1 文の語の戻り（候補 3,000、読み直し 2 回。δ = 0 が (c) の「1 文の戻り」）
  (b) 自作の 10 文を鍵をつけて 1 本に重ね、鍵で戻した時の語の戻り（δ = 0）
  (d) 合わせを探す: 本当は 1 点ずれている時に、位置ごとに δ = −2〜+2 を試して、いちばん強く鳴った所で読む
  あわせて、語の波の 1 点ずらしの自己相関（候補の平均）。
D 2,048 では、位置のずれ SHIFT を AttrSpace の物（D の 0.414 倍）と 7（tools/past/i48_roundtrip.py と同じ）の両方で測る。
手で置く物: 太さの並び、δ の並び、探す幅（±2）、鍵の種。
python tools/past/i48_roundtrip_thick.py --conllu ... --content ... --func ... --out out.json [--test 1000]
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

WIDTHS = (1, 2, 3, 4, 6, 8, 16)
DS = (2048, 512)
DELTAS = (0, 1, 2)
SEARCH = 2


class ThickPlacer(PL.Placer):
    def __init__(self, S, width):
        super().__init__(S, "拡散")
        self.width = width
        k = np.zeros(self.D, dtype=np.complex64)
        k[:width] = 1.0
        self.K = np.fft.fft(k)

    def _code(self, f):
        c = self._codes.get(int(f))
        if c is None:
            c = super()._code(f)
            if self.width > 1:
                c = np.fft.ifft(np.fft.fft(c) * self.K).astype(np.complex64)
                c /= np.linalg.norm(c)
            self._codes[int(f)] = c
        return c


def decode_off(pl, X, n, lex, delta, rounds=RB.ROUNDS):
    cen = RB.centers(pl, n)
    Y0 = np.stack([pl.take(X, c + delta) for c in cen])
    g = np.argmax(lex.score(Y0), axis=1)
    for _ in range(rounds):
        Xr = pl.encode([lex.L(np.array([i]))[0] for i in g])
        Y = Y0 - np.stack([pl.take(Xr, c + delta) for c in cen]) + lex.L(g)
        g = np.argmax(lex.score(Y), axis=1)
    return g


def decode_search(pl, X, n, lex, true_off=1):
    """本当は true_off ずれている波を、位置ごとに −SEARCH〜+SEARCH を試して、いちばん強く鳴った所で読む（読み直しなし）。"""
    cen = RB.centers(pl, n)
    g = []
    for c in cen:
        best = None
        for d in range(-SEARCH, SEARCH + 1):
            sc = lex.score(pl.take(X, c + true_off + d)[None, :])[0]
            j = int(np.argmax(sc))
            if best is None or sc[j] > best[0]:
                best = (sc[j], j)
        g.append(best[1])
    return np.array(g)


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
    own_c = [[t for t in s if G.is_content(t)] for s in own]
    seqs = own_c + [[t for t in s["toks"] if G.is_content(t)] for s in te]
    seqs = [s for s in seqs if s]
    cand = list({RB.key_of(t): t for s in seqs for t in s}.values())
    seen = {RB.key_of(t) for t in cand}
    for t in board_toks:
        if len(cand) >= RB.N_CAND:
            break
        if RB.key_of(t) not in seen:
            cand.append(t)
            seen.add(RB.key_of(t))
    kidx = {RB.key_of(t): i for i, t in enumerate(cand)}
    ids_of = [[kidx[RB.key_of(t)] for t in s] for s in seqs]
    own_ids = ids_of[:len(own_c)]
    res = dict(文=len(seqs), 語=sum(len(s) for s in seqs), 候補=len(cand), 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for D in DS:
        S = G.AttrSpace(cb, func, D, bands=("B", "C", "D", "E", "G"))
        for w, sh in [(w, None) for w in WIDTHS] + ([(w, 7) for w in WIDTHS] if D == DS[0] else []):
            t1 = time.time()
            sp = RB.Sp(S)
            if sh:
                sp.shift = sh                                  # tools/past/i48_roundtrip.py と同じ SHIFT 7（隣の位置が 7 点しか離れない）
            pl = ThickPlacer(sp, w)
            lex = RB.Lex(pl, S, cand)
            enc = lambda ids: pl.encode([lex.L(np.array([i]))[0] for i in ids])
            r = {}
            for d in DELTAS:
                h = n = 0
                for ids in ids_of:
                    g = decode_off(pl, enc(ids), len(ids), lex, d)
                    h += int((g == np.array(ids)).sum())
                    n += len(ids)
                r[f"a_1文_ずれ{d}"] = round(h / n, 4)
            kr = np.random.default_rng(99)
            keys = np.exp(2j * np.pi * kr.random((len(own_ids), D))).astype(np.complex64)
            X = np.zeros(D, dtype=np.complex64)
            for k_, ids in zip(keys, own_ids):
                X += k_ * enc(ids)
            h = n = 0
            for k_, ids in zip(keys, own_ids):
                g = decode_off(pl, np.conj(k_) * X, len(ids), lex, 0)
                h += int((g == np.array(ids)).sum())
                n += len(ids)
            r["b_10文を重ねる"] = round(h / n, 4)
            h = n = 0
            for ids in ids_of:
                g = decode_search(pl, enc(ids), len(ids), lex, true_off=1)
                h += int((g == np.array(ids)).sum())
                n += len(ids)
            r["d_1点ずれを探して読む"] = round(h / n, 4)
            r["自己相関_1点"] = round(float(np.mean(np.abs(np.sum(np.conj(lex.Ln) * np.roll(lex.Ln, 1, axis=1), axis=1)))), 4)
            r["秒"] = round(time.time() - t1)
            nm = f"D {D}・太さ {w}" + (f"・SHIFT {sh}" if sh else f"・SHIFT {S.shift}")
            res["型"][nm] = r
            print(nm, json.dumps(r, ensure_ascii=False), flush=True)
            save()
            del lex, pl
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
