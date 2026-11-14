"""I48g: 語板 v3 の L2。属性の帯（A〜G）を波にする試し（仕様書 9 章の (a) と (b) を並べて測る）。

(a) v2 の割合のまま: 語ごとに、帯の中の属性 → 割合（合計 1）の表。
(b) 波: 語ごと・帯ごとに 1 本の波 = Σ 割合 × 属性の印（乱数の位相）。読む時は印との共鳴 Re<印, 波>/D。
    雑音の床（σ√(2 ln 属性の数)、σ = |波|/(D√2)）より下は 0 にする。
    比べに、全部の帯を 1 本に重ねた形（語ごとに 1 本）も測る。

物差し:
- 全変動: 読み戻した割合（帯の中で合計 1 にそろえる）と、元の割合の差の半分の合計。0 が同じ、1 が全く違う。帯ごと。
- 上位の一致: 元の割合でいちばん大きい属性が、読み戻しでもいちばんか。
- 崩した文の見分け（lexicon/content_light.py の odd_sentence_test と同じ点。AUROC）: 役の入れ替え、名詞の入れ替え。
  点 = 名詞の C の帯の「動詞/役」の割合 ＋ 動詞の C の帯の「主語か目的語:名詞のクラス」の割合。
- 大きさ: 語 1 つあたりのバイト。

実行: python intervention/i48g_attr_band_wave.py --boards <content_boards.json> --triples <test_triples.json> --D 512,2048 --out i48g.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i48_wave_board import Store  # noqa: E402

log = logging.getLogger("i48g")


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


class Bands:
    """属性の印と、語の帯の波。mode = "band"（帯ごとに 1 本）か "one"（全部の帯を 1 本に）。"""

    def __init__(self, boards, D, store="p8", mode="band", seed=0):
        self.D, self.mode, self.store = D, mode, Store(store)
        self.attrs = {}
        for b in boards:
            for band, d in b["attrs"].items():
                self.attrs.setdefault(band, {}).update(dict.fromkeys(d))
        self.attrs = {band: sorted(v) for band, v in sorted(self.attrs.items())}
        self.aix = {band: {x: i for i, x in enumerate(v)} for band, v in self.attrs.items()}
        self.marks = {band: np.exp(1j * np.random.default_rng(zlib.crc32(f"帯:{band}:{seed}".encode())).uniform(0, 2 * np.pi, (len(v), D))
                                   ).astype(np.complex64) for band, v in self.attrs.items()}
        self.n_all = sum(len(v) for v in self.attrs.values())

    def bytes_per_word(self, board):
        n = len(board["attrs"]) if self.mode == "band" else 1
        return n * self.store.bytes_per(self.D)

    def wave(self, board, band):
        """語の帯の波（しまった形から戻した物）。mode "one" では band によらず同じ 1 本。"""
        if self.mode == "band":
            d = board["attrs"].get(band)
            if not d:
                return None
            w = sum(r * self.marks[band][self.aix[band][x]] for x, r in d.items())
        else:
            if not board["attrs"]:
                return None
            w = sum(r * self.marks[bd][self.aix[bd][x]] for bd, d in board["attrs"].items() for x, r in d.items())
        return self.store.decode_np(self.store.encode(w[None]))[0]

    def floor(self, w, band):
        n = len(self.attrs[band]) if self.mode == "band" else self.n_all
        return np.linalg.norm(w) / (self.D * math.sqrt(2)) * math.sqrt(2 * math.log(n))

    def read(self, board, band):
        """帯の属性ごとの明るさ（床より下は 0）。帯が無ければ None。"""
        w = self.wave(board, band)
        if w is None or not board["attrs"].get(band):
            return None
        v = (self.marks[band] @ np.conj(w)).real / self.D
        return np.where(v > self.floor(w, band), v, 0.0)

    def read_one(self, board, band, attr):
        j = self.aix.get(band, {}).get(attr)
        w = self.wave(board, band)
        if j is None or w is None:
            return 0.0
        v = float((self.marks[band][j] @ np.conj(w)).real / self.D)
        return v if v > self.floor(w, band) else 0.0


class Table:
    """(a) 割合のまま。"""

    def read_one(self, board, band, attr):
        return board["attrs"].get(band, {}).get(attr, 0.0)


def fidelity(bd, boards):
    out = {}
    for band in bd.attrs:
        tv, top, kept, n = 0.0, 0, 0.0, 0
        for b in boards:
            d = b["attrs"].get(band)
            if not d:
                continue
            t = np.zeros(len(bd.attrs[band]))
            for x, r in d.items():
                t[bd.aix[band][x]] = r
            v = bd.read(b, band)
            s = v.sum()
            tv += 0.5 * np.abs((v / s if s > 0 else v) - t).sum()
            top += int(v.argmax() == t.argmax() and s > 0)
            kept += float(((v > 0) & (t > 0)).sum() / (t > 0).sum())
            n += 1
        out[band] = dict(語=n, 全変動=round(tv / n, 4), 上位の一致=round(top / n, 4), 残った属性の割合=round(kept / n, 4))
    return out


def odd_test(rd, boards, triples, seed=1):
    idx = {b["name"]: b for b in boards}
    cls_of = {}
    for b in boards:
        dd = b["attrs"].get("D")
        if b["pos"] == "名詞" and dd:
            cls_of[b["name"]] = max(dd, key=dd.get).split(":", 1)[1]
    nouns = [b["name"] for b in boards if b["pos"] == "名詞"]
    rng = random.Random(seed)
    memo = {}

    def score(noun, verb, role):
        k = (noun, verb, role)
        if k not in memo:
            s = rd.read_one(idx[noun], "C", f"{verb}/{role}") if noun in idx else 0.0
            if verb in idx and noun in cls_of:
                s += rd.read_one(idx[verb], "C", f"{'主語' if role == 'する' else '目的語'}:{cls_of[noun]}")
            memo[k] = s
        return memo[k]

    T = [t for t in triples if t["noun"] in idx and t["verb"] in idx]
    pos = [score(t["noun"], t["verb"], t["role"]) for t in T]
    swap = [score(t["noun"], t["verb"], "される" if t["role"] == "する" else "する") for t in T]
    repl = [score(rng.choice(nouns), t["verb"], t["role"]) for t in T]
    return dict(組=len(T), 役の入れ替え=round(auroc(pos, swap), 4), 名詞の入れ替え=round(auroc(pos, repl), 4),
                点が0の割合=round(float(np.mean(np.array(pos) == 0)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", required=True, help="語板 v2 の content_boards.json")
    ap.add_argument("--triples", required=True, help="崩した文の試し用の組 test_triples.json")
    ap.add_argument("--D", default="512,2048")
    ap.add_argument("--store", default="c64,p8")
    ap.add_argument("--n_fidelity", type=int, default=1500, help="全変動を測る語の数（等間隔に取る）")
    ap.add_argument("--n_triples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    boards = [b for b in json.load(open(a.boards, encoding="utf-8")) if b["attrs"]]
    triples = json.load(open(a.triples, encoding="utf-8"))
    triples = random.Random(a.seed).sample(triples, min(a.n_triples, len(triples)))
    sub = boards[:: max(1, len(boards) // a.n_fidelity)]
    n_attr = np.mean([sum(len(d) for d in b["attrs"].values()) for b in boards])
    res = dict(語板=len(boards), 語あたりの属性=round(float(n_attr), 1), rows={})
    res["rows"]["(a) 割合のまま"] = dict(崩した文=odd_test(Table(), boards, triples), 語あたりのバイト=round(float(n_attr) * 6))  # 名札 4 ＋ 割合 2
    log.info("(a) 割合のまま: %s", res["rows"]["(a) 割合のまま"])
    for D in [int(x) for x in a.D.split(",")]:
        for store in a.store.split(","):
            for mode in ("band", "one"):
                bd = Bands(boards, D, store, mode, a.seed)
                label = f"(b) 波 D {D} {store} " + ("帯ごとに 1 本" if mode == "band" else "全部を 1 本")
                r = dict(忠実さ=fidelity(bd, sub), 崩した文=odd_test(bd, boards, triples),
                         語あたりのバイト=round(float(np.mean([bd.bytes_per_word(b) for b in boards]))))
                res["rows"][label] = r
                log.info("%s: 崩した文 %s、バイト %d", label, r["崩した文"], r["語あたりのバイト"])
                log.info("    全変動 %s", {k: v["全変動"] for k, v in r["忠実さ"].items()})
                log.info("    上位の一致 %s", {k: v["上位の一致"] for k, v in r["忠実さ"].items()})
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
