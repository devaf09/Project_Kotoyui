"""alt1b: 「自分が変わったことを自分で言えるか」（alt1）を、本物の作品で測る（数えだけ。本文は出さない）。

alt1（intervention/alt/alt1_self_change.py）の入力の作り手だけを、青空文庫の作品に替える。器（I49a の Vessel(2048, 20, 1024)）も、答え方も同じ。
作品の読み方と文の入力は I49b と同じ（tools/i49b_read_long.py の pick_works、入力 B = 流れから学んだ意味の波、前処理なし）。
  ベクトルが 0 の文は除く（alt1 と数え方をそろえるため。I49b は tick にしていた）。
1 回分（種ごと）:
  作品の中から 10 個を選び、5 つを知っている作品、5 つを初めての作品にする。
  どの作品も、前の 8 割を「読む所」、後ろの 2 割を「読んでいない文」（候補の言い当て用、でたらめに probe 本）に分ける。
  いつもの混ぜ物 = 知っている作品の読む所から、続いた block 文を、作品と場所をでたらめに選んでつなぐ。
  読む作品は、読む所の続いた read 文（場所はでたらめ）。
手で置いた値: 作品の数（--works）、1 作品の文の上限 3,000（I49b と同じ）、8 割と 2 割、block 50、あとは alt1 と同じ（base 1,000、read 200、probe 50、時間 0・100・1,000 文）。

python tools/alt1b_read_works.py --src <元の本文の置き場> --conllu <青空文庫の conllu> --wave <波の板（.npz と .json の名前）> \
    --cache <文のベクトルの置き場 .npz（リポジトリの外）> --out <stats.json> [--works 20] [--trials 100]
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
sys.path.insert(0, os.path.join(HERE, "..", "intervention", "alt"))
import alt1_self_change as A1  # noqa: E402

BLOCK, HEAD_PART = 50, 0.8


class Works:
    """本物の作品の文のベクトル。alt1 の World と同じ口（usual・work・probe）を持つ。"""

    def __init__(self, rng, vecs, known):
        self.rng, self.K = rng, int(known)
        pick = rng.permutation(len(vecs))[: 2 * self.K]
        self.v = [vecs[i] for i in pick]
        self.cut = [int(len(x) * HEAD_PART) for x in self.v]

    def work(self, w, n):
        n = min(n, self.cut[w])
        s = int(self.rng.integers(0, self.cut[w] - n + 1))
        return self.v[w][s: s + n]

    def usual(self, n):
        out, got = [], 0
        while got < n:
            x = self.work(int(self.rng.integers(0, self.K)), min(BLOCK, n - got))
            out.append(x)
            got += len(x)
        return np.concatenate(out) if out else np.zeros((0, self.v[0].shape[1]), dtype=np.complex64)

    def probe(self, w, n):
        tail = np.arange(self.cut[w], len(self.v[w]))
        return self.v[w][self.rng.choice(tail, size=min(n, len(tail)), replace=False)]


def load_vecs(a):
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache)
        return [z[k] for k in sorted(z.files, key=lambda s: int(s[1:]))]
    import spacy
    import i48_wave_board as W48
    import i49b_read_long as R                     # noqa: F401  （tools/ は HERE にある）
    t0 = time.time()
    works = R.pick_works(a.src, a.conllu, n=a.works)
    print("作品", len(works), [len(w) for w in works], f"{time.time() - t0:.0f} 秒", flush=True)
    W = np.load(a.wave + ".npz")["W"]
    m = json.load(open(a.wave + ".json", encoding="utf-8"))
    ix, idf = {w: i for i, w in enumerate(m["vocab"]) if w is not None}, m["idf"]
    nlp = spacy.load("ja_ginza")
    vecs = []
    for w in works:
        rows = []
        for doc in nlp.pipe([s for s, _ in w], batch_size=256):
            ws = [t.lemma_ for t in doc if t.pos_ in W48.CONTENT and t.lemma_ not in W48.STOP and t.lemma_ in ix]
            if not ws:
                continue
            v = (W[[ix[x] for x in ws]] * np.asarray([idf.get(x, 1.0) for x in ws], dtype=np.float32)[:, None]).sum(0)
            n = float(np.linalg.norm(v))
            if n > 0:
                rows.append((v / n).astype(np.complex64))
        vecs.append(np.stack(rows))
        print("読み", len(w), "→", len(rows), f"{time.time() - t0:.0f} 秒", flush=True)
    if a.cache:
        np.savez(a.cache, **{f"w{i}": v for i, v in enumerate(vecs)})
    return vecs


def alike(vecs):
    """作品の見分けやすさの目安: 作品の平均の向きどうしの |cos|（別の作品の前 8 割どうし／同じ作品の前 8 割と後ろ 2 割）。"""
    def mean(x):
        s = x.sum(0)
        return s / max(float(np.linalg.norm(s)), 1e-30)
    head = np.stack([mean(v[: int(len(v) * HEAD_PART)]) for v in vecs])
    tail = np.stack([mean(v[int(len(v) * HEAD_PART):]) for v in vecs])
    c = np.abs(head @ np.conj(head).T)
    other = c[~np.eye(len(vecs), dtype=bool)]
    same = np.abs(np.sum(head * np.conj(tail), axis=1))
    return dict(別の作品=round(float(other.mean()), 4), 同じ作品の前と後ろ=round(float(same.mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    for k in ("src", "conllu", "wave"):
        ap.add_argument("--" + k, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--works", type=int, default=20)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--known", type=int, default=5)
    ap.add_argument("--base", type=int, default=1000)
    ap.add_argument("--read", type=int, default=200)
    ap.add_argument("--probe", type=int, default=50)
    ap.add_argument("--bands", type=int, default=20)
    ap.add_argument("--longest", type=float, default=1024.0)
    a = ap.parse_args()
    t0 = time.time()
    vecs = load_vecs(a)
    assert len(vecs) >= 2 * a.known, "作品が足りない"
    a.dim = int(vecs[0].shape[1])
    res = A1.run(a, world=lambda rng: Works(rng, vecs, a.known))
    res.update(作品の文の数=[int(len(v)) for v in vecs], 向きの近さ=alike(vecs), 秒=round(time.time() - t0))
    print("作品", len(vecs), "文", res["作品の文の数"], "向きの近さ", res["向きの近さ"])
    A1.show(res, a.bands)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
