"""文を波に変えて戻す試しを、わざと失敗させて「上手く行く向き」を探る（持ち主の案、未確定）。tools/past/i48_roundtrip.py の続き。

土台: 自作の 10 文（71 語）。語 = 独立の乱数の位相の符号（拡散）。位置 k = k × SHIFT の巡回のずれ。読み直し 2 回。
失敗の縁に置いて（土台が少しだけ外れる所）、1 つずつ作りを変え、戻せた割合が上がるか下がるかを見る。
  縁 A: 1 文を 1 本の波・候補 3,000・D 64（土台 .86 前後）
  縁 B: 10 文を鍵をつけて 1 本に重ねる・候補 300・D 512（土台 .82 前後）
変える物（それぞれ、ほかは土台のまま）:
  符号: 乱数の位相（土台）／ ±1 ／ 疎ら（D の 1/8 だけ値あり）／ 低い帯だけ（滑らかな符号: D の 1/8 の帯の外を 0 にした物）／ 振幅もばらばら（複素正規）
  位置の置き方: ずれ SHIFT 7（土台）・1・D/2 ／ 位置ごとの並べ替え（乱数の並べ替え）／ 位置ごとの位相の板
  似た語: 候補の符号に共通の成分を混ぜる（似た語の多さ。ρ = 0.3・0.6・0.9）
  雑音: 波に複素の雑音を足す（波の強さに対して 0.5・1・2 倍）
  読み直し: 0・1・2（土台）・4 回 ／ 1 つずつ読む（いちばん近さの大きい位置から決めて引く）
  語の強さ: 1 語だけ 3 倍に強くする ／ 語ごとにばらばら（0.5〜1.5 倍）
  文の長さ: 2 文をつないで 1 文にする（語の数 2 倍）
  鍵（縁 B だけ）: 乱数の位相（土台）／ ±1 ／ ずれだけ（文ごとに違う巡回のずれ）／ 重ねる文の数 5・20
測り: 語を戻せた割合（上位 1）、文の全部の語を戻せた割合。乱数の種 10 通りの平均と、種ごとのばらつき（標準偏差）。
手で置く物: 縁の D と候補の数、変える量（上の並び）、種の数 10。
python tools/past/i48_roundtrip_fail.py [出力.json]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import json
import os
import sys
import time

import numpy as np

SENTS = ["今日 は 天気 が いい です", "友達 と 映画 を 見 ました", "猫 が 窓 の そば で 寝て います",
         "駅 まで 歩いて 十分 かかり ます", "明日 は 早く 起き ない と いけ ない", "この 本 は とても 面白 かった",
         "雨 が 降っ たら 傘 を 持って 行く", "先生 に 質問 を して も いい です か", "昨日 の 夜 は よく 眠れ なかった",
         "コーヒー を 飲み ながら 音楽 を 聞く"]
SENTS = [s.split() for s in SENTS]
VOC = sorted({w for s in SENTS for w in s})
WID = {w: i for i, w in enumerate(VOC)}
SEEDS = range(10)


def make_codes(V, D, kind, rng, rho=0.0):
    if kind == "±1":
        C = (rng.integers(0, 2, (V, D)) * 2 - 1).astype(np.complex64)
    elif kind == "疎ら":
        C = np.zeros((V, D), dtype=np.complex64)
        k = max(1, D // 8)
        for i in range(V):
            idx = rng.choice(D, k, replace=False)
            C[i, idx] = np.exp(2j * np.pi * rng.random(k))
    elif kind == "低い帯だけ":
        F = np.zeros((V, D), dtype=np.complex64)
        k = max(1, D // 8)
        F[:, :k] = np.exp(2j * np.pi * rng.random((V, k)))
        C = np.fft.ifft(F, axis=1).astype(np.complex64)
    elif kind == "振幅もばらばら":
        C = (rng.normal(size=(V, D)) + 1j * rng.normal(size=(V, D))).astype(np.complex64)
    else:
        C = np.exp(2j * np.pi * rng.random((V, D))).astype(np.complex64)
    if rho > 0:
        common = np.exp(2j * np.pi * rng.random(D)).astype(np.complex64)
        C = np.sqrt(1 - rho) * C + np.sqrt(rho) * common[None, :]
    return C / np.linalg.norm(C, axis=1, keepdims=True)


class Pos:
    """位置の置き方: put(y, k) と take(X, k)。"""

    def __init__(self, D, kind, rng, nmax=32, shift=7):
        self.kind, self.D, self.shift = kind, D, shift
        if kind == "並べ替え":
            self.perm = [rng.permutation(D) for _ in range(nmax)]
            self.inv = [np.argsort(p) for p in self.perm]
        if kind == "位相の板":
            self.plate = np.exp(2j * np.pi * rng.random((nmax, D))).astype(np.complex64)

    def put(self, y, k):
        if self.kind == "並べ替え":
            return y[self.perm[k]]
        if self.kind == "位相の板":
            return y * self.plate[k]
        return np.roll(y, k * self.shift)

    def take(self, X, k):
        if self.kind == "並べ替え":
            return X[self.inv[k]]
        if self.kind == "位相の板":
            return X * np.conj(self.plate[k])
        return np.roll(X, -k * self.shift)


def encode(pos, C, ids, amp):
    X = np.zeros(C.shape[1], dtype=np.complex64)
    for k, (i, a) in enumerate(zip(ids, amp)):
        X += a * pos.put(C[i], k)
    return X


def decode(pos, X, n, C, rounds, est_amp=False):
    Y0 = np.stack([pos.take(X, k) for k in range(n)])
    g = np.argmax(np.real(Y0 @ np.conj(C).T), axis=1)
    for _ in range(rounds):
        sc0 = np.real(Y0 @ np.conj(C).T)
        a = sc0[np.arange(n), g] if est_amp else np.ones(n, dtype=np.float32)   # 語の強さを変えた時だけ、読めた強さで引く
        Xr = encode(pos, C, g, a)
        Y = Y0 - np.stack([pos.take(Xr, k) for k in range(n)]) + a[:, None] * C[g]
        g = np.argmax(np.real(Y @ np.conj(C).T), axis=1)
    return g


def decode_one_by_one(pos, X, n, C):
    R = X.copy()
    g = [None] * n
    for _ in range(n):
        Y = np.stack([pos.take(R, k) for k in range(n)])
        sc = np.real(Y @ np.conj(C).T)
        best = max((k for k in range(n) if g[k] is None), key=lambda k: sc[k].max())
        g[best] = int(np.argmax(sc[best]))
        R = R - pos.put(C[g[best]], best)
    return np.array(g)


def run(edge, cfg, seed):
    rng = np.random.default_rng(seed)
    D, V = (64, 3000) if edge == "A" else (512, 300)
    C = make_codes(V, D, cfg.get("符号", "位相"), rng, cfg.get("似た語", 0.0))
    shift = {"1": 1, "D/2": D // 2 + 1}.get(cfg.get("ずれ"), 7)
    pos = Pos(D, cfg.get("位置", "ずれ"), rng, shift=shift)
    sents = SENTS
    if cfg.get("長さ") == "2文をつなぐ":
        sents = [SENTS[i] + SENTS[(i + 1) % 10] for i in range(10)]
    rounds = cfg.get("読み直し", 2)
    tok = sen = n = 0

    def amps(L):
        if cfg.get("強さ") == "1語だけ3倍":
            a = np.ones(L, dtype=np.float32)
            a[0] = 3
            return a
        if cfg.get("強さ") == "ばらばら":
            return rng.uniform(0.5, 1.5, L).astype(np.float32)
        return np.ones(L, dtype=np.float32)

    def noisy(X):
        s = cfg.get("雑音", 0.0)
        if s > 0:
            lvl = s * float(np.linalg.norm(X)) / np.sqrt(2 * D)
            X = X + (lvl * (rng.normal(size=D) + 1j * rng.normal(size=D))).astype(np.complex64)
        return X

    def dec(X, L):
        if cfg.get("読み方") == "1つずつ":
            return decode_one_by_one(pos, X, L, C)
        return decode(pos, X, L, C, rounds, est_amp="強さ" in cfg)

    if edge == "A":
        for s in sents:
            ids = np.array([WID[w] for w in s])
            g = dec(noisy(encode(pos, C, ids, amps(len(s)))), len(s))
            tok += int((g == ids).sum())
            n += len(s)
            sen += int((g == ids).all())
        return tok / n, sen / len(sents)
    nst = cfg.get("重ねる数", 10)
    group = [sents[i % 10] for i in range(nst)]
    kk = cfg.get("鍵", "位相")
    if kk == "±1":
        keys = (rng.integers(0, 2, (nst, D)) * 2 - 1).astype(np.complex64)
    elif kk == "ずれだけ":
        sh = rng.choice(D, nst, replace=False)
        keys = None
    else:
        keys = np.exp(2j * np.pi * rng.random((nst, D))).astype(np.complex64)
    X = np.zeros(D, dtype=np.complex64)
    for j, s in enumerate(group):
        x = encode(pos, C, np.array([WID[w] for w in s]), amps(len(s)))
        X += np.roll(x, sh[j]) if keys is None else keys[j] * x
    X = noisy(X)
    for j, s in enumerate(group):
        ids = np.array([WID[w] for w in s])
        Xj = np.roll(X, -sh[j]) if keys is None else np.conj(keys[j]) * X
        g = dec(Xj, len(s))
        tok += int((g == ids).sum())
        n += len(s)
        sen += int((g == ids).all())
    return tok / n, sen / len(group)


VARIANTS = [("土台", {})] + \
    [(f"符号: {k}", {"符号": k}) for k in ("±1", "疎ら", "低い帯だけ", "振幅もばらばら")] + \
    [("位置: ずれ 1", {"ずれ": "1"}), ("位置: ずれ D/2", {"ずれ": "D/2"}), ("位置: 並べ替え", {"位置": "並べ替え"}), ("位置: 位相の板", {"位置": "位相の板"})] + \
    [(f"似た語: ρ {r}", {"似た語": r}) for r in (0.3, 0.6, 0.9)] + \
    [(f"雑音: {s} 倍", {"雑音": s}) for s in (0.5, 1.0, 2.0)] + \
    [(f"読み直し: {r} 回", {"読み直し": r}) for r in (0, 1, 4)] + [("読み方: 1 つずつ", {"読み方": "1つずつ"})] + \
    [("語の強さ: 1 語だけ 3 倍", {"強さ": "1語だけ3倍"}), ("語の強さ: ばらばら", {"強さ": "ばらばら"})] + \
    [("文の長さ: 2 文をつなぐ", {"長さ": "2文をつなぐ"})]
KEY_VARIANTS = [("鍵: ±1", {"鍵": "±1"}), ("鍵: ずれだけ", {"鍵": "ずれだけ"}), ("重ねる数: 5", {"重ねる数": 5}), ("重ねる数: 20", {"重ねる数": 20})]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "roundtrip_fail.json"
    t0 = time.time()
    res = dict(縁=dict(A="1 文を 1 本の波・候補 3,000・D 64", B="10 文を鍵をつけて 1 本に重ねる・候補 300・D 512"), 種=len(SEEDS), 型={})
    for edge in ("A", "B"):
        res["型"][edge] = {}
        for nm, cfg in VARIANTS + (KEY_VARIANTS if edge == "B" else []):
            a = np.array([run(edge, cfg, sd) for sd in SEEDS])
            res["型"][edge][nm] = dict(語=round(float(a[:, 0].mean()), 4), 語_ばらつき=round(float(a[:, 0].std()), 4),
                                      文=round(float(a[:, 1].mean()), 4))
            print(edge, nm, res["型"][edge][nm], round(time.time() - t0), "秒", flush=True)
            json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
