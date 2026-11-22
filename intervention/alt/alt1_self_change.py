"""alt1: 自分が変わったことを、自分で言えるか（別の助言からの試し。会話にはつながない）。

問い: 器（I49a の Vessel、作りは変えない）が何かを読んで変わった時、
  (a) 「いつもと違う物を読んだ」と、中の数だけから分かるか。
  (b) 「何を読んで変わったか」を、候補の中から当てられるか。
  (c) (b) に、前の自分（読む前の波の写し）は要るか。今の自分だけでも当たるなら、写しは要らない。

やり方（自作の入力。本物の文章は使わない）:
  作品 = 向きの中心 1 本。文 = 共通の向き + 作品の中心 + 雑音（長さ 1）。
  知っている作品 5 つを混ぜて base 文読む → 写し 0 → 作品 1 つを read 文読む → 写し 1
  → 知っている作品の混ぜ物を、さらに 100 文、1,000 文読んだ所でも写しを取る（時間がたった後）。
  候補は、知っている 5 つ + 初めての 5 つ。候補ごとに、読んでいない文 probe 本で言い当てさせ、外れ（波ごと）を見る。
    前の自分と比べる: 外れがいちばん下がった候補を答える（写し 0 の外れ − 今の外れ）。
    今の自分だけ    : 今の外れがいちばん小さい候補を答える。
    他人の前と比べる: 同じ作品の別の文を読んだ別の個体の写しと比べる（前の「自分」である事が要るかの比べ）。
  読む物は 2 通り: 初めての作品、知っている作品の読み直し。でたらめは 10%。
  (a) は、同じ長さだけ「いつもの混ぜ物」を読んだ時と、波の向きの替わり方（1 − |cos|）を比べる（AUC）。

手で置いた値: 次元 256、波 20 本・半減期 1〜1,024、作品の数 5 + 5、base 1,000、read 200、probe 50、
  時間の 3 通り（0・100・1,000 文）、文の作り（共通 1、中心 mix、雑音 1）、mix の 3 通り、回数、種。
乱数は入力を作る所だけ（種で再現できる）。器は乱数を使わない。

python intervention/alt/alt1_self_change.py --trials 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import i49a_inner_vessel as V  # noqa: E402

DELAYS = (0, 100, 1000)


def cvec(rng, *shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class World:
    """作品の中心と、共通の向き。sent(w, n) で作品 w の文を n 本作る。"""

    def __init__(self, rng, dim, works, mix):
        self.rng, self.D, self.mix = rng, dim, float(mix)
        self.common = unit(cvec(rng, dim))
        self.center = unit(cvec(rng, works, dim))

    def sent(self, w, n):
        w = np.broadcast_to(np.asarray(w), (n,))
        noise = unit(cvec(self.rng, n, self.D))
        return unit(self.common[None] + self.mix * self.center[w] + noise).astype(np.complex64)


def sur_of(S, X):
    """波の写し S（波, 次元）で、文 X（本, 次元）を言い当てた時の外れ（波, 本）。Vessel.guess と同じ式。"""
    n = np.linalg.norm(S, axis=1)
    out = np.ones((S.shape[0], X.shape[0]), dtype=np.float32)
    ok = n > 0
    out[ok] = 1.0 - np.minimum(1.0, np.abs(S[ok] @ np.conj(X).T) / n[ok, None])
    return out


def turn(S0, S1):
    """波ごとの向きの替わり方 1 − |cos|。"""
    d = np.abs(np.sum(S0 * np.conj(S1), axis=1))
    n = np.linalg.norm(S0, axis=1) * np.linalg.norm(S1, axis=1)
    return 1.0 - d / np.maximum(n, 1e-30)


def one(seed, kind, a):
    """1 回分。kind: new（初めての作品）・re（知っている作品の読み直し）・usual（いつもの混ぜ物）。"""
    rng = np.random.default_rng(seed)
    K = a.known
    wd = World(rng, a.dim, 2 * K, a.mix)
    v = V.Vessel(a.dim, a.bands, a.longest)
    for x in wd.sent(rng.integers(0, K, a.base), a.base):
        v.hear(x)
    S0 = v.S.copy()
    o = V.Vessel(a.dim, a.bands, a.longest)                                        # 別の個体: 同じ作品の、別の文を読んだ
    for x in wd.sent(rng.integers(0, K, a.base), a.base):
        o.hear(x)
    target = int(rng.integers(K, 2 * K)) if kind == "new" else int(rng.integers(0, K)) if kind == "re" else -1
    read = wd.sent(target, a.read) if target >= 0 else wd.sent(rng.integers(0, K, a.read), a.read)
    for x in read:
        v.hear(x)
    probes = [wd.sent(w, a.probe) for w in range(2 * K)]
    s0 = np.stack([sur_of(S0, p).mean(axis=1) for p in probes], axis=1)            # （波, 候補）
    so = np.stack([sur_of(o.S, p).mean(axis=1) for p in probes], axis=1)
    out = dict(target=target, turn=turn(S0, v.S), drop=[], now=[], other=[])
    done = 0
    for d in DELAYS:
        for x in wd.sent(rng.integers(0, K, d - done), d - done):
            v.hear(x)
        done = d
        s1 = np.stack([sur_of(v.S, p).mean(axis=1) for p in probes], axis=1)
        out["drop"].append(s0 - s1)
        out["now"].append(-s1)
        out["other"].append(so - s1)
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def run(a):
    half = V.Vessel(a.dim, a.bands, a.longest).half
    res = dict(half=[round(float(h), 1) for h in half], mix=a.mix, trials=a.trials, chance=1.0 / (2 * a.known))
    runs = {k: [one(a.seed + i, k, a) for i in range(a.trials)] for k in ("new", "re", "usual")}
    for kind in ("new", "re"):
        t = np.array([r["target"] for r in runs[kind]])
        for name in ("drop", "now", "other"):
            for j, d in enumerate(DELAYS):
                x = np.stack([r[name][j] for r in runs[kind]])            # （回, 波, 候補）大きいほど答えらしい
                band = (x.argmax(axis=2) == t[:, None]).mean(axis=0)
                allb = float((x.mean(axis=1).argmax(axis=1) == t).mean())
                res[f"{kind}/{name}/{d}"] = dict(band=[round(float(b), 3) for b in band], all=round(allb, 3))
        tp, tn = np.stack([r["turn"] for r in runs[kind]]), np.stack([r["turn"] for r in runs["usual"]])
        res[f"{kind}/turn_auc"] = [round(auc(tp[:, k], tn[:, k]), 3) for k in range(len(half))]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--bands", type=int, default=20)
    ap.add_argument("--longest", type=float, default=1024.0)
    ap.add_argument("--known", type=int, default=5)
    ap.add_argument("--base", type=int, default=1000)
    ap.add_argument("--read", type=int, default=200)
    ap.add_argument("--probe", type=int, default=50)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = run(a)
    pick = [0, 4, 8, 12, 16, a.bands - 1]
    print("半減期", [res["half"][k] for k in pick], "/ でたらめ", res["chance"], "/ mix", a.mix)
    for key, val in res.items():
        if isinstance(val, dict):
            print(f"{key:14s} 全部の波 {val['all']:.2f}  波ごと", [val["band"][k] for k in pick])
        elif key.endswith("turn_auc"):
            print(f"{key:14s} 波ごと", [val[k] for k in pick])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
