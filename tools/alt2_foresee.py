"""alt2: 外れそうだと、先に分かるか（別の助言からの試し。数えだけ）。

助言の「確信度の校正」を、器の言葉に直した問い: 器は聞く前に、次の入力がどのくらい外れるかの見込みを持てているか。
器（I49a の Vessel）の記録（dump の npz: src・act・gain・sur・half）だけから数える。器は動かさない。本文は要らない。
聞く前に器の中にある量（波ごと）:
  いつもの外れ  記録の外れから器と同じ式で作り直す（usual += a × (外れ − usual)、初めは 1）
  直前の外れ    1 つ前に聞いた時の外れ
  感度・大きさ  1 つ前の刻みの値
数える物（入力を聞いた刻みだけ、初めの skip 回は除く、波ごと）:
  r      その量と、実際の外れの相関
  ずれ    |いつもの外れ − 実際の外れ| の平均（比べ: |直前の外れ − 実際|、|全部の平均 − 実際|。全部の平均は後知恵の物差し）
  段      いつもの外れで 5 段に分けた時の、段ごとの実際の外れの平均（見込みが高い段ほど実際も高ければ、合っている）
手で置いた値: skip（初めに除く回数、既定 50）、段の数 5。

python tools/alt2_foresee.py --npz <器の記録.npz> --out <stats.json>
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return round(float((x * y).sum() / d), 4) if d > 0 else None


def measure(z, skip=50, bins=5):
    src, sur, half = z["src"], z["sur"].astype(np.float64), z["half"].astype(np.float64)
    a = 1.0 - np.power(0.5, 1.0 / half)
    T, N = sur.shape
    usual = np.ones(N)
    U, L = np.zeros((T, N)), np.zeros((T, N))                    # 聞く前のいつもの外れ・直前の外れ
    last = np.ones(N)
    for t in range(T):
        U[t], L[t] = usual, last
        if src[t] != 0:
            usual = usual + a * (sur[t] - usual)
            last = sur[t]
    heard = np.flatnonzero(src != 0)[skip:]
    heard = heard[heard > 0]
    y = sur[heard]
    pre = dict(いつもの外れ=U[heard], 直前の外れ=L[heard], 感度=np.log2(z["gain"][heard - 1].astype(np.float64)),
               大きさ=z["act"][heard - 1].astype(np.float64))
    res = dict(半減期=[round(float(h), 1) for h in half], 聞いた回数=int(len(heard)), 外れの平均=[round(float(v), 4) for v in y.mean(0)],
               r={k: [corr(v[:, n], y[:, n]) for n in range(N)] for k, v in pre.items()})
    res["ずれ"] = dict(いつもの外れ=[round(float(v), 4) for v in np.abs(U[heard] - y).mean(0)],
                     直前の外れ=[round(float(v), 4) for v in np.abs(L[heard] - y).mean(0)],
                     全部の平均=[round(float(v), 4) for v in np.abs(y.mean(0)[None] - y).mean(0)])
    steps = []
    for n in range(N):
        order = np.argsort(U[heard][:, n], kind="stable")
        steps.append([round(float(y[idx, n].mean()), 4) for idx in np.array_split(order, bins)])
    res["段"] = steps
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", type=int, default=50)
    a = ap.parse_args()
    res = measure(np.load(a.npz), a.skip)
    pick = [0, 4, 8, 12, 16, len(res["半減期"]) - 1]
    print("半減期", [res["半減期"][k] for k in pick], "/ 聞いた回数", res["聞いた回数"])
    for k, v in res["r"].items():
        print(f"r {k}", [v[i] for i in pick])
    for k, v in res["ずれ"].items():
        print(f"ずれ {k}", [v[i] for i in pick])
    for i in pick:
        print("段", res["半減期"][i], res["段"][i])
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
