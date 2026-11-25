"""周波数の語板（tools/past/i48_freq_board.py）の続き: 帯の低い側で全体の向き、高い側で前後を読む（持ち主の案、未確定）。

帯の中の周波数を 2 つに分ける（W = 1、ν_j = j / 100）:
  低い側 ν < 0.1（j < 10）: 行の中の距離が数歩なら位相がほとんど回らない → 位置によらず「行全体の向き（話題）」を見る
  高い側（残り）: 位置で速く回る → すぐ前後のつながりを見る
覚え方 2 通り: 全部の組を同じ強さ（今まで） ／ 高い側は距離 2 歩以内の組だけ覚える（離れた組は全体の向きだけに入れる）
読み方:
  そのまま: 全部の周波数の共振で 1 位
  残す → ふるう → 決める: 共振がノイズの床（候補の平均 ＋ 1 標準偏差）より上の候補を残し、その中で低い側の共振が上半分の物に絞り、
    高い側の共振で 1 位を決める（ピークを取らず、かすかな候補を残す）
行の波（持ち主の案、未確定）: 行全体に、前から後ろへ周波数が上がる波（チャープ）を足す。位置 x = p / (行の長さ − 1) に、
  行の帯（100 個の周波数）の 10 + 80x 番目を中心とする幅 5 の山を置く。語ごとに「自分が行のどこで鳴ったか」を行の帯の箱に足し、
  読む時は候補の行の帯の箱と、その位置の行の波との重なり × γ を共振に足す（γ = 0・0.5・1・2）。
  あわせて、語順を逆にした行との見分け（行の平均の z の AUC）も出す。
合わない所の見つけ方: 位置ごとに、本当の語の「低い側の z − 高い側の z」（全体の向きでは合うのに、前後で合わない度合い）。
  行の中の最大で、おかしな文と大体正しい文を見分ける AUC。今までの見分け（全部の共振の z の行の平均・最小）と並べる。
手で置く物: 行の波の山の中心（10 + 80x）と幅 5、γ の並び、低い側の境 ν 0.1、近い組の距離 2、床の 1 標準偏差、ふるいの上半分、W 1。
python tools/past/i48_freq_board3.py --content .../content_boards.json --out out.json
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48_freq_board as FB  # noqa: E402

NB, NC = len(FB.BANDS), FB.NC
LOW_NU = 0.1
NEAR = 2
GAMMAS = (0.0, 0.5, 1.0, 2.0)


def chirp(x):
    j = np.arange(NC)
    c = np.exp(-0.5 * ((j - (10 + 80 * x)) / 5.0) ** 2)
    return c / np.linalg.norm(c)


def learn_line(bd, lines):
    RL = np.zeros((len(bd.S), NC))
    for ln in lines:
        if len(ln) < 2:
            continue
        for p, w in enumerate(ln):
            RL[bd.li[w]] += chirp(p / (len(ln) - 1))
    return RL


def auc(pos, neg):
    a = np.array(list(neg) + list(pos))
    r = a.argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def learn(bd, lines, near_only_high):
    low = bd.nu < LOW_NU
    R = np.zeros_like(bd.S)
    for ln in lines:
        ids = [bd.li[w] for w in ln]
        for p, i in enumerate(ids):
            for q, k in enumerate(ids):
                if q == p:
                    continue
                add = bd.S[k] * bd.ph(q - p)
                if near_only_high and abs(q - p) > NEAR:
                    add = add * low
                R[i] += add
    return R


def parts(bd, R, ids, p, RL=None, gamma=0.0):
    """位置 p の候補ごとの (全部, 低い側, 高い側) の共振。行の波があれば、候補の行の帯の箱とその位置の行の波の重なり × γ を全部と高い側に足す。"""
    low = bd.nu < LOW_NU
    Q = np.zeros(bd.S.shape[1], dtype=np.complex128)
    for pi, i in enumerate(ids):
        if pi != p:
            Q += np.conj(R[i]) * bd.ph(p - pi)
    Z = np.real(bd.S * Q[None, :])
    lo, hi = Z[:, low].sum(1), Z[:, ~low].sum(1)
    if RL is not None and gamma > 0:
        b = gamma * (RL @ chirp(p / (len(ids) - 1)))
        hi = hi + b
    return lo + hi, lo, hi


def z(x):
    return (x - x.mean()) / max(1e-12, x.std())


def pick(tot, lo, hi):
    keep = np.where(tot > tot.mean() + tot.std())[0]
    if len(keep) == 0:
        return int(np.argmax(tot))
    lk = lo[keep]
    keep = keep[lk >= np.median(lk)]
    return int(keep[np.argmax(hi[keep])])


def evaluate(bd, R, lines, RL=None, gamma=0.0):
    h = h5 = hp = n = 0
    rows = []
    for ln in lines:
        ids = [bd.li[w] for w in ln]
        if len(ids) < 2:
            continue
        zt, gap = [], []
        for p, t in enumerate(ids):
            tot, lo, hi = parts(bd, R, ids, p, RL, gamma)
            h += int(np.argmax(tot) == t)
            h5 += int((tot > tot[t]).sum() < 5)
            hp += int(pick(tot, lo, hi) == t)
            n += 1
            zt.append(z(tot)[t])
            gap.append(z(lo)[t] - z(hi)[t])
        rows.append(dict(平均=float(np.mean(zt)), 最小=float(np.min(zt)), 低高の差の最大=float(np.max(gap))))
    return dict(そのまま_一位=round(h / n, 4), そのまま_上位5=round(h5 / n, 4), 残すふるう決める_一位=round(hp / n, 4)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    cb_all = {}
    for w in json.load(open(a.content, encoding="utf-8")):
        if w["name"] not in cb_all or w.get("freq", 0) > cb_all[w["name"]].get("freq", 0):
            cb_all[w["name"]] = w
    train = [s.split() for _, s in FB.LINES]
    ok = [s.split() for _, s in FB.UNSEEN_OK]
    bad = [s.split() for _, s in FB.UNSEEN_BAD]
    lemmas = sorted({w for s in train for w in s})
    bd = FB.Board(cb_all, lemmas, "乱数", None, 1.0)
    res = dict(語=len(lemmas), 学習の文=len(train), 低い側の境=LOW_NU, 近い組の距離=NEAR, 型={})
    RL = learn_line(bd, train)
    for (nm, near), gamma in [(x, g) for x in (("全部の組を同じ強さ", False), ("高い側は近い組だけ", True)) for g in GAMMAS]:
        R = learn(bd, train, near)
        r = {}
        for tag, L in (("学習した文", train), ("学習に無い大体正しい文", ok), ("おかしな文", bad)):
            r[tag], rows = evaluate(bd, R, L, RL, gamma)
            r[tag + "_行"] = rows
        rev = {}
        for tag, L in (("学習した文", [s for s in train if len(s) >= 2]), ("学習に無い大体正しい文", ok)):
            _, ro_ = evaluate(bd, R, L, RL, gamma)
            _, rr_ = evaluate(bd, R, [s[::-1] for s in L], RL, gamma)
            rev[tag] = round(auc([x["平均"] for x in ro_], [x["平均"] for x in rr_]), 4)
        r["語順を逆にした行を見分けるAUC"] = rev
        ro, rb = r["学習に無い大体正しい文_行"], r["おかしな文_行"]
        r["おかしな文を見分けるAUC"] = dict(
            全部の共振の平均=round(auc([x["平均"] for x in ro], [x["平均"] for x in rb]), 4),
            全部の共振の最小=round(auc([x["最小"] for x in ro], [x["最小"] for x in rb]), 4),
            低い側で合うのに高い側で合わない度合いの最大=round(auc([-x["低高の差の最大"] for x in ro], [-x["低高の差の最大"] for x in rb]), 4))
        for tag in ("学習した文", "学習に無い大体正しい文", "おかしな文"):
            del r[tag + "_行"]
        nm = f"{nm}・行の波 γ {gamma}"
        res["型"][nm] = r
        print(nm, json.dumps(r, ensure_ascii=False), flush=True)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
