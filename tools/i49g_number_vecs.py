"""I49g: 本物の語の波の板で、「数」の向きは近い数ほど似ているか。数えだけ（持ち主の案「A＋B＝C」の下調べ、未確定）。

1. 語の波の板から、0〜18 に当たる語の向きを引く。書き方 3 通り: 半角（"3"）、全角（"３"）、漢数字（"三"、"十二"）。
   板の語は名前で引く（板の語の表は名前 → 行。品詞は板が持っていないので、名前があれば使う）。書き方ごとに板に有る数を出す。
2. 板に有る数どうしで、近さ |cos| と、表裏を残した近さ（Re<a, b> / |a||b|）を、数の差ごとに平均する。比べ: 数でない名詞との近さの平均。
3. 向きの数え: 「n+1 の向き − n の向き」どうしの、表裏を残した近さの平均（n によらず同じ向きを指すなら 1 に近い）。
4. tools/i49g_add_count.py（中身は変えない）の holdout に、数 → 板の向き（半角の書き方。無い数は None）を渡し、見せていない 20 問の当たりと外れの大きさを出す。
   比べ: 数え歌・向きなし。
手で置く物: 0〜18、漢数字の書き方（十一〜十八）、数でない名詞の比べの語（NOUNS）。
python tools/i49g_number_vecs.py --wave <波の板> --out <stats.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

TOP = 19
KANJI = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
         "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八"]
NOUNS = ["猫", "山", "水", "家", "空", "道", "手", "花", "本", "人", "木", "川", "雨", "朝", "町", "海", "犬", "目", "声", "火"]


def spell(n, how):
    if how == "半角":
        return str(n)
    if how == "全角":
        return "".join(chr(ord(c) - ord("0") + ord("０")) for c in str(n))
    return KANJI[n]


def row_of(wv, name):
    i = wv.ix.get(name)
    return None if i is None else wv.W[i]


def sims(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    if d == 0:
        return 0.0, 0.0
    z = np.vdot(a, b) / d
    return float(abs(z)), float(z.real)


def by_gap(vecs):
    """vecs: 数 → 向き。差ごとの |cos| と 表裏を残した近さ の平均。"""
    out = {}
    ns = sorted(vecs)
    for g in range(1, TOP):
        ps = [(a, a + g) for a in ns if a + g in vecs]
        if not ps:
            continue
        s = [sims(vecs[a], vecs[b]) for a, b in ps]
        out[str(g)] = dict(組=len(ps), 近さ=round(float(np.mean([x for x, _ in s])), 4), 表裏を残す=round(float(np.mean([y for _, y in s])), 4))
    return out


def step_dirs(vecs):
    """「n+1 − n」どうしの、表裏を残した近さの平均と、組の数。"""
    d = [vecs[n + 1] - vecs[n] for n in sorted(vecs) if n + 1 in vecs]
    s = [sims(d[i], d[j])[1] for i in range(len(d)) for j in range(i + 1, len(d))]
    return dict(差の向き=len(d), 組=len(s), 表裏を残した近さの平均=round(float(np.mean(s)), 4) if s else None)


def main():
    import i49c_read_words as Rw
    import i49g_add_count as AC
    ap = argparse.ArgumentParser()
    for k in ("wave", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    wv = Rw.WordVecs.load(a.wave)
    res = dict(書き方={})
    nouns = [row_of(wv, w) for w in NOUNS if row_of(wv, w) is not None]
    for how in ("半角", "全角", "漢数字"):
        vecs = {n: row_of(wv, spell(n, how)) for n in range(TOP)}
        vecs = {n: v for n, v in vecs.items() if v is not None}
        base = [sims(v, u) for v in vecs.values() for u in nouns]
        res["書き方"][how] = dict(板に有る数=sorted(vecs), 数=len(vecs), 差ごと=by_gap(vecs), 差の向き=step_dirs(vecs),
                               数でない名詞との近さ=dict(組=len(base), 近さ=round(float(np.mean([x for x, _ in base])), 4) if base else None,
                                              表裏を残す=round(float(np.mean([y for _, y in base])), 4) if base else None))
    half = {str(n): row_of(wv, spell(n, "半角")) for n in range(TOP)}
    board = lambda w: half.get(w)
    res["見せていない20問"] = {"板の向き（半角）": AC.holdout(board), "数え歌": AC.holdout(AC.VECS["数え歌"]), "向きなし": AC.holdout(AC.VECS["なし"])}
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
