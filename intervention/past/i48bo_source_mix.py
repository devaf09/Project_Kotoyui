"""I48bo: 外れの行ほど、間ごとの選びの「出どころ」が散らばっているか（持ち主の見立て「いくつか提案する下地があって、それが絡まっている」、数えだけ）。

型は I48bi と同じ（UModel・型 C、5 万文で学ぶ、介入の単位は空）。試しの 1,000 文で、間ごとの選び（今 ＋ R1 ＋ R2 ＋ R3 の和のいちばん）について:
  出どころの点  今（数えの表）・R1（全体の波）・R2（属性の板）・R3（文法の並び）の 4 つ。
  引っ張り役    選んだ候補と 2 位の候補の点の差を 4 つの出どころに分け、出どころごとに候補の中の点の広がり（標準偏差）で割った時、
                いちばん大きく選んだ候補の側に引いた出どころ（割らないと、点の幅の大きい「今」がいつも引っ張り役になる）。
  ひとりの選び  出どころごとに、その出どころだけで選んだ候補（その他は伏せる、候補は本体と同じ組）。
  食い違い      4 つのひとりの選びが、いくつの別の候補に分かれたか（1〜4）。
行ごとに: 引っ張り役の種類の数・散らばり（エントロピー）、間ごとの食い違いの平均、隣の間と引っ張り役が替わる割合。
比べ: 全部当たった行 ／ 1 つ以上外れた行（行の長さ＝間の数ごとにそろえて比べる）、間ごとに 当たり ／ 外れ。
  AUC は「外れの方が大きい」確からしさ。
注意: 出どころは足し算の項の分け方で、「言いたいこと（意図）」そのものではない。意図の混ざりを直に見る物ではなく、その手前の下調べ。
手で置く物: 出どころの 4 つの分け方、広がりで割る事、間の数のそろえ方（2・3・4・5 以上）。
python intervention/past/i48bo_source_mix.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ap_units as AP  # noqa: E402

SRC = ("今", "R1", "R2", "R3")


def auc(pos, neg):
    if not pos or not neg:
        return None
    a = np.array(list(neg) + list(pos), dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=a.argsort().argsort() + 1.0) / cnt)[inv]
    return round(float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def entropy(c):
    p = np.array(list(c.values()), dtype=float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def line_of(m, s):
    xs = AE.items(m.S, s, "節")
    if not xs:
        return None
    keys = AK.keys_of(s)
    fic = m.who.fiction(keys)
    ups = {u for _, u in keys}
    placed, gaps = [], []
    for x in xs:
        g = m.gap_scores(keys, x, fic)
        r3 = m.gram.score(m.gram.state(ups, placed, x["last"]))
        comp = {"今": g["今"].copy(), "R1": g["R1"].copy(), "R2": g["R2"].copy(), "R3": np.asarray(r3, dtype=float).copy()}
        three = g["R1"] + g["R2"] + r3
        both = g["今"] + three
        three[-1] = both[-1] = -1e9
        cand = (set(np.argsort(-g["今"])[:AK.K]) | set(np.argsort(-three)[:AK.K])) - {m.nc - 1}
        order = sorted(cand, key=lambda c: -both[c])
        pick = order[0]
        second = order[1] if len(order) > 1 else order[0]
        cl = sorted(cand)
        diff = {k: float(v[pick] - v[second]) / (float(np.std(v[cl])) + 1e-9) for k, v in comp.items()}   # 出どころごとの点の広がりで割る
        lead = max(diff, key=diff.get)
        solo = {k: max(cand, key=lambda c: v[c]) for k, v in comp.items()}
        gaps.append(dict(ok=pick == m.cid(x["y"]), lead=lead, split=len(set(solo.values())),
                         solo_hit={k: c == pick for k, c in solo.items()}))
        placed.append(m.pg.cls[pick])
    return gaps


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    for s in tr:
        fill.see(s)
    m = AP.UModel(S, fill, a.r1).fit(tr, hold).init_units("C")
    m.refresh()
    print(f"学び {time.time() - t0:.0f} 秒", flush=True)
    lines = [g for g in (line_of(m, s) for s in te) if g]
    # 間ごと
    G_all = [x for g in lines for x in g]
    per_gap = dict(間=len(G_all), 当たり=round(float(np.mean([x["ok"] for x in G_all])), 4),
                   引っ張り役の内訳={k: dict(数=sum(x["lead"] == k for x in G_all),
                                         当たり=round(float(np.mean([x["ok"] for x in G_all if x["lead"] == k])), 4) if any(x["lead"] == k for x in G_all) else None)
                                for k in SRC},
                   食い違いの数ごとの当たり={str(n): dict(数=sum(x["split"] == n for x in G_all),
                                                 当たり=round(float(np.mean([x["ok"] for x in G_all if x["split"] == n])), 4))
                                          for n in sorted({x["split"] for x in G_all})},
                   食い違いが外れで大きいAUC=auc([x["split"] for x in G_all if not x["ok"]], [x["split"] for x in G_all if x["ok"]]),
                   ひとりの選びが本体と同じ割合={k: round(float(np.mean([x["solo_hit"][k] for x in G_all])), 4) for k in SRC})
    # 隣の間と引っ張り役が替わる時の外れ
    sw_ok, st_ok = [], []
    for g in lines:
        for a_, b_ in zip(g, g[1:]):
            (sw_ok if a_["lead"] != b_["lead"] else st_ok).append(b_["ok"])
    per_gap["隣と引っ張り役が替わった間の当たり"] = round(float(np.mean(sw_ok)), 4) if sw_ok else None
    per_gap["隣と同じ引っ張り役の間の当たり"] = round(float(np.mean(st_ok)), 4) if st_ok else None
    # 行ごと（間の数でそろえる）
    per_line = {}
    for band in (2, 3, 4, 5):
        L = [g for g in lines if (len(g) == band if band < 5 else len(g) >= 5)]
        bad = [g for g in L if not all(x["ok"] for x in g)]
        good = [g for g in L if all(x["ok"] for x in g)]
        def feats(g):
            c = Counter(x["lead"] for x in g)
            return dict(種類=len(c), 散らばり=entropy(c), 食い違い=float(np.mean([x["split"] for x in g])),
                        替わる割合=float(np.mean([a_["lead"] != b_["lead"] for a_, b_ in zip(g, g[1:])])))
        fb, fg = [feats(g) for g in bad], [feats(g) for g in good]
        per_line[f"間 {band}" + (" 以上" if band == 5 else "")] = dict(
            行=len(L), 外れの行=len(bad), 全部当たりの行=len(good),
            **{f"{k}_外れの行": round(float(np.mean([f[k] for f in fb])), 4) if fb else None for k in ("種類", "散らばり", "食い違い", "替わる割合")},
            **{f"{k}_当たりの行": round(float(np.mean([f[k] for f in fg])), 4) if fg else None for k in ("種類", "散らばり", "食い違い", "替わる割合")},
            **{f"{k}_AUC": auc([f[k] for f in fb], [f[k] for f in fg]) for k in ("散らばり", "食い違い", "替わる割合")})
    res = dict(学びの文=len(tr), 試しの文=len(te), 行=len(lines), 間ごと=per_gap, 行ごと=per_line, 秒=round(time.time() - t0))
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False)[:3000], flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
