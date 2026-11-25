"""I48bm: 試行錯誤のループ（持ち主の案、未確定）: 違和感あり → どこに → その所の候補をいくつもの出どころから出す →
行全体の違和感でそろえる → 違和感チェック → もう 1 回 ／ 出力 ／ 解無し（仮の板に記録だけ）。

型は I48bi と同じ（UModel・型 C、5 万文で学ぶ。介入の単位は空）。試しの 1,000 文で、答えを使わない違和感だけで回す（教えない）。
違和感の線: 本体で試しの文を照らした時の、迷い・自信のなさ（選び）の上位 1 割の線。どちらかを越えた間を「違和感のある間」とする。
候補の出どころ（どれも詰め物のクラス）:
  表の上位   本体の今の点（数えの表）の上位
  借りる     本体 ＋ 数えの表 ＋ 借りる（I48be）の今の点の上位
  属性       R2（左右の語・述語の属性の板）の上位
  骨組み     学びの文で、(左の語の品詞, 右の語の品詞 か 文末) の間に来た詰め物の数えの上位
  合わせ     上の 4 つの和集合（各 K 個ずつ）
そろえ方: 候補を 1 つずつその間に置き、行の残りを置き直して（後ろの間の選びは R3 の並びで変わる）、行全体の違和感を読む。
  行の最小: 行の中でいちばん浮いている所（間ごとの 6 種類の違和感の、本体の並びの中での順位の割合の最大）の、行の中の最大
  行の和  : 間ごとの 6 種類の割合の平均の、行の中の和
  そろえで選ぶのは、違和感がいちばん小さくなる候補（元の選びも候補に入る）。
ループ: 1 回ごとに、違和感のある間のうち、いちばん強い間を 1 つ直す。出口:
  出力       違和感のある間が無くなった
  解無し     2 回続けて行の違和感が減らない ／ 同じ並びに戻った ／ 回数の上限（6 回）。解無しの間は仮の板に記録だけ（開けない）
測る物:
  (1) 候補に正解が入っている割合: 出どころ別・合わせ、候補の数 K = 1・3・5・10・20 ごとの曲線（違和感のある間で）
  (2) 1 回のそろえで正解を選べる割合（行の最小・行の和。元の選びの当たりと並べる）と、1 か所を直すと別の間の選びが動く割合
  (3) 何回で止まるか、出口ごとの割合、止め方ごとの数、何回目で違和感が消えた事が多いか、解無しの名札の内訳
  「違和感が消えた」と「正しくなった」を分けて数える（消えたが外れ、の割合）。
置き換えた間の違和感: 迷い・自信のなさは置いた候補自身の点の順位で決まり、置き換えると必ず強くなるので、置き換えた間だけは
  前後との合い方の種類（筋と波の食い違い・筋の無い共振・組の珍しさ。どれも試しの文での上位 1 割の線）で見る。
手で置く物: 違和感の線（上位 1 割）、K の並び、合わせの K（10）、回数の上限 6、止め方の 3 つ。
python intervention/past/i48bm_trial_loop.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
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
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48bc_usage_board as BC  # noqa: E402
import i48be_borrow as BE  # noqa: E402
import i48bi_odd_kinds as BI  # noqa: E402

LOOP_KINDS = ("迷い", "自信のなさ_選び")
KS = (1, 3, 5, 10, 20)
K_UNION = 10
MAX_ROUNDS = 6
SOURCES = ("表の上位", "借りる", "属性", "骨組み")


def line_rows(m, keys, xs, nat, force=None):
    """行の選びと、間ごとの違和感の値。force = {間: 詰め物} の所はその詰め物に置く。返り値に、間ごとの点（今・R2）も持つ。"""
    force = force or {}
    fic = m.who.fiction(keys)
    ups = {u for _, u in keys}
    placed, rows = [], []
    for x in xs:
        g = m.gap_scores(keys, x, fic)
        r3 = m.gram.score(m.gram.state(ups, placed, x["last"]))
        logic, wave = g["R2"] + r3, g["R1"]
        three = wave + logic
        both = g["今"] + three
        three[-1] = both[-1] = -1e9
        cand = (set(np.argsort(-g["今"])[:AK.K]) | set(np.argsort(-three)[:AK.K])) - {m.nc - 1}
        pick = force.get(x["i"], max(cand, key=lambda c: both[c]))
        top2 = np.sort(both)[-2:]
        p = np.exp(both - both.max())
        p /= p.sum()
        lg, wv = logic.copy(), wave.copy()
        lg[-1] = wv[-1] = -1e9
        pred = AJ.next_pred(keys, x["i"])
        rows.append(dict(i=int(x["i"]), gold=int(m.cid(x["y"])), pick=int(pick), now=g["今"], r2=g["R2"],
                         迷い=-float(top2[1] - top2[0]) if x["i"] not in force else -float(both[pick] - np.max(np.delete(both, pick))),
                         自信のなさ_選び=-float(p[pick]), 筋と波の食い違い=float(AJ.ranks(lg)[pick] + AJ.ranks(wv)[pick]),
                         筋の無い共振=float(lg[pick] <= np.median(lg[:-1]) and int(np.argmax(wv)) == pick),
                         組の珍しさ=-nat.score(m.pg.cls[pick], pred[0] if pred else None),
                         手持ちに無い=float(np.exp(g["今"][-1] - np.logaddexp.reduce(g["今"])))))
        placed.append(m.pg.cls[pick])
    return rows


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
    nat = RC.Natural(fill, tr)
    # 借りる型
    ub = BC.UsageBoard(m, dk=2048).fit(m, tr)
    rows_, nf = BE.usage_vectors(ub)
    nbrs = BE.neighbours(ub, rows_, nf)
    mb = m.clone()
    mb.extras.append(BE.BorrowTable(m).fit(m, tr, ub, nbrs))
    mb.refresh()
    # 骨組み: (左の品詞, 右の品詞 か 文末) → 詰め物の数え
    skel = defaultdict(Counter)
    for s in tr:
        cont, gaps, _ = G.split(s)
        for i, g in enumerate(gaps):
            skel[(cont[i][2], cont[i + 1][2] if i + 1 < len(cont) else "文末")][m.cid(g)] += 1
    rows_te = BI.rows_of(m, te, nat)
    ref = {kd: np.sort([r[kd] for r in rows_te]) for kd in BI.LABEL_KINDS}
    thr = {kd: float(np.quantile(ref[kd], 0.9)) for kd in BI.LABEL_KINDS}
    CTX = ("筋と波の食い違い", "筋の無い共振", "組の珍しさ")     # 置いた候補そのものの点の順位に依らない種類（置き換えた間はこれだけで見る）

    def pcts(r):
        return {kd: float(np.searchsorted(ref[kd], r[kd]) / len(ref[kd])) for kd in BI.LABEL_KINDS}

    def odd(r, forced=False):
        # 置き換えた間は、迷い・自信のなさ（自分の点の順位で決まる）では見ず、前後との合い方の種類だけで見る
        if forced:
            return any(r[kd] > thr[kd] for kd in CTX)
        return any(r[kd] > thr[kd] for kd in LOOP_KINDS)

    def strength(r):
        return max(pcts(r)[kd] for kd in LOOP_KINDS)

    def label(r):
        pc = pcts(r)
        return max(pc, key=pc.get)

    def line_odd(rows, forced=()):
        P = []
        for r in rows:
            pc = pcts(r)
            if r["i"] in forced:
                pc = {kd: pc[kd] for kd in CTX}
            P.append(pc)
        return max(max(p.values()) for p in P), float(sum(np.mean(list(p.values())) for p in P))
    res = dict(学びの文=len(tr), 試しの文=len(te), 線=thr, K=list(KS), 合わせのK=K_UNION, 回数の上限=MAX_ROUNDS)

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"用意 {time.time() - t0:.0f} 秒", flush=True)
    t1 = time.time()
    cover = {src: Counter() for src in SOURCES + ("合わせ",)}
    n_odd_gaps = 0
    pick1 = Counter()
    moved_other = n_one = 0
    exits, stops, fixed_round = Counter(), Counter(), Counter()
    rounds_hist = Counter()
    gone_right = gone_wrong = 0
    unsolved_lab = Counter()
    unsolved_wrong = 0
    final_hit = base_hit = n_gaps = 0
    karioki = []
    for si, s in enumerate(te):
        xs = AE.items(S, s, "節")
        if not xs:
            continue
        keys = AK.keys_of(s)
        force = {}
        rows = line_rows(m, keys, xs, nat)
        base_rows = rows
        start_odd = [r["i"] for r in rows if odd(r)]
        seen_states = {tuple(r["pick"] for r in rows)}
        best_line = line_odd(rows)
        no_gain = 0
        exit_kind = None
        rnd = 0
        first_round = True
        counted = set()
        while True:
            targets = [r for r in rows if odd(r, r["i"] in force)]   # 一度直した間も、まだ違和感があればもう一度選ばれる
            if not targets:
                exit_kind = "出力"
                break
            if rnd >= MAX_ROUNDS:
                exit_kind, stop = "解無し", "回数の上限"
                break
            rnd += 1
            t = max(targets, key=strength)
            i = t["i"]
            x = xs[i]
            # 候補を出す
            nowb = line_rows(mb, keys, xs, nat, force)[i]["now"]
            srcs = {"表の上位": [int(c) for c in np.argsort(-t["now"]) if c != m.nc - 1],
                    "借りる": [int(c) for c in np.argsort(-nowb) if c != m.nc - 1],
                    "属性": [int(c) for c in np.argsort(-t["r2"]) if c != m.nc - 1],
                    "骨組み": [c for c, _ in skel.get((keys[i][1], keys[i + 1][1] if i + 1 < len(keys) else "文末"), Counter()).most_common() if c != m.nc - 1]}
            if i not in counted:
                counted.add(i)
                n_odd_gaps += 1
                g = t["gold"]
                for src, lst in srcs.items():
                    for k in KS:
                        cover[src][k] += g in lst[:k]
                for k in KS:
                    u = set().union(*[set(l[:k]) for l in srcs.values()])
                    cover["合わせ"][k] += g in u
            cands = sorted(set().union(*[set(l[:K_UNION]) for l in srcs.values()]) | {t["pick"]})
            scored = []
            for c in cands:
                f2 = dict(force)
                f2[i] = c
                r2 = line_rows(m, keys, xs, nat, f2)
                scored.append((line_odd(r2, set(f2) if c != t["pick"] or i in force else set(force)), c, r2))
            best_min = min(scored, key=lambda z: (z[0][0], z[0][1]))
            best_sum = min(scored, key=lambda z: (z[0][1], z[0][0]))
            if first_round:
                n_one += 1
                pick1["元の選び"] += t["pick"] == t["gold"]
                pick1["行の最小"] += best_min[1] == t["gold"]
                pick1["行の和"] += best_sum[1] == t["gold"]
                moved_other += any(r["pick"] != q["pick"] for r, q in zip(best_min[2], rows) if r["i"] != i)
                first_round = False
            new_line, c, r_new = best_min
            force[i] = c
            state = tuple(r["pick"] for r in r_new)
            if new_line < best_line:
                best_line, no_gain = new_line, 0
            else:
                no_gain += 1
            rows = r_new
            if state in seen_states and c != t["pick"]:
                exit_kind, stop = "解無し", "同じ並びに戻った"
                break
            seen_states.add(state)
            if no_gain >= 2:
                exit_kind, stop = "解無し", "2 回続けて減らない"
                break
        exits[exit_kind] += 1
        rounds_hist[rnd] += 1
        for i in start_odd:                                   # 初めに違和感のあった間のうち、終わりに違和感が消えた間（正解か外れか）
            r = rows[i]
            if not odd(r, i in force):
                if r["pick"] == r["gold"]:
                    gone_right += 1
                else:
                    gone_wrong += 1
        if exit_kind == "出力":
            fixed_round[rnd] += 1
        else:
            stops[stop] += 1
            for r in rows:
                if odd(r, r["i"] in force):
                    unsolved_lab[label(r)] += 1
                    unsolved_wrong += r["pick"] != r["gold"]
                    karioki.append(dict(文=si, 間=r["i"], 選び=m.pg.cls[r["pick"]], 名札=label(r), 止め方=stop))
        final_hit += sum(r["pick"] == r["gold"] for r in rows)
        base_hit += sum(r["pick"] == r["gold"] for r in base_rows)
        n_gaps += len(rows)
        if (si + 1) % 200 == 0:
            print(si + 1, dict(exits), round(time.time() - t1), "秒", flush=True)
    res["1_候補に正解が入っている割合"] = dict(違和感のある間=n_odd_gaps, **{src: {str(k): round(cover[src][k] / max(1, n_odd_gaps), 4) for k in KS} for src in cover})
    res["2_一回のそろえ"] = dict(間=n_one, **{k: round(v / max(1, n_one), 4) for k, v in pick1.items()},
                             別の間の選びが動いた割合=round(moved_other / max(1, n_one), 4))
    sent_n = sum(exits.values())
    unsolved_n = sum(unsolved_lab.values())
    res["3_止まり方"] = dict(文=sent_n, 出口={k: dict(数=v, 割合=round(v / max(1, sent_n), 4)) for k, v in exits.items()}, 止め方=dict(stops),
                         回数の分布=dict(sorted(rounds_hist.items())), 出力になった回=dict(sorted(fixed_round.items())),
                         違和感が消えた間_正解=gone_right, 違和感が消えた間_外れ=gone_wrong,
                         消えたが外れの割合=round(gone_wrong / max(1, gone_right + gone_wrong), 4),
                         解無しの間=unsolved_n, 解無しの名札=dict(unsolved_lab), 解無しの間の外れの割合=round(unsolved_wrong / max(1, unsolved_n), 4))
    res["全体の当たり"] = dict(間=n_gaps, 元=round(base_hit / max(1, n_gaps), 4), ループの後=round(final_hit / max(1, n_gaps), 4))
    res["仮の板_件数"] = len(karioki)
    res["秒"] = round(time.time() - t0)
    print(json.dumps({k: v for k, v in res.items() if k not in ("線",)}, ensure_ascii=False)[:3000], flush=True)
    save()
    json.dump(karioki, open(a.out.replace(".json", "_karioki.json"), "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
