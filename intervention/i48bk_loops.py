"""I48bk: 違和感の名札で入るループを決めて、違和感が消えるかを数える（持ち主の案「違和感が無くなるまで調整を自分でする」、未確定）。

型は I48bi と同じ（UModel・型 C、5 万文で学ぶ）。名札は I48bi の答えを使わない 6 種類の「いちばん強い種類」。
ループ:
  1. 教える（介入の板。1 文 = 1 単位、本体は書き換えない、記録と取り消しあり）: 学んだ文の二回り目で、名札の付いた間を強い順に選び、
     その文を単位として教える（正解の詰め物を使う＝人や先生が教えた事に当たる）。N = 100・1,000 文。
     選び方: 迷い・自信のなさの名札 ／ 乱数 ／ ほかの名札（筋と波・組の珍しさ・手持ちに無い） ／ I48at の和（正解の確からしさを含む）の上位。
  2. 回す: 迷い・自信のなさの名札で N 1,000 文ずつ 3 回。回ごとに、教えた後の型で名札を付け直し、まだ教えていない文の中から選ぶ。
  3. 試して比べる（答えを使わない、自分だけの調整）: 試しの文で、名札が迷いの間を、2 位の候補に替えて行の残りを置き直し、
     行全体の自信のなさの和（1 − 選んだ候補の確からしさ、の和）が小さい方を採る。
  4. 借りる: 本体 ＋ 数えの表 ＋ 借りる（I48be）を、自信のなさの名札の間に効かせた時の、その間の外れと自信のなさ。
測る物: 教えた所そのものの迷い・自信のなさと外れ（教える前と後）、試しの 1,000 文の 間・文末・文・助詞ごとの害、
  種類ごとの違和感の量（教える前の型の上位 1 割の線を越えた間の割合）、回すごとの名札の内訳、
  どのループでも下がらずに残った間の数・名札の内訳・外れのうち「言える」（I48ax の物差し）割合。
手で置く物: N の並び、回す数 3、二回り目の文の数 5,000、上位 1 割の線、乱数の種。
python intervention/i48bk_loops.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000] [--recheck 5000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48ax_sayable as AX  # noqa: E402
import i48bc_usage_board as BC  # noqa: E402
import i48be_borrow as BE  # noqa: E402
import i48bi_odd_kinds as BI  # noqa: E402

NS = tuple(int(x) for x in os.environ.get("BK_NS", "100,1000").split(","))   # 初めは 100・300・1,000・3,000。単位が増えるほど重いので絞った（BK_NS で渡す）
ROUNDS = 3
N_ROUND = int(os.environ.get("BK_N_ROUND", 1000))
LOOP_KINDS = ("迷い", "自信のなさ_選び")
OTHER_KINDS = ("筋と波の食い違い", "組の珍しさ", "手持ちに無い")


def labels(rows):
    P = {kd: BI.pct(np.array([r[kd] for r in rows])) for kd in BI.LABEL_KINDS}
    M = np.stack([P[kd] for kd in BI.LABEL_KINDS], axis=1)
    return np.array(BI.LABEL_KINDS)[np.argmax(M, axis=1)], M.max(1)


def sent_order(rows, lab, st, kinds, exclude=()):
    """名札が kinds の間を強い順に並べ、文の番号（重なりなし）の並びにする。"""
    idx = [i for i in np.argsort(-st, kind="stable") if lab[i] in kinds]
    out, seen = [], set(exclude)
    for i in idx:
        j = rows[i]["j"]
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


def amounts(rows, thr):
    return {kd: round(float(np.mean([r[kd] > thr[kd] for r in rows])), 4) for kd in thr}


def walk_scores(m, keys, xs, force=None):
    """本体の選び（今 ＋ R1 ＋ R2 ＋ R3、その他は伏せる）。force = (間, 候補) でその間だけ選びを替える。返り値: 選びと、選びの確からしさ・1 位と 2 位。"""
    fic = m.who.fiction(keys)
    ups = {u for _, u in keys}
    placed, out = [], []
    for x in xs:
        g = m.gap_scores(keys, x, fic)
        r3 = m.gram.score(m.gram.state(ups, placed, x["last"]))
        three = g["R1"] + g["R2"] + r3
        both = g["今"] + three
        three[-1] = both[-1] = -1e9
        cand = (set(np.argsort(-g["今"])[:AK.K]) | set(np.argsort(-three)[:AK.K])) - {m.nc - 1}
        order = sorted(cand, key=lambda c: -both[c])
        pick = force[1] if force is not None and force[0] == x["i"] else order[0]
        p = np.exp(both - both.max())
        p /= p.sum()
        out.append(dict(pick=int(pick), p=float(p[pick]), second=int(order[1]) if len(order) > 1 else int(order[0])))
        placed.append(m.pg.cls[pick])
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--recheck", type=int, default=5000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    ref = tr_all[:-2000][a.train:]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill, surf = AD.Fill(S), G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    m = AP.UModel(S, fill, a.r1).fit(tr, hold).init_units("C")
    m.refresh()
    fill_ref = AD.Fill(S)
    for s in ref:
        fill_ref.see(s)
    say = AX.Sayable(S, ref)
    nat = RC.Natural(fill, tr)
    rc = tr[: a.recheck]
    res = dict(学びの文=len(tr), 試しの文=len(te), 二回り目の文=len(rc), N=list(NS), 回す数=ROUNDS, 回すごとの文=N_ROUND, 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rows_rc = BI.rows_of(m, rc, nat)
    rows_te = BI.rows_of(m, te, nat)
    thr = {kd: float(np.quantile([r[kd] for r in rows_te], 0.9)) for kd in BI.LABEL_KINDS}
    lab_rc, st_rc = labels(rows_rc)
    lab_te, st_te = labels(rows_te)
    wrong_te0 = np.array([r["pick"] != r["gold"] for r in rows_te])

    def measure(mm, nm, extra=None):
        t1 = time.time()
        r = AO.measure(mm, te, surf)
        rt = BI.rows_of(mm, te, nat)
        r["違和感の量_試し"] = amounts(rt, thr)
        r["外れの割合_試し"] = round(float(np.mean([x["pick"] != x["gold"] for x in rt])), 4)
        if extra:
            r.update(extra)
        r["秒"] = round(time.time() - t1)
        res["型"][nm] = r
        print(nm, r["秒"], "秒", json.dumps({k: r[k] for k in ("間", "文末", "文", "違和感の量_試し")}, ensure_ascii=False), flush=True)
        save()
        return rt
    res["土台"] = dict(違和感の量_試し=amounts(rows_te, thr), 外れの割合_試し=round(float(wrong_te0.mean()), 4),
                     名札_二回り目=dict(Counter(lab_rc.tolist())), 名札_試し=dict(Counter(lab_te.tolist())), 線=thr)
    base_rt = measure(m, "本体")
    print(f"用意 {time.time() - t0:.0f} 秒", flush=True)

    def teach(mm, js, tag):
        for j in js:
            s = rc[j]
            xs = AE.items(S, s, "節")
            if xs:
                mm.teach_unit((tag, tag, AK.keys_of(s), G.split(s)[1]), xs=xs)
        mm.refresh()

    def taught_effect(mm, js, kinds):
        """教えた文の、名札が kinds だった間の、教える前と後の値・外れ。"""
        sub = [rc[j] for j in js]
        after = BI.rows_of(mm, sub, nat)
        pos = {j: k for k, j in enumerate(js)}
        before = [(r, lab_rc[i]) for i, r in enumerate(rows_rc) if r["j"] in pos and lab_rc[i] in kinds]
        amap = {(pos[r0["j"]], r0["i"]): None for r0, _ in before}
        for r in after:
            if (r["j"], r["i"]) in amap:
                amap[(r["j"], r["i"])] = r
        pairs = [(r0, amap[(pos[r0["j"]], r0["i"])]) for r0, _ in before if amap[(pos[r0["j"]], r0["i"])] is not None]
        if not pairs:
            return {}
        return dict(間=len(pairs),
                    **{f"{kd}_前": round(float(np.mean([p[0][kd] for p in pairs])), 4) for kd in LOOP_KINDS},
                    **{f"{kd}_後": round(float(np.mean([p[1][kd] for p in pairs])), 4) for kd in LOOP_KINDS},
                    外れ_前=round(float(np.mean([p[0]["pick"] != p[0]["gold"] for p in pairs])), 4),
                    外れ_後=round(float(np.mean([p[1]["pick"] != p[1]["gold"] for p in pairs])), 4),
                    線を越えたまま=round(float(np.mean([any(p[1][kd] > thr[kd] for kd in LOOP_KINDS) for p in pairs])), 4))
    # 1. 選び方 × N
    rng = np.random.default_rng(3)
    all_js = sorted({r["j"] for r in rows_rc})
    rs = sum(np.argsort(np.argsort(v, kind="stable"), kind="stable")
             for v in (np.array([-r["迷い"] for r in rows_rc]), np.array([-r["自信のなさ_正解"] for r in rows_rc]),
                       -np.array([r["筋と波の食い違い"] for r in rows_rc]), -np.array([r["筋の無い共振"] for r in rows_rc]),
                       np.array([-r["組の珍しさ"] for r in rows_rc])))
    at_order, seen = [], set()
    for i in np.argsort(rs, kind="stable"):
        if rows_rc[i]["j"] not in seen:
            seen.add(rows_rc[i]["j"])
            at_order.append(rows_rc[i]["j"])
    sels = {"迷い・自信の名札": sent_order(rows_rc, lab_rc, st_rc, LOOP_KINDS), "乱数": list(rng.permutation(all_js)),
            "ほかの名札": sent_order(rows_rc, lab_rc, st_rc, OTHER_KINDS), "I48atの和": at_order}
    res["選べる文の数"] = {k: len(v) for k, v in sels.items()}
    for nm, js in sels.items():
        for n in NS:
            if n > len(js):
                continue
            mm = m.clone()
            mm.kind = "C"
            teach(mm, js[:n], nm)
            eff = taught_effect(mm, js[:n], LOOP_KINDS)
            measure(mm, f"1_教える・{nm}・N {n}", dict(教えた所=eff))
            del mm
    # 2. 回す
    mm = m.clone()
    mm.kind = "C"
    done = set()
    rows_now, lab_now, st_now = rows_rc, lab_rc, st_rc
    loop_log = []
    for k in range(ROUNDS):
        js = sent_order(rows_now, lab_now, st_now, LOOP_KINDS, exclude=done)[:N_ROUND]
        teach(mm, js, f"回す {k + 1}")
        done |= set(js)
        rows_now = BI.rows_of(mm, rc, nat)
        lab_now, st_now = labels(rows_now)
        loop_log.append(dict(回=k + 1, 教えた文=len(js), 違和感の量_二回り目=amounts(rows_now, thr),
                             外れの割合_二回り目=round(float(np.mean([r["pick"] != r["gold"] for r in rows_now])), 4),
                             名札_二回り目=dict(Counter(lab_now.tolist()))))
        rt_loop = measure(mm, f"2_回す・{k + 1} 回目", dict(回す=loop_log[-1]))
    res["2_回す"] = loop_log
    # 3. 試して比べる（試しの文、本体、答えを使わない）
    t1 = time.time()
    changed = better = worse = n_tried = 0
    hit0 = hit1 = n_all = 0
    k_rows = 0
    self_picks = {}
    for s in te:
        xs = AE.items(S, s, "節")
        if not xs:
            continue
        keys = AK.keys_of(s)
        w0 = walk_scores(m, keys, xs)
        golds = [m.cid(x["y"]) for x in xs]
        best = w0
        for x, w in zip(xs, w0):
            if lab_te[k_rows + x["i"]] != "迷い":
                continue
            n_tried += 1
            w1 = walk_scores(m, keys, xs, force=(x["i"], w["second"]))
            if sum(1 - y["p"] for y in w1) < sum(1 - y["p"] for y in best):
                best = w1
        if best is not w0:
            changed += 1
            a0 = sum(y["pick"] == g for y, g in zip(w0, golds))
            a1 = sum(y["pick"] == g for y, g in zip(best, golds))
            better += a1 > a0
            worse += a1 < a0
        hit0 += sum(y["pick"] == g for y, g in zip(w0, golds))
        hit1 += sum(y["pick"] == g for y, g in zip(best, golds))
        n_all += len(xs)
        for x, y in zip(xs, best):
            self_picks[(k_rows + x["i"])] = y["pick"]
        k_rows += len(xs)
    res["3_試して比べる"] = dict(試した間=n_tried, 選びを替えた文=changed, 良くなった文=better, 悪くなった文=worse,
                            当たり_前=round(hit0 / max(1, n_all), 4), 当たり_後=round(hit1 / max(1, n_all), 4), 秒=round(time.time() - t1))
    print("3", json.dumps(res["3_試して比べる"], ensure_ascii=False), flush=True)
    save()
    # 4. 借りる
    ub = BC.UsageBoard(m, dk=2048).fit(m, tr)
    rows_, nf = BE.usage_vectors(ub)
    nbrs = BE.neighbours(ub, rows_, nf)
    tb = BE.BorrowTable(m).fit(m, tr, ub, nbrs)
    mb = m.clone()
    mb.extras.append(tb)
    mb.refresh()
    rt_b = measure(mb, "4_借りる")
    sel = lab_te == "自信のなさ_選び"
    res["4_借りる_自信のなさの名札の間"] = dict(
        間=int(sel.sum()), 外れ_前=round(float(wrong_te0[sel].mean()), 4),
        外れ_後=round(float(np.mean([rt_b[i]["pick"] != rt_b[i]["gold"] for i in np.nonzero(sel)[0]])), 4),
        自信のなさ_前=round(float(np.mean([rows_te[i]["自信のなさ_選び"] for i in np.nonzero(sel)[0]])), 4),
        自信のなさ_後=round(float(np.mean([rt_b[i]["自信のなさ_選び"] for i in np.nonzero(sel)[0]])), 4))
    print("4", json.dumps(res["4_借りる_自信のなさの名札の間"], ensure_ascii=False), flush=True)
    # 残った間: 本体で迷い・自信のなさの線を越えた試しの間のうち、回す 3 回目・借りる・試して比べる のどれでも線を下回らなかった物
    flagged = [i for i, r in enumerate(rows_te) if any(r[kd] > thr[kd] for kd in LOOP_KINDS)]
    keep = []
    for i in flagged:
        still = [any(rt[i][kd] > thr[kd] for kd in LOOP_KINDS) for rt in (rt_loop, rt_b)]
        if all(still) and self_picks.get(i, rows_te[i]["pick"]) == rows_te[i]["pick"]:
            keep.append(i)
    xs_te = [x for s in te for x in (AE.items(S, s, "節") or [])]
    say_k = Counter()
    for i in keep:
        r = rows_te[i]
        if r["pick"] != r["gold"]:
            say_k[say.judge(xs_te[i], m.pg.cls[r["pick"]])] += 1
    res["残った間"] = dict(線を越えた間=len(flagged), 残った間=len(keep), 名札の内訳=dict(Counter(lab_te[keep].tolist())),
                       外れの割合=round(float(np.mean([rows_te[i]["pick"] != rows_te[i]["gold"] for i in keep])), 4) if keep else None,
                       外れの言える物=dict(say_k), 外れの中身=dict(Counter(("候補に無い" if rows_te[i]["gold"] == m.nc - 1 else ("文末の形" if rows_te[i]["last"] else "別の助詞"))
                                                         for i in keep if rows_te[i]["pick"] != rows_te[i]["gold"])))
    print("残り", json.dumps(res["残った間"], ensure_ascii=False), flush=True)
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
