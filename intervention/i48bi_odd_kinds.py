"""I48bi: 違和感を種類に分ける（今ある物の名札づけ。持ち主の案、未確定）。

I48at (三) の違和感は 5 つの材料の順位の和だった。ここでは和にせず、種類ごとの値を別々に持ち、間ごとに「いちばん強い種類」を名札にする。
種類（どれも大きいほど違和感が強い向きにそろえる）:
  迷い            選びの 1 位と 2 位の点の差（今 ＋ R1 ＋ R2 ＋ R3）の符号を返した物
  自信のなさ_選び  選んだ候補の確からしさ（同じ点の softmax）の符号を返した物（答えを知らなくても出せる）
  自信のなさ_正解  正解の確からしさの符号を返した物（I48at と同じ。答えを知っている二回り目だけで出せる）
  筋と波の食い違い 選んだ候補の、論理の筋（R2 ＋ R3）の順位 ＋ 全体の波（R1）の順位
  筋の無い共振     選んだ候補の筋の点がその間の中央値以下で、波の点がいちばん（0 か 1）
  組の珍しさ       選んだ詰め物の助詞と、右のいちばん近い述語の PMI（学びの文の数え、I48at の Natural）の符号を返した物
  手持ちに無い     本体の今の点（log p）での「その他」の確からしさ
  介入の板と本体の食い違い: 介入の板が空（何も教えていない）なので、この試しでは出せない（記録だけ）。
名札: 種類ごとに、その照らしの全部の間の中での順位の割合（0〜1、1 がいちばん違和感）を出し、いちばん大きい種類（同じなら並びの先）。
測る物（試しの文 1,000 文と、学んだ文の先頭 1 万文の二回り目）:
  種類ごとの上位 1 割が外れだった割合・外れのうち上位 1 割に入った割合、和（I48at と同じ順位の和）の同じ値、
  外れの中身（候補に無い＝正解がその他 ／ 文末の形＝最後の間 ／ 別の助詞＝それ以外）ごとの、種類の上位 1 割に入った割合、
  種類どうしの上位 1 割の重なり（ジャカード）、名札ごとの間の数と外れの割合。
手で置く物: 上位 1 割、学んだ文の照らし直しの数 1 万、種類の並び（名札が同じ値の時の順）。
python intervention/i48bi_odd_kinds.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000] [--recheck 10000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402

KINDS = ("迷い", "自信のなさ_選び", "自信のなさ_正解", "筋と波の食い違い", "筋の無い共振", "組の珍しさ", "手持ちに無い")
LABEL_KINDS = ("迷い", "自信のなさ_選び", "筋と波の食い違い", "筋の無い共振", "組の珍しさ", "手持ちに無い")   # 名札は答えを使わない種類だけ
TOP = 0.1


def rows_of(m, sents, nat):
    rows = []
    for j, s in enumerate(sents):
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        keys = AK.keys_of(s)
        fic = m.who.fiction(keys)
        ups = {u for _, u in keys}
        placed = []
        for x in xs:
            g = m.gap_scores(keys, x, fic)
            r3 = m.gram.score(m.gram.state(ups, placed, x["last"]))
            logic, wave = g["R2"] + r3, g["R1"]
            three = wave + logic
            both = g["今"] + three
            p_other = float(np.exp(g["今"][-1] - np.logaddexp.reduce(g["今"])))
            three[-1] = both[-1] = -1e9
            cand = (set(np.argsort(-g["今"])[:AK.K]) | set(np.argsort(-three)[:AK.K])) - {m.nc - 1}
            pick = max(cand, key=lambda c: both[c])
            top2 = np.sort(both)[-2:]
            p = np.exp(both - both.max())
            p /= p.sum()
            gold = m.cid(x["y"])
            lg, wv = logic.copy(), wave.copy()
            lg[-1] = wv[-1] = -1e9
            pred = AJ.next_pred(keys, x["i"])
            rows.append(dict(j=j, i=int(x["i"]), last=bool(x["last"]), gold=int(gold), pick=int(pick),
                             迷い=-float(top2[1] - top2[0]), 自信のなさ_選び=-float(p[pick]), 自信のなさ_正解=-float(p[gold]),
                             筋と波の食い違い=float(AJ.ranks(lg)[pick] + AJ.ranks(wv)[pick]),
                             筋の無い共振=float(lg[pick] <= np.median(lg[:-1]) and int(np.argmax(wv)) == pick),
                             組の珍しさ=-nat.score(m.pg.cls[pick], pred[0] if pred else None), 手持ちに無い=p_other))
            placed.append(m.pg.cls[pick])
    return rows


def pct(v):
    """順位の割合（0〜1、大きいほど違和感が強い）。同じ値は同じ割合。"""
    order = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    u, inv = np.unique(v, return_inverse=True)
    mean_rank = np.bincount(inv, weights=order) / np.bincount(inv)
    return mean_rank[inv] / max(1, len(v) - 1)


def analyse(m, rows):
    n = len(rows)
    wrong = np.array([r["pick"] != r["gold"] for r in rows])
    other = m.nc - 1
    etype = np.array(["正解なし" if not w else ("候補に無い" if r["gold"] == other else ("文末の形" if r["last"] else "別の助詞"))
                      for r, w in zip(rows, wrong)])
    k = max(1, int(n * TOP))
    P = {kd: pct(np.array([r[kd] for r in rows])) for kd in KINDS}
    tops = {}
    out = dict(間=n, 外れの割合_全体=round(float(wrong.mean()), 4),
               外れの中身={t: int((etype == t).sum()) for t in ("候補に無い", "文末の形", "別の助詞")}, 種類={})
    for kd in KINDS:
        v = np.array([r[kd] for r in rows])
        idx = np.argsort(-v, kind="stable")[:k]
        if kd == "筋の無い共振":
            idx = np.nonzero(v > 0)[0]                         # 0 か 1 なので、立った間の全部
        tops[kd] = set(idx.tolist())
        w = wrong[idx]
        out["種類"][kd] = dict(上位の間=len(idx), 上位が外れだった割合=round(float(w.mean()), 4) if len(idx) else None,
                             外れのうち上位に入った割合=round(float(w.sum() / max(1, wrong.sum())), 4),
                             外れの中身ごとの上位に入った割合={t: round(float(np.isin(np.nonzero(etype == t)[0], idx).mean()), 4)
                                                     for t in ("候補に無い", "文末の形", "別の助詞") if (etype == t).any()})
    # 和（I48at と同じ順位の和。材料は 差・正解の確からしさ・食い違い・筋無し・自然さ）
    rs = sum(np.argsort(np.argsort(v, kind="stable"), kind="stable")
             for v in (np.array([-r["迷い"] for r in rows]), np.array([-r["自信のなさ_正解"] for r in rows]),
                       -np.array([r["筋と波の食い違い"] for r in rows]), -np.array([r["筋の無い共振"] for r in rows]),
                       np.array([-r["組の珍しさ"] for r in rows])))
    idx = np.argsort(rs, kind="stable")[:k]
    out["和_I48atと同じ"] = dict(上位が外れだった割合=round(float(wrong[idx].mean()), 4),
                             外れのうち上位に入った割合=round(float(wrong[idx].sum() / max(1, wrong.sum())), 4))
    # 答えを使わない和（正解の確からしさの代わりに選びの確からしさ）
    rs2 = sum(np.argsort(np.argsort(-P[kd], kind="stable"), kind="stable") for kd in LABEL_KINDS)
    idx2 = np.argsort(rs2, kind="stable")[:k]
    out["和_答えを使わない種類だけ"] = dict(上位が外れだった割合=round(float(wrong[idx2].mean()), 4),
                                   外れのうち上位に入った割合=round(float(wrong[idx2].sum() / max(1, wrong.sum())), 4))
    out["重なり_ジャカード"] = {f"{a}×{b}": round(len(tops[a] & tops[b]) / max(1, len(tops[a] | tops[b])), 3)
                            for ia, a in enumerate(KINDS) for b in KINDS[ia + 1:]}
    M = np.stack([P[kd] for kd in LABEL_KINDS], axis=1)
    lab = np.array(LABEL_KINDS)[np.argmax(M, axis=1)]
    out["名札"] = {kd: dict(間=int((lab == kd).sum()), 外れの割合=round(float(wrong[lab == kd].mean()), 4) if (lab == kd).any() else None,
                         外れの中身={t: int(((lab == kd) & (etype == t)).sum()) for t in ("候補に無い", "文末の形", "別の助詞")})
                  for kd in LABEL_KINDS}
    # 名札の強さ（いちばん強い種類の割合）の上位 1 割
    strength = M.max(1)
    idx3 = np.argsort(-strength, kind="stable")[:k]
    out["名札の強さの上位"] = dict(上位が外れだった割合=round(float(wrong[idx3].mean()), 4),
                            名札の内訳={kd: int((lab[idx3] == kd).sum()) for kd in LABEL_KINDS})
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--recheck", type=int, default=10000)
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
    fill, surf = AD.Fill(S), G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    m = AP.UModel(S, fill, a.r1).fit(tr, hold).init_units("C")       # I48at と同じ型
    m.refresh()
    res = dict(学びの文=len(tr), 試しの文=len(te), 照らし直しの文=a.recheck, 上位=TOP, 種類=list(KINDS), 名札に使う種類=list(LABEL_KINDS),
               介入の板と本体の食い違い="介入の板が空なので出せない（教えた後の型で測る物）", 土台=AO.measure(m, te, surf))
    print(f"学び {time.time() - t0:.0f} 秒", json.dumps(res["土台"], ensure_ascii=False)[:200], flush=True)
    nat = RC.Natural(fill, tr)

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for nm, ss in (("試しの文", te), ("学んだ文の二回り目", tr[: a.recheck])):
        t1 = time.time()
        rows = rows_of(m, ss, nat)
        res[nm] = analyse(m, rows)
        res[nm]["秒"] = round(time.time() - t1)
        print(nm, json.dumps(res[nm], ensure_ascii=False)[:1500], flush=True)
        save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
