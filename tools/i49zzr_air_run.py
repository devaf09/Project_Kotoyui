"""I49zzr: 状況波をずらして会話を書き替え、違和感チェックに通す道具（`intervention/i49zzr_air.py`。持ち主の案、未確定の試し）。

材料: 話し手の印の有る会話（`--jsonl`）か、空の行を場の切れ目にした 1 行 1 発話のテキスト（`--text`）。
流れ:
  0. 違和感チェックそのものを確かめる: 材料の全部で数え、材料の本物の文（どこかをつなぎ目とみなす。自分の 1 回は除く）と、
     壊した文（終わりの k 文字を別の文の終わりと入れ替えた物）を通す。本物を通し、壊した文を止めるほど良い。
  1. 状況波（位相の形、DIMS 次元。1 個前の発話 → 今の発話の組で作る）に自然に出た強い向きを DIRS 本取り出す。
     向きごとに、両端の候補の終わりに付く印（丁寧・過去・問い）の割合を出す（読むためだけ。向きに名前は付けない）。
  2. 会話（`--lines`、｜区切り。無ければ道具の中の 5 行）の各行を、「前の行の終わり」の状況波を向きごとに ± へずらして書き替える。
     出すのは、合い方が一番高い候補（チェック無し）と、チェックを通った中で一番高い候補（チェック有り）。
  3. 同じ事を、材料から黄金比の並びで抜いた PICK 文でもやり、通った割合を数える（文そのものは出さない）。
     比べ役: 状況波を見ずに、つなぎ目の表に有る候補から 1 つ取った時の通る割合。
  4. 中身を守る: 消す終わりと入れる終わりを、手前の並びの種類が下限（KEEPS）以上の物に限って、2 と 3 をもう一度数える。
     読むための印: 中身の字（漢字・かたかな）が消えた・増えた行の割合、残った頭の割合、状況波との合い方の平均（守ると、どれだけ効きが減るか）。
手で置いた物: DIMS、DIRS、PICK、ずらす大きさ、違和感を見る並びの長さ、打ち消しの印の決め方、つなぎ目の表の下限。
`--out` には書き替えた行（材料の文の終わりが入る）が出るので、リポジトリには入れない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49_situation as SIT  # noqa: E402
import i49zzq_phase_run as ZQ  # noqa: E402
import i49zzr_air as AIR  # noqa: E402
from i49zzq_phase_wave import PhaseAssoc  # noqa: E402

DIMS = 2048
DIRS = 3
PICK = 300
COUNT = 20
TALK = ("今日は仕事が早く終わりました", "それは良かったですね", "帰りに駅前の店でご飯を食べたんだ", "何を食べたの", "魚の定食がおいしかったよ")
NEG = ("ない", "ません", "なかった", "ぬ", "ず")   # 打ち消しの印（読むためだけ。手で決めた物）


def has_neg(s):
    return any(x in s[-8:] for x in NEG)


def check_the_check(sents, n=None, limit=40000):
    """違和感チェックの確かめ。材料の全部で数え、材料の文を 1 つずつ通す（自分の 1 回は除く）。
    本物: 文のどこか（終わりの k 文字の手前）をつなぎ目とみなす。壊した文: 終わりの k 文字を、別の文の終わりと入れ替える。
    → 本物を通す割合、壊した文を通す割合、点（本物を通す割合と、壊した文を止める割合の平均 × 100）。"""
    chk = AIR.OddCheck(sents, n)
    test = [sents[i] for i in SIT.golden_order(len(sents))[:limit]]
    real = broke = nb = 0
    m = len(test)
    for i, s in enumerate(test):
        k = 1 + i % SIT.TAIL
        if len(s) < k + 2:
            continue
        o = test[(i + m // 2 + 1) % m]
        if len(o) < k or o[-k:] == s[-k:]:
            continue
        nb += 1
        real += chk.passes(s, k, least=2)
        broke += chk.passes(s[:-k] + o[-k:], k)
    r, b = real / max(nb, 1), broke / max(nb, 1)
    return dict(見る並びの長さ=chk.n, 試した文=nb, 本物を通す割合=round(r, 4), 壊した文を通す割合=round(b, 4), 点=round(50 * (r + 1 - b), 1))


def describe(phase, dirs, cands, top=30):
    out = []
    for d in dirs:
        order = sorted(cands, key=lambda c: (-phase.fit(d, c), c))
        row = {}
        for name, part in (("正の側", order[:top]), ("負の側", order[-top:])):
            row[name] = {k: round(sum(SIT.marks(c)[k] for c in part) / max(len(part), 1), 4) for k in ("丁寧", "過去", "問い")}
        out.append(row)
    return out


KEEPS = (0, 5, 20, 50, 100)   # 中身を守る時の、手前の並びの種類の数の下限（0 は守らない。手で置いた値の並び）
SHOW = 20                     # 会話の 30 行を出す時の下限（手で置いた値）


def content(s):
    """中身の字（漢字とかたかな）。読むための印で、仕組みには使わない。"""
    return {c for c in s if "\u4e00" <= c <= "\u9fff" or "\u30a1" <= c <= "\u30f6"}


def kept(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), 1)


def one(sent, sit, phase, table, chk, variety=None, least=0):
    cands = AIR.rewrites(sent, sit, phase, table, variety if least else None, least)
    free = cands[0]
    ok = [c for c in cands if c[2] == 0 or chk.passes(c[0], c[2])]
    gated = ok[0] if ok else (sent, 0.0, 0, 0)
    changed = [c for c in cands if c[2] > 0]
    # 比べ役: 状況波を見ずに、つなぎ目の表に有る候補から 1 つ取る（黄金比の並びで決める。乱数は使わない）
    rnd = changed[int(((len(sent) + 1) * SIT.PHI % 1.0) * len(changed))] if changed else None
    return dict(元=sent, チェック無し=free[0], 状況を見ない候補は通るか=(None if rnd is None else bool(chk.passes(rnd[0], rnd[2]))),
                チェック無しは通るか=bool(free[2] == 0 or chk.passes(free[0], free[2])),
                チェック有り=gated[0], 候補=len(cands), 変わった=bool(free[0] != sent), 点=round(free[1], 5),
                残った頭=round(kept(sent, free[0]), 4), 消した字=free[3],
                中身の字が消えた=bool(content(sent) - content(free[0])), 中身の字が増えた=bool(content(free[0]) - content(sent)),
                打ち消しが変わった=bool(has_neg(sent) != has_neg(free[0])), 打ち消しが変わった_チェック有り=bool(has_neg(sent) != has_neg(gated[0])))


def tally_of(rows):
    n = max(len(rows), 1)
    ch = [r for r in rows if r["変わった"]]
    m = max(len(ch), 1)
    rnd = [r for r in rows if r["状況を見ない候補は通るか"] is not None]
    return dict(書き替え=len(rows), 変わった=len(ch), 変わった割合=round(len(ch) / n, 4),
                変わった行のうち通る割合=round(sum(r["チェック無しは通るか"] for r in ch) / m, 4),
                状況を見ない候補が通る割合=round(sum(r["状況を見ない候補は通るか"] for r in rnd) / max(len(rnd), 1), 4),
                中身の字が消えた割合=round(sum(r["中身の字が消えた"] for r in ch) / m, 4),
                中身の字が増えた割合=round(sum(r["中身の字が増えた"] for r in ch) / m, 4),
                打ち消しが変わった割合=round(sum(r["打ち消しが変わった"] for r in ch) / m, 4),
                残った頭の平均=round(sum(r["残った頭"] for r in ch) / m, 4), 消した字の平均=round(sum(r["消した字"] for r in ch) / m, 3),
                合い方の平均=round(sum(r["点"] for r in rows) / n, 5))


def run(lines, talk=TALK, dims=DIMS, pick=PICK, keeps=KEEPS, show=SHOW):
    sents = SIT.distinct(list(SIT.sentences(lines)))
    across = list(SIT.across_pairs(SIT.runs_of(lines)))
    a = SIT.Assoc(across)
    phase = PhaseAssoc(a, dims)
    table = SIT.joins(sents)
    variety = AIR.varieties(table)
    chk = AIR.OddCheck(sents)
    cnt = {}
    for _, c in across:
        cnt[c] = cnt.get(c, 0) + 1
    cands = sorted((c for c, n in cnt.items() if n >= COUNT), key=lambda c: (-cnt[c], c))
    dirs, share = AIR.air_directions(phase, DIRS)
    norms = [float(np.linalg.norm(phase.situation([c]))) for c in cands[:200]]
    ref = float(np.mean(norms)) if norms else 1.0
    states = [("向き %d の%s" % (k + 1, "正" if sg > 0 else "負"), k, sg) for k in range(len(dirs)) for sg in (1, -1)]
    res = dict(文=len(sents), 組=len(across), 候補=len(cands), 位相の次元=dims, 終わりの種類=len(variety),
               違和感チェックの確かめ=[check_the_check(sents, n) for n in (3, 4, 5)],
               向きの取り分=[round(x, 4) for x in share], 向きの両端の印=describe(phase, dirs, cands))
    for label, least in (("会話", 0), ("会話_中身を守る", show)):
        out = []
        for i, s in enumerate(talk):
            base = phase.situation([SIT.tail_of(talk[i - 1])]) if i else np.zeros(dims, dtype=np.complex64)
            for name, k, sg in states:
                out.append(dict(行=i + 1, 状況=name, **one(s, AIR.shifted(base, dirs[k], sg, ref), phase, table, chk, variety, least)))
        res[label] = out
    res["会話の数え"] = {"守らない": tally_of(res["会話"]), "中身を守る": tally_of(res["会話_中身を守る"])}
    pool = [s for s in sents if 8 <= len(s) <= 30]
    picked = [pool[i] for i in SIT.golden_order(len(pool))[:pick]]
    by = {}
    for least in keeps:
        rows = []
        for j, s in enumerate(picked):
            base = phase.situation([SIT.tail_of(picked[j - 1])]) if j else np.zeros(dims, dtype=np.complex64)
            for name, k, sg in states:
                rows.append(one(s, AIR.shifted(base, dirs[k], sg, ref), phase, table, chk, variety, least))
        by[str(least)] = tally_of(rows)
    res["材料の文での数え"] = dict(抜いた文=len(picked), 下限ごと=by)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--lines", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    lines = ZQ.read_lines(a)
    talk = tuple(x for x in a.lines.split("｜") if x) or TALK
    res = dict(行=len(lines), 手で置いた値=dict(DIMS=DIMS, DIRS=DIRS, PICK=PICK, COUNT=COUNT, KEEPS=list(KEEPS), SHOW=SHOW, BETA=AIR.BETA, N=AIR.N, JOIN=SIT.JOIN, SEEN=SIT.SEEN, TAIL=SIT.TAIL),
               **run(lines, talk))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("会話") or k == "会話の数え"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
