"""I49zzza: 話の中でもう出た語を「持つ／遠ざける」形で、次の 3 つ組の語を当てる（77 節の続き。数えだけの形。未確定の試し）。

77 節: 人は 8 割近くが話の中にもう居る人、物と動きは 8〜9 割がその話で初めての語。なら、席ごとに使い方を変える。
場面は 74 節と同じ（話を並べ替えて半分に分け、前半を聞かせ、後半は問うてから聞かせる）。候補は、話の中の語。
だれが の形:
  引き継ぎ／遷移の数え（74 節と同じ。比べ役）
  続いた数で選ぶ:        今の人が何行続いているか（1〜4）ごとに、「続く」と「1 つ前の別の人へ戻る」のどちらが多かったかを聞いた話から数え、多い方を答える。
  話の中の人から遷移で:  候補を、その話でもう出た人だけにして、遷移の数えで選ぶ。
  続いた数＋話の中の人:  続いた数で「続く」が多ければ前の人。そうでなければ、前の人を除いた話の中の人から、遷移の数えで選ぶ（居なければ前の人）。
何を・どうした の形:
  遷移の数え（比べ役）／話でもう出た語を除く: その話の、その席にもう出た語を候補から除いて、遷移の数えで選ぶ。
  当てずっぽうも 2 つ出す（全部の候補から／除いた後の候補から）。候補は話の中の語なので、除くだけで当たりやすくなる分を見るため。
手で置いた物: 続いた数を 4 でまとめる事、場面の作り方。乱数は使わない。材料の語や例は出さない。
引数: --text 材料（カンマ区切り） --out 記録.json [--ks 1,2,3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49zp_text_talks_run as ZP  # noqa: E402
import i49zzw_next_triple_run as W  # noqa: E402
import i49zzz_talk_memory_run as Z  # noqa: E402

WHO = ("引き継ぎ", "遷移の数え", "続いた数で選ぶ", "話の中の人から遷移で", "続いた数＋話の中の人")
WHAT = ("遷移の数え", "話でもう出た語を除く")


class RunTally:
    """続いた数ごとの、「続く」「1 つ前の別の語へ戻る」の回数。"""

    def __init__(self):
        self.n = {r: [0, 0] for r in range(1, Z.RUN_MAX + 1)}

    def learn(self, hist, nxt):
        k, run = Z.kind_of(hist, nxt)
        if k in Z.KINDS[:2]:
            self.n[run][Z.KINDS.index(k)] += 1

    def stay(self, run):
        return self.n[run][0] >= self.n[run][1]


def back_of(hist):
    """今の語が続き始める前に、その席に居た語（無ければ None）と、続いた数。"""
    j = len(hist) - 1
    while j >= 0 and hist[j] == hist[-1]:
        j -= 1
    return (hist[j] if j >= 0 else None), min(len(hist) - 1 - j, Z.RUN_MAX)


def among(cands, scores, keep):
    ix = [i for i, c in enumerate(cands) if keep(c)]
    return W.pick([cands[i] for i in ix], [scores[i] for i in ix]) if ix else None


def count(first, second):
    m, tally = W.Count(), RunTally()

    def hear(tris):
        for i in range(1, len(tris)):
            m.learn(tris[i - 1], tris[i])
            tally.learn([t[0] for t in tris[:i]], tris[i][0])

    for tris, _ in first:
        hear(tris)
    out = {"だれが": dict(問い=0, 当てずっぽう=0.0, **{n: 0 for n in WHO})}
    for seat in ZP.SEATS[1:]:
        out[seat] = dict(問い=0, 当てずっぽう=0.0, 当てずっぽう_除いた後=0.0, 正解がもう出た語=0, **{n: 0 for n in WHAT})
    for tris, words in second:
        cands = sorted(words, key=str)
        for i in range(1, len(tris)):
            a, b = tris[i - 1], tris[i]
            hist = [t[0] for t in tris[:i]]
            sc = m.scores(0, a, cands)
            back, run = back_of(hist)
            inside = set(hist)
            other = among(cands, sc, lambda c: c in inside and c != a[0])
            ans = {"引き継ぎ": a[0], "遷移の数え": W.pick(cands, sc), "続いた数で選ぶ": a[0] if tally.stay(run) or back is None else back,
                   "話の中の人から遷移で": among(cands, sc, lambda c: c in inside),
                   "続いた数＋話の中の人": a[0] if tally.stay(run) or other is None else other}
            c = out["だれが"]
            c["問い"] += 1
            c["当てずっぽう"] += 1 / len(cands)
            for n in WHO:
                c[n] += ans[n] == b[0]
            for si in (1, 2):
                seen = {t[si] for t in tris[:i]}
                sc = m.scores(si, a, cands)
                left = [x for x in cands if x not in seen]
                c = out[ZP.SEATS[si]]
                c["問い"] += 1
                c["当てずっぽう"] += 1 / len(cands)
                c["当てずっぽう_除いた後"] += (b[si] not in seen) / max(len(left), 1)
                c["正解がもう出た語"] += b[si] in seen
                c["遷移の数え"] += W.pick(cands, sc) == b[si]
                c["話でもう出た語を除く"] += among(cands, sc, lambda x: x not in seen) == b[si]
        hear(tris)
    return out


def run(talks, ks=(1, 2, 3)):
    res, total = {}, {}
    for k in ks:
        seq = [talks[i] for i in ZP.order(len(talks), k)]
        d = count(seq[:len(seq) // 2], seq[len(seq) // 2:])
        res["元の順" if k == 0 else "並べ替え %d" % k] = d
        for seat, c in d.items():
            t = total.setdefault(seat, {})
            for key, v in c.items():
                t[key] = round(t.get(key, 0) + v, 1)
    res["合計"] = total
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="1,2,3")
    a = ap.parse_args()
    talks = []
    for f in a.text.split(","):
        talks += [(t[0], t[1]) for t in ZP.talks_of(Z.read(f))[0]]
    res = dict(話=len(talks), **run(talks, tuple(int(x) for x in a.ks.split(","))))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for seat, c in res["合計"].items():
        print(seat, c["問い"], {k: round(v / c["問い"], 3) for k, v in c.items() if k != "問い"})


if __name__ == "__main__":
    main()
