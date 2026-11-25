"""I49zzq: 状況波を位相の形に置き換えて、元の形と比べる道具（`intervention/i49zzq_phase_wave.py`。持ち主の案、未確定の試し）。

材料: 1 行 1 発話のテキスト（`--text`）か、話し手の印の有る会話（`--jsonl`。発話を 1 行として使う）。組は 2 通り: 同じ発話の中の続いた 2 文（同じ話し手が続ける）と、1 個前の発話 → 今の発話（次の話し手が受ける）。
数える物（位相の次元 DIMS を変えて）:
  1. 2 つから選ぶ（I49zzm と同じ試し）: 元の形の当たりと、位相の形の当たり。2 つの形が同じ方を選んだ割合。
  2. 向き: 位相の波の強い向きが、元の波の強い向きと同じか（向きどうしの重なり。1 が同じ）。
  3. 入れ替え（I49zzo と同じ試し）: 丁寧な状況とくだけた状況の状況波を作り、合う順に並べた上位 TOP の終わりに丁寧の印が付く割合。
  4. 席の位相の波どうしの混ざり（別の席の波との重なりの大きさの平均。0 が混ざらない）。
手で置いた物: DIMS の並び、TOP、COUNT、入れる終わり（I49zzo と同じ）、学ぶ半分と試す半分の分け方。
出すのは数だけ。材料の文や終わりは出さない。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49_situation as SIT  # noqa: E402
from i49zzq_phase_wave import PhaseAssoc  # noqa: E402

DIMS = (16, 64, 256, 1024, 4096)
TOP = 50
COUNT = 20
POLITE = ("ですね", "ました", "ますよ")
CASUAL = ("だよ", "んだ", "だぜ")


def two_choice(score, held):
    """本当に続いた終わりと、別の組（半分ずらした所）の終わりから選ぶ。→ (当たり, 選んだ方の並び)。"""
    n = len(held)
    shift = n // 2 + 1
    ok, picks = 0.0, []
    for i, (p, c) in enumerate(held):
        other = held[(i + shift) % n][1]
        if other == c:
            ok += .5
            picks.append(0)
            continue
        s1, s0 = score(p, c), score(p, other)
        ok += 1.0 if s1 > s0 else (.5 if s1 == s0 else 0.0)
        picks.append(1 if s1 > s0 else (-1 if s1 < s0 else 0))
    return ok / max(n, 1), picks


def overlap_of_dirs(w, z, k=3):
    """元の波 w（実数）と位相の波 z（複素）の、強い向き k 本ずつの重なり（切り出しの側の向きで比べる）。"""
    if not w.size:
        return []
    u, _, _ = np.linalg.svd(w.astype(np.float64), full_matrices=False)
    v, _, _ = np.linalg.svd(z.astype(np.complex128), full_matrices=False)
    k = min(k, u.shape[1], v.shape[1])
    # 元の向き 1 本ごとに、位相の強い k 本の向きが張る所にどれだけ入るか
    return [round(float(np.sqrt((np.abs(v[:, :k].conj().T @ u[:, d]) ** 2).sum())), 4) for d in range(k)]


def mix(e, take=200):
    x = e[:take]
    g = np.abs(x @ x.conj().T) / e.shape[1]
    n = len(x)
    return float((g.sum() - np.trace(g)) / max(n * (n - 1), 1))


def polite_share(order):
    top = order[:TOP]
    return round(sum(SIT.marks(c)["丁寧"] for c in top) / max(len(top), 1), 4)


def run(inside, dims=DIMS):
    order = SIT.golden_order(len(inside))
    half = len(order) // 2
    train = [inside[i] for i in order[:half]]
    held = [inside[i] for i in order[half:]]
    a = SIT.Assoc(train)
    base, base_picks = two_choice(a.score, held)
    cnt = {}
    for _, c in inside:
        cnt[c] = cnt.get(c, 0) + 1
    cands = sorted((c for c, n in cnt.items() if n >= COUNT), key=lambda c: (-cnt[c], c))
    full = SIT.Assoc(inside)

    def rank_real(probe):
        return sorted(cands, key=lambda c: (-sum(full.score(p, c) for p in probe), c))

    res = dict(組=len(inside), 席=len(a.seats), 数えた切り出し=len(a.units), 候補=len(cands),
               候補の全体の丁寧の割合=round(sum(SIT.marks(c)["丁寧"] for c in cands) / max(len(cands), 1), 4),
               元の形=dict(二つから選ぶ=round(base, 4), 丁寧な状況の上位の丁寧=polite_share(rank_real(POLITE)),
                        くだけた状況の上位の丁寧=polite_share(rank_real(CASUAL))), 位相の形={})
    for d in dims:
        t0 = time.time()
        p = PhaseAssoc(a, d)
        acc, picks = two_choice(p.score, held)
        same = sum(x == y for x, y in zip(picks, base_picks)) / max(len(picks), 1)
        pf = PhaseAssoc(full, d)
        sp, sc = pf.situation(POLITE), pf.situation(CASUAL)
        res["位相の形"][str(d)] = dict(
            二つから選ぶ=round(acc, 4), 元の形と同じ方を選んだ割合=round(same, 4),
            強い向きの重なり=overlap_of_dirs(a.w, p.z), 席の波の混ざり=round(mix(p.e), 4),
            丁寧な状況の上位の丁寧=polite_share(sorted(cands, key=lambda c: (-pf.fit(sp, c), c))),
            くだけた状況の上位の丁寧=polite_share(sorted(cands, key=lambda c: (-pf.fit(sc, c), c))),
            秒=round(time.time() - t0, 2))
    return res


def read_lines(a):
    lines = []
    if a.jsonl:
        op = gzip.open if a.jsonl.endswith(".gz") else open
        with op(a.jsonl, "rt", encoding="utf-8") as f:
            last = None
            for row in f:
                r = json.loads(row)
                if last is not None and r["dialogue_id"] != last:
                    lines.append("")                     # 会話の切れ目は、場の切れ目
                last = r["dialogue_id"]
                lines.append(r["text"].replace("\n", "。"))
    else:
        for path in a.text.split(","):
            with open(path, encoding="utf-8") as f:
                lines += [x.rstrip("\n") for x in f]
    return lines[:a.limit] if a.limit else lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    lines = read_lines(a)
    res = dict(行=len(lines), 手で置いた値=dict(DIMS=list(DIMS), TOP=TOP, COUNT=COUNT, TAIL=SIT.TAIL, SEATS=SIT.SEATS, MIN=SIT.MIN), 同じ話し手が続ける=run(SIT.pairs_of(lines)[0]), 次の話し手が受ける=run(list(SIT.across_pairs(SIT.runs_of(lines)))))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
