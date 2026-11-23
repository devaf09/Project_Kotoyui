"""I49j: 表を持たず向きのずれだけで答える形（intervention/i49j_shift_vec.ShiftVec）を、文の 3 つ組で関係の表と比べる。数えだけ。

材料・流れ・数え方は tools/i49h_tally_run.py と同じ（会話を 1 本ずつ、答える前に問う）。引数も同じ。
並べる物: Relation（数の表）／ ShiftVec ずれ ／ 真ん中 ／ 近さ。始め方: 白紙から ／ 青空の 3 つ組を先に learn してから。
足して数える物: 答えた問いだけの 1 位の割合（白紙を除く）、見せた語と答えの 3 語すべてに向きが有る問いだけの 1 位の割合、
                述語側の語を初めて聞く問い・2 語の組を初めて聞く問いだけの 1 位の割合（見せたことの無い組で運べるか）。
記録はリポジトリの外。手で置く物: i49h_tally_run と同じ。
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49h_tally_run as HR  # noqa: E402


def makers(vec_of):
    import i49e_relation as RL
    import i49j_shift_vec as SV
    return (("Relation", lambda: RL.Relation(vec_of, half=HR.HALF)),) + tuple(
        (m, (lambda m=m: SV.ShiftVec(vec_of, mode=m))) for m in SV.MODES)


def flags(vec_of, talks, pre):
    """stream_rec と同じ並びで、問いごとの印: (3 語とも向きが有る, 述語側の語を前に聞いた, 見せた 2 語の組を前に聞いた)。"""
    from i49e_relation import NEAR
    seen_near, seen_pair = set(), set()

    def note(tri):
        for si in range(3):
            near = NEAR[si]
            other = next(j for j in range(3) if j not in (si, near))
            seen_near.add((si, tri[near]))
            seen_pair.add((si, tri[near], tri[other]))
    for tri in pre:
        note(tri)
    out = []
    for tris, words in talks:
        for tri in tris:
            for si in range(3):
                shown = {tri[j] for j in range(3) if j != si}
                if len({w for w in words if w not in shown} | {tri[si]}) < 2:
                    continue
                near = NEAR[si]
                other = next(j for j in range(3) if j not in (si, near))
                out.append((all(vec_of(w) is not None for w in tri), (si, tri[near]) in seen_near, (si, tri[near], tri[other]) in seen_pair))
            note(tri)
    return out


def rate(recs):
    return dict(問い=len(recs), 一位=round(sum(r[2] for r in recs) / max(1, len(recs)), 4))


def run(vec_of, talks, pre, mk=None):
    """mk = ((名前, 作る関数), ...) を渡せば、それを並べる（1 つ目が比べる元）。"""
    fl = flags(vec_of, talks, pre)
    res, base = {}, None
    for name, make in (mk or makers(vec_of)):
        rel = make()
        for tri in pre:
            rel.learn(tri)
        t1 = time.time()
        recs = HR.stream_rec(rel, talks)
        assert len(recs) == len(fl)
        r = HR.summary(recs, len(talks))
        r["秒"] = round(time.time() - t1, 1)
        r["答えた問いだけ"] = rate([x for x in recs if x[3] != "答えられない"])
        r["3 語とも向きが有る問い"] = rate([x for x, f in zip(recs, fl) if f[0]])
        r["述語側の語は聞いたが、2 語の組は初めて"] = rate([x for x, f in zip(recs, fl) if f[1] and not f[2]])
        r["2 語の組を前に聞いた"] = rate([x for x, f in zip(recs, fl) if f[2]])
        r["述語側の語も初めて"] = rate([x for x, f in zip(recs, fl) if not f[1]])
        if base is None:
            base = recs
        else:
            r["Relation との違い"] = HR.compare(base, recs)
        res[name] = r
    return res


def main():
    a, vec_of, talks, aoz, t0 = HR.load()
    res = dict(会話=len(talks), 青空の組=len(aoz))
    print("用意", round(time.time() - t0), flush=True)
    for name, pre in (("白紙から", []), ("青空を先に", aoz)):
        res[name] = run(vec_of, talks, pre)
        print(name, round(time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
