"""I49k: 材料（青空の 3 つ組）の量を変えて、当たりが伸び続けるかを見る（持ち主の案「学習の資料不足の検証も必要かも」）。数えだけ。

材料・流れ・引数は tools/i49h_tally_run.py と同じ。青空の 3 つ組を、並びの 8 つおき・4 つおき・2 つおき・全部（1/8、1/4、1/2、1）で使う。
量ごとに並べる物: 関係の表（語板の向き。その量を先に learn）／ ShiftVec 真ん中・ずれ（その量から作った向き、数の重さ）。
量ごとに数える物: 1 位の割合、白紙の割合、作った向きを持つ会話の内容語の割合。
読み方の目安: 量を 2 倍にするたびの伸びが小さくなっていれば、作りの方の限り。同じだけ伸びていれば、材料不足。
記録はリポジトリの外。手で置く物: 量の刻み（1/8、1/4、1/2、1）、何個おきに取るかの取り方、i49h_tally_run と同じ物。
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

STEPS = (8, 4, 2, 1)


def run(board_of, talks, aoz, steps=STEPS):
    import i49e_relation as RL
    import i49j_shift_vec as SV
    import i49k_seat_vec as KV
    words = sorted({w for _, ws in talks for w in ws}, key=str)
    out = {}
    for st in steps:
        sub = aoz[::st]
        sv = KV.SeatVec(board_of, weight="数")
        for tri in sub:
            sv.hear(tri)
        row = dict(組=len(sub), 作った向きを持つ内容語の割合=round(sum(sv.of(w) is not None for w in words) / max(1, len(words)), 4))
        for name, make in (("関係の表", lambda: RL.Relation(board_of, half=HR.HALF)),
                           ("真ん中", lambda: SV.ShiftVec(sv.of, mode="真ん中")),
                           ("ずれ", lambda: SV.ShiftVec(sv.of, mode="ずれ"))):
            rel = make()
            for tri in sub:
                rel.learn(tri)
            s = HR.summary(HR.stream_rec(rel, talks), len(talks))
            row[name] = dict(一位=s["全体"]["一位"], 白紙=s["全体"]["白紙"])
        out["1/%d" % st] = row
    return out


def main():
    a, vec_of, talks, aoz, t0 = HR.load()
    res = dict(会話=len(talks), 青空の組=len(aoz))
    res["量ごと"] = run(vec_of, talks, aoz)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
