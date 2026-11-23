"""I49k: 座り方から足し算で作った向き（intervention/i49k_seat_vec.SeatVec）で、I49j の比べをやり直す。数えだけ。

材料・流れ・引数は tools/i49h_tally_run.py と同じ。向きは青空の 3 つ組だけから作り、会話を流している間は変えない（会話の中身は向きに入れない）。
始め方は「青空を先に」だけ（向きを青空から作るので、そろえる）。
並べる物（向きの作り方ごとに）: 関係の表（語板の向き。比べる元）／ 関係の表（作った向きで借りる）／ ShiftVec ずれ・真ん中・近さ（作った向き）。
向きの作り方: 数（聞いた数のまま）／ 偏り（たまたまより多い分だけ）。
足して数える物: 作った向きを持つ語の割合（会話の内容語のうち）、語板の向きは無いが作った向きは有る語の数。
                物差しの目安: 作った向きどうしの |cos| の中央値（会話の内容語から、並び順に 2,000 組）。向きがつぶれて全部同じになっていないかを見る。
記録はリポジトリの外。手で置く物: i49h_tally_run と同じ、目安に使う組の数 2,000。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49h_tally_run as HR  # noqa: E402
import i49j_shift_vec_run as JR  # noqa: E402


def coverage(board_of, made_of, talks, pairs=2000):
    words = sorted({w for _, ws in talks for w in ws}, key=str)
    has = [w for w in words if made_of(w) is not None]
    new = [w for w in has if board_of(w) is None]
    cs = []
    for a, b in zip(has[0::2], has[1::2]):
        va, vb = made_of(a), made_of(b)
        cs.append(float(np.abs(np.vdot(va, vb)) / (np.linalg.norm(va) * np.linalg.norm(vb))))
        if len(cs) >= pairs:
            break
    return dict(内容語=len(words), 語板の向きが有る=sum(board_of(w) is not None for w in words), 作った向きが有る=len(has),
                語板には無いが作った向きは有る=len(new), 作った向きどうしの近さの中央値=round(float(np.median(cs)), 4) if cs else None)


def run(board_of, talks, aoz, weight):
    import i49e_relation as RL
    import i49j_shift_vec as SV
    import i49k_seat_vec as KV
    sv = KV.SeatVec(board_of, weight=weight)
    for tri in aoz:
        sv.hear(tri)
    made = sv.of
    mk = (("関係の表（語板の向き）", lambda: RL.Relation(board_of, half=HR.HALF)),
          ("関係の表（作った向き）", lambda: RL.Relation(made, half=HR.HALF))) + tuple(
        (m, (lambda m=m: SV.ShiftVec(made, mode=m))) for m in SV.MODES)
    res = JR.run(made, talks, aoz, mk)
    res["向き"] = coverage(board_of, made, talks)
    return res


def main():
    a, vec_of, talks, aoz, t0 = HR.load()
    res = dict(会話=len(talks), 青空の組=len(aoz))
    print("用意", round(time.time() - t0), flush=True)
    for w in ("数", "偏り"):
        res[w] = run(vec_of, talks, aoz, w)
        print(w, round(time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
