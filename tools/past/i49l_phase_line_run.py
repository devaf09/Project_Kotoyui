"""I49l: 位相で席を回す形・表なしで解く形・行のうねりで作る向きを、同じ流れで比べる。数えだけ。

材料・流れ・引数は tools/i49h_tally_run.py と同じ（会話を 1 本ずつ、答える前に問う）。比べる元はどれも 関係の表（語板の向き）。
1. 作った向きを止めたまま（青空の 3 つ組だけから作る。始め方は「青空を先に」）。向きの作り方ごとに ShiftVec ずれ・真ん中・近さ を並べる:
     座り方・ずらす（I49k と同じ）／ 座り方・位相 ／ 行のうねり・全部 ／ 行のうねり・強い所
     行のうねりの「行」は、青空の 3 つ組 1 つ（3 語）を 1 行とみなす（青空の行そのものは、この道具からは見えない）。
     座り方の 2 つには、表なしで解く形（Unbind）も並べる。
   --only_online を付けると、2 だけを数える。
2. 聞きながら育つ形（解く・位相 には、作った向きから「たまたまの分」を引く形も並べる）。（Unbind online。答え合わせの後に、その 3 つ組も向きに足す）: 白紙から ／ 青空を先に、回し方 ずらす ／ 位相。
数える物: I49j と同じ（全体・席ごと・問いの種類ごとの 1 位、白紙、比べる元との違い）。行のうねり・強い所では、足した周波数の割合の平均。
          作った向きを持つ内容語の割合、作った向きどうしの |cos| の中央値（つぶれていないかの目安）。
記録はリポジトリの外。手で置く物: i49h_tally_run と同じ、目安に使う組の数 2,000、3 つ組を 1 行とみなす事。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49h_tally_run as HR  # noqa: E402
import i49j_shift_vec_run as JR  # noqa: E402
import i49k_seat_vec_run as KR  # noqa: E402


def static(board_of, talks, aoz):
    import i49e_relation as RL
    import i49j_shift_vec as SV
    import i49k_seat_vec as KV
    import i49l_line_vec as LV
    out = {}
    for name in ("座り方・ずらす", "座り方・位相", "行のうねり・全部", "行のうねり・強い所"):
        kind, opt = name.split("・")
        if kind == "座り方":
            mk_vec = KV.SeatVec(board_of, weight="数", bind=opt)
            for tri in aoz:
                mk_vec.hear(tri)
        else:
            mk_vec = LV.LineVec(board_of, where=opt)
            for tri in aoz:
                mk_vec.hear_line(tri)
        made = mk_vec.of
        mk = (("関係の表（語板の向き）", lambda: RL.Relation(board_of, half=HR.HALF)),) + tuple(
            (m, (lambda m=m: SV.ShiftVec(made, mode=m))) for m in SV.MODES)
        if kind == "座り方":
            mk += (("解く", lambda: KV.Unbind(mk_vec)),)
        res = JR.run(made, talks, aoz, mk)
        res["向き"] = KR.coverage(board_of, made, talks)
        if kind == "行のうねり":
            res["足した周波数の割合の平均"] = round(mk_vec.used, 4)
        out[name] = res
    return out


def online(board_of, talks, aoz):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    out = {}
    for start, pre in (("白紙から", []), ("青空を先に", aoz)):
        def unb(bind, center=False, pair=False):
            sv = KV.SeatVec(board_of, weight="数", bind=bind, center=center, pair=pair)
            return KV.Unbind(sv, online=True)
        mk = (("関係の表（語板の向き）", lambda: RL.Relation(board_of, half=HR.HALF)),
              ("解く・ずらす", lambda: unb("ずらす")), ("解く・位相", lambda: unb("位相")),
              ("解く・位相・たまたまの分を引く", lambda: unb("位相", True)),
              ("解く・位相・組も束ねる", lambda: unb("位相", False, True)))
        out[start] = JR.run(board_of, talks, pre, mk)
    return out


def main():
    only = "--only_online" in sys.argv
    if only:
        sys.argv.remove("--only_online")
    a, vec_of, talks, aoz, t0 = HR.load()
    res = dict(会話=len(talks), 青空の組=len(aoz))
    print("用意", round(time.time() - t0), flush=True)
    res["聞きながら育つ形"] = online(vec_of, talks, aoz)
    print("育つ形", round(time.time() - t0), flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not only:
        res["向きを止めた形"] = static(vec_of, talks, aoz)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
