"""I49zf: 板は、出た語が無い時も答えを替えてしまうから崩れるのか、を確かめる（未確定の試し）。数えだけ。

I49ze で、並べた板は上位 3 から選ぶと崩れた。決まりで書いた「出た語」は、出た語が上位に無ければ一位のまま動かさない。
ここでは、並べた板・上位 3 に関門を付けて、差がそこに有るかを見る。
並べる物（解く・属性（席あり）の答えを選び直す）:
  板無し ／ 出た語・上位 3（決まりで書いた比べ役）
  並べた板・上位 3 × 板 1 枚・席ごと（半分まで 1 組）× 関門 無し ／ 出た語が有る時だけ（決まり） ／ 同じ波が板に有る候補だけ（波）
  同じ波の関門・1 枚で、半分まで 4 組 ／ 薄れない（話ごとに空）
  続き: 板を話ごとに空に戻し、話の中だけで薄れさせる版（同じ波だけ × 1 枚の 1 組・4 組、席ごとの 1 組 ／ 関門無し・1 枚・1 組）
採点、流れ、引数、記録の書き方は tools/i49ze_row_board_run.py と同じ。
手で置く物: 薄れる速さ、上位 3、同じ波と見なす丸めの幅（1e-4）。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49ze_row_board_run as ZE  # noqa: E402

B = dict(pick="上位", top=3, board="並べる")
VERS = [("板無し", "属性", dict(pick="そのまま")), ("出た語・上位 3", "属性", dict(pick="出た語", top=3))]
for seat, sname in ((False, "1 枚"), (True, "席ごと")):
    for gate, gname in ((None, "関門無し"), ("出た語", "出た語が有る時だけ"), ("同じ波", "同じ波だけ")):
        VERS.append(("並べた板｜%s｜%s｜半分まで 1 組" % (sname, gname), "語の波", dict(B, by_seat=seat, gate=gate, half=1, carry=True)))
VERS.append(("並べた板｜1 枚｜同じ波だけ｜半分まで 4 組", "語の波", dict(B, gate="同じ波", half=4, carry=True)))
VERS.append(("並べた板｜1 枚｜同じ波だけ｜薄れない", "語の波", dict(B, gate="同じ波")))
# 続き: 話をまたいで板を持ち越すのを止め、話ごとに空に戻す（決まりで書いた「出た語」と同じ区切り）。薄れは話の中だけ。
VERS.append(("並べた板｜1 枚｜同じ波だけ｜半分まで 1 組・話ごとに空", "語の波", dict(B, gate="同じ波", half=1)))
VERS.append(("並べた板｜1 枚｜同じ波だけ｜半分まで 4 組・話ごとに空", "語の波", dict(B, gate="同じ波", half=4)))
VERS.append(("並べた板｜席ごと｜同じ波だけ｜半分まで 1 組・話ごとに空", "語の波", dict(B, by_seat=True, gate="同じ波", half=1)))
VERS.append(("並べた板｜1 枚｜関門無し｜半分まで 1 組・話ごとに空", "語の波", dict(B, half=1)))


def run(wave_of, attr_of, talks, aoz, done=None, save=None):
    old, ZE.VERS = ZE.VERS, VERS
    try:
        return ZE.run(wave_of, attr_of, talks, aoz, done, save)
    finally:
        ZE.VERS = old


def main():
    ZE.VERS = VERS
    ZE.main()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
