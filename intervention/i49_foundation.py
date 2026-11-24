"""今の土台の入口（2026-11-25 時点。試しの記録ではなく、今使う形を 1 か所から取れるようにするための薄い入口）。

今の土台は 2 つの部品でできている。新しい道具は、部品を直に組まずにここから取る。
  body(wave_of, attr_of) → 本体（数えて選ぶ。I49zc の Mix(upper=None)）
    数えた表（Relation）と、波で解く形（Unbind(SeatVec)）の答えのうち、段ごとにそれまで多く当たった方を使う。同じ数なら表。
  with_memory(wave_of, attr_of, reflect=True) → 本体に、短い覚え（外付け）を付けた物（I49zzd の TaughtBody）
    教わった 3 つ組は、話の間は外付けに持つ。reflect=True なら、話の終わり（tick）に本体へ聞かせて外付けを空に戻す。
    save() で本体を写し、restore(写し) で反映の前へ戻せる。
  どちらも、ask(隠す席, 3 つ組, 候補) / learn(3 つ組) / tick() で使う。with_memory には teach(席, 3 つ組, 答えた語) と settle(3 つ組) が有る。
手で置いた値: 表の薄れる速さ（HALF = 1024）、同じ数なら表・同じ数なら本体、反映の時機（話の終わり）。
記録: docs/i49_inner_vessel.md の 30 節（本体）、51 節（外付け・反映・セーブ）。
"""
from __future__ import annotations

HALF = 1024.0


def body(wave_of, attr_of):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    from i49zb_mix import Mix
    return Mix(RL.Relation(wave_of, half=HALF), KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True), upper=None)


def with_memory(wave_of, attr_of, reflect=True, base=None):
    from i49zzd_taught_body import TaughtBody

    def fresh():
        return body(wave_of, attr_of)

    return TaughtBody(base if base is not None else fresh(), fresh(), fresh=fresh, reflect=reflect)
