"""今の土台の入口（2026-11-25 時点。試しの記録ではなく、今使う形を 1 か所から取れるようにするための薄い入口）。

今の土台は 2 つの部品でできている。新しい道具は、部品を直に組まずにここから取る。
  body(wave_of, attr_of) → 本体（数えて選ぶ。I49zc の Mix(upper=None)）
    数えた表（Relation）と、波で解く形（Unbind(SeatVec)）の答えのうち、段ごとにそれまで多く当たった方を使う。同じ数なら表。
  with_memory(wave_of, attr_of, reflect=True) → 本体に、短い覚え（外付け）を付けた物（I49zzd の TaughtBody）
    教わった 3 つ組は、話の間は外付けに持つ。reflect=True なら、話の終わり（tick）に本体へ聞かせて外付けを空に戻す。
    save() で本体を写し、restore(写し) で反映の前へ戻せる。
  どちらも、ask(隠す席, 3 つ組, 候補) / learn(3 つ組) / tick() で使う。with_memory には teach(席, 3 つ組, 答えた語) と settle(3 つ組) が有る。
手で置いた値は HANDS に並べた（1 か所で見られるようにする。値は各ファイルに有り、ここを直しても動きは変わらない）。
  python intervention/i49_foundation.py で、表の形で出る。
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


# 今の土台の、手で置いている物の一覧: (名前, 値, 置き場所, 役目, 種類, 置き換えの案)
#   種類: 「減らす候補」= 数えや波から決められそうな物 ／ 「形」= 仕組みの形として置いた物（乱数の代わりなど） ／ 「場面」= 数える場面の決め事
HANDS = [
    ("薄れる速さ", HALF, "i49e_relation.Relation(half=)", "表の数えが半分に薄れるまでの時間（話の数）", "減らす候補", "材料の話の間隔や、同じ組がまた出るまでの間の分布から決める"),
    ("段の下がる順", "2語 → 片方だけ借りる → 述語側 → もう片方 → 借りる", "i49e_relation.Relation.ask", "表が、どの手掛かりから先に答えるか", "減らす候補", "段ごとの当たりを数えて並べ直す（Mix がしている選び方を、表の中にも入れる）"),
    ("借りる時の近さ", "|cos|", "i49e_relation.Relation", "似た語の数えを借りる時の、語どうしの近さ", "形", "—"),
    ("同じ数なら表", "表", "i49zb_mix.Mix.ask", "表と解く形の当たりが同じ時に使う方", "形", "—"),
    ("席の回し方", "位相の傾き = (席の組の番号 + 1) × 黄金比の小数部", "i49k_seat_vec.SeatVec(bind=\"位相\")", "どの席で一緒だったかを向きに残す", "形", "—（乱数を置かないための決め方）"),
    ("重さ", "聞いた数そのまま", "i49k_seat_vec.SeatVec(weight=\"数\")", "一緒に座った語の向きを足す時の重さ", "減らす候補", "たまたまより多い分だけにする形（weight=\"偏り\"）は有る。数えで選ぶ形にはしていない"),
    ("向きの長さ", "1 にそろえる", "i49k_seat_vec.SeatVec._base", "語の元の向き（属性の波）を足す前にそろえる", "形", "—"),
    ("解く形の近さ", "cos の実部（候補の向きの長さで割る）", "i49k_seat_vec.Unbind(norm=\"cos\")", "候補の語と、見せた 2 語から作った向きの近さ", "減らす候補", "割らない形・半分の形は有る。数えで選ぶ形にはしていない"),
    ("属性の強さ", "割合そのまま（帯ごとの重さなし）", "i49m_attr_base.AttrBase", "語の元の向きを、属性の割合から作る", "減らす候補", "席ごとに効く帯が違う（I49zzf）。帯を数えて選ぶ形は、当たりが動かなかった"),
    ("品詞の対応・同じ名前の語の選び方", "NOUN→名詞 など。品詞が合う物、無ければ初めの物", "i49m_attr_base.AttrBase.of", "材料の語を、語板の語に合わせる", "形", "—"),
    ("同じなら本体", "本体", "i49zzd_taught_body.TaughtBody.ask", "本体と外付けの当たりが同じ時に使う方", "形", "—"),
    ("選ぶ粒", "（本体の答えた段, 外付けの答えた段）", "i49zzd_taught_body.TaughtBody.ask", "本体と外付けのどちらを使うかを数える単位", "減らす候補", "害はこの粒の粗さから来る（I49zzd の続き 3）。段の組より細かい手掛かりを探す"),
    ("反映の時機", "話の終わり", "i49zzd_taught_body.TaughtBody.tick", "外付けの中身を本体へ聞かせる時", "減らす候補", "外付けの当たりが本体を上回らなくなった時、など数えで決める形"),
    ("外付けが聞く時", "その 3 つ組の問いが全部終わった後", "i49zzd_taught_body.TaughtBody.settle", "教わった 3 つ組を外付けに入れる時", "形", "—（先に入れると、同じ 3 つ組のほかの問いで答えを知っている事になる）"),
    ("数える場面", "話を半分に分ける・並べ替えの k・介入の割合 10% / 30% / 100%", "tools/i49zx_teach_board_run.py", "本体が聞いていない事を教える場面の作り方", "場面", "—"),
]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("| 名前 | 値 | 置き場所 | 役目 | 種類 | 置き換えの案 |\n|---|---|---|---|---|---|")
    for row in HANDS:
        print("| " + " | ".join(str(x).replace("|", "／") for x in row) + " |")
