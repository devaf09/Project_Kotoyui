"""I49m: 元の向きを属性の割合にした形（intervention/i49m_attr_base.AttrBase）で、表なしで解く形と、表の借り方を比べる。数えだけ。

材料・流れは tools/i49h_tally_run.py と同じ（会話を 1 本ずつ、答える前に問う。聞きながら育つ形）。引数も同じで、--content <内容語の板.json> を足す。
始め方: 白紙から ／ 青空を先に。比べる元は 関係の表（語の波の近さで借りる）。並べる物:
  解く・語の波            I49l の 解く・位相（比べ役）
  解く・属性              元の向き = 属性の割合（全部の帯）。席は位相で回す
  解く・属性・引く        上に、たまたまの分を引く（どの語にも付く属性が強く出るのを減らす）
  解く・属性・束ねる      上（引かない）に、一緒に座った 2 語を束ねた向きも足す
  解く・属性（話題の帯を除く）  G 帯を除いた属性だけ
  解く・属性・見せた語から   見せた語の作った向きを回し戻して、隠した席に来る語の属性を言い当て、候補の属性と比べる（聞いた事の無い候補も比べられる）
  解く・属性・両方 ／ 解く・語の波・両方   上と、今までの形（候補の作った向きを見る）の点を足す
  解く・属性・長さで割らない ／ 長さの平方根で割る   候補の作った向きの長さで割る割り方を変える（聞くほど下がる理由の見立て:
                          長さで割ると、たくさん聞いた語ほど不利になる。会話でよく答えになる語が、材料が増えるほど負けていく）
  表（属性の近さで借りる）  関係の表のまま、借りる時の近さだけを属性の割合の cos にする
足して数える物: 属性の向きを持つ会話の内容語の数（語の波を持つ数と並べる）、属性の数、属性の向きどうしの |cos| の中央値（並び順に 2,000 組）。
記録はリポジトリの外。手で置く物: i49h_tally_run と同じ、話題の帯を G とする事、目安に使う組の数 2,000。
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
import i49j_shift_vec_run as JR  # noqa: E402
import i49k_seat_vec_run as KR  # noqa: E402


def run(wave_of, board, talks, aoz):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    import i49m_attr_base as AB
    full = AB.AttrBase(board)
    no_g = AB.AttrBase(board, bands=[b for b in sorted({k[0] for k in full.ix}) if b != "G"])

    def unb(base, norm="cos", side="候補", **kw):
        return KV.Unbind(KV.SeatVec(base, weight="数", bind="位相", **kw), online=True, norm=norm, side=side)
    mk = (("表（語の波の近さで借りる）", lambda: RL.Relation(wave_of, half=HR.HALF)),
          ("解く・語の波", lambda: unb(wave_of)),
          ("解く・属性", lambda: unb(full.of)),
          ("解く・属性・引く", lambda: unb(full.of, center=True)),
          ("解く・属性・束ねる", lambda: unb(full.of, pair=True)),
          ("解く・属性（話題の帯を除く）", lambda: unb(no_g.of)),
          ("解く・属性・見せた語から", lambda: unb(full.of, side="見せた語")),
          ("解く・属性・両方", lambda: unb(full.of, side="両方")),
          ("解く・語の波・両方", lambda: unb(wave_of, side="両方")),
          ("解く・属性・長さで割らない", lambda: unb(full.of, norm="なし")),
          ("解く・属性・長さの平方根で割る", lambda: unb(full.of, norm="半分")),
          ("表（属性の近さで借りる）", lambda: RL.Relation(full.of, half=HR.HALF)))
    out = {"向き": dict(属性の数=full.n_attr, 話題の帯を除いた属性の数=no_g.n_attr, **KR.coverage(wave_of, full.of, talks))}
    for start, pre in (("白紙から", []), ("青空を先に", aoz)):
        out[start] = JR.run(full.of, talks, pre, mk)
    return out


def main():
    i = sys.argv.index("--content")
    content = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    a, vec_of, talks, aoz, t0 = HR.load()
    board = json.load(open(content, encoding="utf-8"))
    print("用意", round(time.time() - t0), flush=True)
    res = dict(会話=len(talks), 青空の組=len(aoz))
    res.update(run(vec_of, board, talks, aoz))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
