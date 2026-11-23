"""I49p: 語の波そのものを聞くたびに動かす形（LiveVec）を、今までの形と同じ問いで比べる。数えだけ。

引数は tools/i49n_line_words_run.py と同じ（--content 付き）。記録はリポジトリの外。
始め方: 白紙から ／ 青空の 3 つ組を先に ／ 青空の本当の行を 16 行に 1 行だけ先に ／ 青空の本当の行を全部先に。
並べる物: 関係の表（比べる元）／ 解く・属性（席あり、I49m の形）／ 行の語・属性（I49n の形。元の向きを足す）／
          今の波・属性 ／ 今の波・属性・遠ざける ／ 今の波・属性・席 ／ 今の波・属性・席・遠ざける ／ 今の波・語の波 ／ 今の波・語の波・遠ざける。
          「席」の付く形は、3 つ組は席の回し方で聞かせる。本当の行は席が無いので、行の語として聞かせる。
足して数える物: 今の波の形で、会話の内容語が持つ「足した語の数」（I49n と同じ数え）。
手で置く物: i49n と同じ、本当の行を飛ばして取る数（16）。
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
import i49n_line_words_run as NR  # noqa: E402

LIVE = (("今の波・属性", "属性", False, "なし"), ("今の波・属性・遠ざける", "属性", True, "なし"),
        ("今の波・属性・席", "属性", False, "位相"), ("今の波・属性・席・遠ざける", "属性", True, "位相"),
        ("今の波・語の波", "語の波", False, "なし"), ("今の波・語の波・遠ざける", "語の波", True, "なし"))


def run(wave_of, attr_of, talks, aoz, lines, step=16):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    import i49p_live_vec as PV
    words = sorted({w for _, ws in talks for w in ws}, key=str)
    starts = [("白紙から", "無し", ()), ("青空の 3 つ組を先に", "組", aoz)]
    if lines is not None:
        starts += [("本当の行 %d 行に 1 行" % step, "行", lines[::step]), ("本当の行 全部", "行", lines)]
    out = {}
    for start, kind, pre in starts:
        keep = {}

        def table():
            r = RL.Relation(wave_of, half=HR.HALF)
            for tri in (aoz if kind != "無し" else ()):
                r.learn(tri)
            return r

        def seat():
            sv = KV.SeatVec(attr_of, weight="数", bind="位相")
            for tri in (aoz if kind != "無し" else ()):
                sv.hear(tri)
            return KV.Unbind(sv, online=True)

        def line():
            sv = KV.SeatVec(attr_of, weight="数", bind="なし")
            for ws in pre:
                sv.hear_line(ws)
            return KV.Unbind(sv, online=True)
        mk = [("関係の表", table), ("解く・属性（席あり）", seat), ("行の語・属性", line)]
        for name, bname, far, bind in LIVE:
            def make(name=name, bname=bname, far=far, bind=bind):
                lv = PV.LiveVec(attr_of if bname == "属性" else wave_of, far=far, bind=bind)
                for ws in pre:
                    (lv.hear if kind == "組" else lv.hear_line)(ws)
                keep[name] = lv
                return PV.LiveAsk(lv)
            mk.append((name, make))
        res = JR.run(attr_of, talks, [], mk)
        for name, lv in keep.items():
            res[name]["波に乗っている数"] = NR.load_stats(lv, words)
        res["聞かせた行の数"] = len(pre)
        out[start] = res
        print(start, flush=True)
    return out


def main():
    import i49m_attr_base as AB
    i = sys.argv.index("--content")
    content = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    a, vec_of, talks, aoz, t0 = HR.load()
    full = AB.AttrBase(json.load(open(content, encoding="utf-8")))
    lines = NR.read_lines(a.conllu, a.cap)
    print("用意", round(time.time() - t0), "行", len(lines), flush=True)
    res = dict(会話=len(talks), 青空の組=len(aoz), 青空の行=len(lines))
    res.update(run(vec_of, full.of, talks, aoz, lines))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
