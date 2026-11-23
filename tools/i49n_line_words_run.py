"""I49n: 行の語だけで語の波を作る（品詞の骨組みや席は見ない。持ち主の案「連想では単語をまず充実する」「ひとつの行に含まれる単語だけで」、未確定）。数えだけ。

語の波 = 同じ行に出たほかの語の元の向きを、全部足した物（回さない）。問いは今までと同じ（3 つ組の 1 語を隠し、会話の内容語から選ぶ）で、
見せた 2 語の元の向きを足した物に、作った波が近い候補を答える（席は使わない）。聞きながら育つ形で、答え合わせの後に、その 3 つ組の 3 語を 1 行として足す。
引数は tools/i49m_attr_run.py と同じ（--content 付き）。
始め方: 白紙から ／ 青空の 3 つ組を先に（3 語の組を 1 行とみなす）／ 青空の本当の行を先に（1 文の内容語を 1 行とする。試しの作品も除かずに全部読む。会話は入っていない）。
並べる物（始め方ごと）: 関係の表（比べる元。青空の時は 3 つ組を先に learn）／ 解く・属性（席あり、I49m の形。本当の行の始め方では 3 つ組を先に聞かせる）／
                        行の語・属性 ／ 行の語・語の波。
足して数える物: 行の語の 2 つの形で、会話の内容語が波に持つ「足した語の数」（1 本の波に何個分が乗っているか）の中央値・9 割の所・最大、200 を超える語と 1,000 を超える語の割合。
                青空の本当の行の数と、1 行の内容語の数の平均。
注: 「問いの種類ごと」の数えは、どの始め方でも、会話の中で前に聞いたかどうかだけで分けている（青空で聞いた分は入れていない）。
記録はリポジトリの外。手で置く物: i49h_tally_run と同じ、3 つ組を 1 行とみなす事、割合を出す境の 200 と 1,000（自作の波の試しで、2,048 本なら 200 個で崩れ始め、1,000 個でほぼ崩れた所）。
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


def load_stats(sv, words):
    n = sorted(sv.n_add.get(w, 0) for w in words)
    if not n:
        return {}
    a = np.array(n)
    return dict(語=len(n), 足した語の数の中央値=float(np.median(a)), 九割の所=float(np.percentile(a, 90)), 最大=int(a.max()),
                二百を超える語の割合=round(float((a > 200).mean()), 4), 千を超える語の割合=round(float((a > 1000).mean()), 4),
                波を持つ語の割合=round(float((a > 0).mean()), 4))


def run(wave_of, attr_of, talks, aoz, lines):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    words = sorted({w for _, ws in talks for w in ws}, key=str)
    out = {}
    for start in ("白紙から", "青空の 3 つ組を先に", "青空の本当の行を先に"):
        if start == "青空の本当の行を先に" and lines is None:
            continue
        keep = {}

        def table():
            r = RL.Relation(wave_of, half=HR.HALF)
            for tri in (aoz if start != "白紙から" else ()):
                r.learn(tri)
            return r

        def seat():
            sv = KV.SeatVec(attr_of, weight="数", bind="位相")
            for tri in (aoz if start != "白紙から" else ()):
                sv.hear(tri)
            return KV.Unbind(sv, online=True)

        def line(base, name):
            sv = KV.SeatVec(base, weight="数", bind="なし")
            if start == "青空の 3 つ組を先に":
                for tri in aoz:
                    sv.hear_line(tri)
            elif start == "青空の本当の行を先に":
                for ws in lines:
                    sv.hear_line(ws)
            keep[name] = sv
            return KV.Unbind(sv, online=True)
        mk = (("関係の表", table), ("解く・属性（席あり）", seat),
              ("行の語・属性", lambda: line(attr_of, "行の語・属性")), ("行の語・語の波", lambda: line(wave_of, "行の語・語の波")))
        res = JR.run(attr_of, talks, [], mk)
        for name, sv in keep.items():
            res[name]["波に乗っている数"] = load_stats(sv, words)
        out[start] = res
    return out


def read_lines(conllu, cap):
    import i48_wave_board as W48
    import i49e_triple_count as TC
    lines = []
    for sents in TC.read_conllu(conllu, set(), cap=cap).values():
        for s in sents:
            ws = [(t["l"], t["p"]) for t in s if t["p"] in W48.CONTENT and t["l"] not in W48.STOP]
            if len(ws) >= 2:
                lines.append(ws)
    return lines


def main():
    import i49m_attr_base as AB
    i = sys.argv.index("--content")
    content = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    a, vec_of, talks, aoz, t0 = HR.load()
    full = AB.AttrBase(json.load(open(content, encoding="utf-8")))
    lines = read_lines(a.conllu, a.cap)
    print("用意", round(time.time() - t0), "行", len(lines), flush=True)
    res = dict(会話=len(talks), 青空の組=len(aoz), 青空の行=len(lines), 一行の内容語の数の平均=round(sum(map(len, lines)) / max(1, len(lines)), 2))
    res.update(run(vec_of, full.of, talks, aoz, lines))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
