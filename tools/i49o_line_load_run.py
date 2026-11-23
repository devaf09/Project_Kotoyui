"""I49o: 行の語だけで作る波（I49n）で、乗りすぎの分と、材料が合わない分を分けて見る。数えだけ。

I49n では、青空の本当の行を先に聞かせると大きく下がった。ただ、乗る語の数と材料の種類が一緒に変わっていて、どちらのせいかは分からなかった。
ここでは 2 つを動かす。
  聞かせる量: 本当の行を 64 行に 1 行、16 行に 1 行、4 行に 1 行、全部（並びの順に飛ばして取る。乱数は使わない）。
              少ない量でも低いなら材料のせい、量が増えるほど下がるなら乗りすぎのせい、と読める。
  乗りすぎを抑える形: そのまま ／ たまたまの分を引く（center）／ よく出る語を軽くして足す（light）／ 両方。どれも足し算と数えだけ。
元の向きは、属性と語の波の 2 通り。始め方は、白紙から、青空の 3 つ組を先に、本当の行の 4 つの量。
引数は tools/i49n_line_words_run.py と同じ。記録はリポジトリの外。
手で置く物: 量の 4 段（64・16・4・1 行に 1 行）、軽くする重さの形 log(1 + 聞いた行の数 ÷ その語が出た行の数)、ほかは I49n と同じ。
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

FORMS = (("そのまま", False, False), ("引く", True, False), ("軽く", False, True), ("引く・軽く", True, True))
STEPS = (64, 16, 4, 1)


def run(wave_of, attr_of, talks, aoz, lines, steps=STEPS):
    import i49k_seat_vec as KV
    words = sorted({w for _, ws in talks for w in ws}, key=str)
    starts = [("白紙から", ()), ("青空の 3 つ組を先に", aoz)]
    starts += [("本当の行 全部" if k == 1 else "本当の行 %d 行に 1 行" % k, lines[::k]) for k in steps]
    out = {}
    for start, pre in starts:
        keep, mk = {}, []
        for bname, base in (("属性", attr_of), ("語の波", wave_of)):
            for fname, center, light in FORMS:
                name = "行の語・%s・%s" % (bname, fname)

                def make(base=base, center=center, light=light, name=name):
                    sv = KV.SeatVec(base, weight="数", bind="なし", center=center, light=light)
                    for ws in pre:
                        sv.hear_line(ws)
                    keep[name] = sv
                    return KV.Unbind(sv, online=True)
                mk.append((name, make))
        res = JR.run(attr_of, talks, [], mk)
        for name, sv in keep.items():
            res[name]["波に乗っている数"] = NR.load_stats(sv, words)
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
