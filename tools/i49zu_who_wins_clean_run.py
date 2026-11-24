"""I49zu: 「正解が上位に居るのに、属性の似た語に負ける」の数え（I49y・I49zl・I49zn）を、切り出しを通さない材料で数え直す。

持ち主の問い「材料が悪かった可能性のある実験は過去にある？」から。古い材料では、正解そのものが崩れていた恐れが有る。
数えは tools/i49zn_who_wins_two_run.py の run をそのまま使う（3 つの形: 属性だけ ／ 属性＋固有の位相 ／ ＋振幅）。先に聞かせる材料は無し。
材料と話の作り方、並べ替えは tools/i49zp_text_talks_run.py と同じ（材料はリポジトリに入れない）。
引数: --text 材料（カンマ区切り） --content 内容語の板 --out 記録.json
手で置く物: i49zn と同じ。並べ替えの k（0 と 1）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49zn_who_wins_two_run as ZN  # noqa: E402
import i49zp_text_talks_run as ZP  # noqa: E402


def run(attr_of, talks, ks=(0, 1)):
    plain = [(t[0], t[1]) for t in talks]
    out = {}
    for k in ks:
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        out["元の順" if k == 0 else "並べ替え %d" % k] = ZN.run(attr_of, seq, ())["白紙から"]
    return out


def main():
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "content", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t0 = time.time()
    talks = []
    for f in a.text.split(","):
        talks += ZP.talks_of([l.strip() for l in open(f, encoding="utf-8").read().splitlines()])[0]
    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    res = dict(話=len(talks), 三つ組=sum(len(t[0]) for t in talks))
    res.update(run(full.of, talks))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
