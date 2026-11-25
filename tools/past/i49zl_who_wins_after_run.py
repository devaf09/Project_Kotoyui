"""I49zl: 誤差で直した後の解く形で、「正解が上位に居るのに、属性の似た語に負ける」問いがどれだけ残っているかを数える。数えだけ。

持ち主の言葉:「属性が近いとわからなくなる問題はまだ残ってるよね？」
I49y と同じ数え（tools/past/i49y_who_wins_run.py の who）を、3 つの形で並べる:
  直さない ／ 振幅だけ直す（毎回・席ごと・.01） ／ 位相＋振幅を直す（位相 .1・合図 聞いた組）
数える物（正解が 2〜3 位だった問い）: その数と全体に占める割合、1 位が正しいと言える物だった割合、
  1 位と正解の属性の近さの平均、正解とほかの候補の近さの平均、席ごとの内訳。
近さは、動かす前の属性の波どうしで測る（位相を動かす形でも、物差しは同じ）。
  初めの版は動かした後の波で測っていて、位相を動かす形だけ近さが全体に下がり、比べられなかった。それを直した。
流れ・引数は tools/past/i49y_who_wins_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: 「上位」を 3 位までとする事、直す速さ、temp .05。
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
import i49w_also_right_run as WR  # noqa: E402
import i49y_who_wins_run as Y  # noqa: E402

AMP = dict(phase_lr=0.0, rule="毎回", lr=.01, by_seat=True)
BOTH = dict(phase_lr=.1, teach="聞いた組", rule="毎回", lr=.01, by_seat=True)
VERS = (("直さない", None), ("振幅だけ直す", AMP), ("位相＋振幅を直す", BOTH))


def attr_cos(attr_of):
    """動かす前の属性の波どうしの近さ（I49y の _cos と同じ形。sv は見ない）。"""
    import numpy as np
    unit = {}

    def u(w):
        if w not in unit:
            v = attr_of(w)
            if v is not None:
                v = np.asarray(v, dtype=np.complex64)
                n = float(np.linalg.norm(v))
                v = v / n if n > 0 else None
            unit[w] = v
        return unit[w]

    def cos(sv, a, b):
        x, y = u(a), u(b)
        return None if x is None or y is None else float(np.real(np.vdot(x, y)))
    return cos


def run(attr_of, talks, aoz):
    old, Y._cos = Y._cos, attr_cos(attr_of)
    try:
        return _run(attr_of, talks, aoz)
    finally:
        Y._cos = old


def _run(attr_of, talks, aoz):
    import i49k_seat_vec as KV
    from i49zj_phase_read import PhaseRead
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out[start] = {}
        for name, kw in VERS:
            sv = KV.SeatVec(attr_of, weight="数", bind="位相")
            rel = PhaseRead(sv, **kw) if kw else KV.Unbind(sv, online=True)
            for tri in pre:
                sv.hear(tri)
                if kw:
                    rel.hear_seen(tri)
            d = Y.who(rel, talks, ok)
            lost = d["正解が二〜三位だった問い"]["全体"]["問い"]
            d["正解が二〜三位だった割合"] = round(lost / max(1, d["全部の問い"]), 4)
            d["正解が一位だった割合"] = round(d["正解が一位だった問い"]["問い"] / max(1, d["全部の問い"]), 4)
            out[start][name] = d
            print(start, name, d["正解が二〜三位だった割合"], d["正解が二〜三位だった問い"]["全体"], flush=True)
    return out


def main():
    import i49m_attr_base as AB
    i = sys.argv.index("--content")
    content = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    a, vec_of, talks, aoz, t0 = HR.load()
    full = AB.AttrBase(json.load(open(content, encoding="utf-8")))
    res = dict(会話=len(talks), 青空の組=len(aoz))
    res.update(run(full.of, talks, aoz))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
