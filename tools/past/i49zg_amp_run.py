"""I49zg: 読む時の振幅だけを誤差で直す形を数える（持ち主の案の第 1 段、未確定の試し）。数えだけ。

波（周波数、位相、覚えた波）は固定。周波数ごとの読む時の振幅だけを、答え合わせの後に直す（intervention/i49zg_amp_read.py）。
並べる物（解く・属性（席あり））:
  振幅を直さない（今までの形） ／ 出た語・上位 3（比べ役）
  外れた時だけ直す × 速さ .003・.01・.03 × 振幅 1 組・席ごと
  毎回直す（ソフトマックス） × 速さ .01・.03 × 振幅 1 組・席ごと
  外れた時だけ・席ごと・.01 ＋ 出た語・上位 3
採点は、正しい物と、合わせて（I49w の集まり、話と青空）。振幅を直さない形から 当たりに変わった数・外れに変わった数、振幅の広がりも出す。
聞く前に問う 1 周だけ（何周も回さない）。青空の 3 つ組を先に聞かせる時は、覚えるだけで、振幅は直さない。
流れは tools/i49h_tally_run.py と同じ。引数は tools/i49m_attr_run.py と同じ（--content 付き）。記録はリポジトリの外。
版が 1 つ出るたびに記録を書く。記録が既に有れば、出ている版は飛ばす（振幅を直さない形だけは比べの元なので数え直す）。
手で置く物: 直す速さ、temp .05、上位 3。
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
import i49za_fold_run as ZR  # noqa: E402

BASE = "振幅を直さない"
VERS = [(BASE, None, None), ("出た語・上位 3", None, dict(pick="出た語", top=3))]
for rule, lrs in (("外れた時", (.003, .01, .03)), ("毎回", (.01, .03))):
    for seat, sname in ((False, "振幅 1 組"), (True, "席ごと")):
        for lr in lrs:
            VERS.append(("%s｜%s｜速さ %s" % (rule, sname, ("%g" % lr).lstrip("0")), dict(rule=rule, lr=lr, by_seat=seat), None))
VERS.append(("外れた時｜席ごと｜速さ .01｜＋出た語・上位 3", dict(rule="外れた時", lr=.01, by_seat=True), dict(pick="出た語", top=3)))


def run(attr_of, talks, aoz, done=None, save=None):
    import i49k_seat_vec as KV
    from i49u_flow_pick import FlowPick
    from i49zg_amp_read import AmpRead
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = dict(done or {})
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out.setdefault(start, {})
        base = None
        for name, amp, pick in VERS:
            if name in out[start] and name != BASE:
                continue
            t1 = time.time()
            sv = KV.SeatVec(attr_of, weight="数", bind="位相")
            for tri in pre:
                sv.hear(tri)
            rel = AmpRead(sv, **amp) if amp else KV.Unbind(sv, online=True)
            fp = FlowPick(rel, attr_of, **(pick or dict(pick="そのまま")))
            recs = ZR.stream(fp, talks, ok)
            d = ZR.summary(recs, fp.picked, base)
            base = recs if base is None else base
            if amp:
                d["振幅"] = rel.spread()
            d["秒"] = round(time.time() - t1)
            out[start][name] = d
            if save:
                save(out)
            print(start, name, d["正しい物"], d["合わせて"], flush=True)
    return out


def main():
    import i49m_attr_base as AB
    i = sys.argv.index("--content")
    content = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    a, vec_of, talks, aoz, t0 = HR.load()
    full = AB.AttrBase(json.load(open(content, encoding="utf-8")))

    def save(out):
        res = dict(会話=len(talks), 青空の組=len(aoz), 秒=round(time.time() - t0))
        res.update(out)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    done = None
    if os.path.exists(a.out):
        old = json.load(open(a.out, encoding="utf-8"))
        done = {k: v for k, v in old.items() if isinstance(v, dict)}
    save(run(full.of, talks, aoz, done, save))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
