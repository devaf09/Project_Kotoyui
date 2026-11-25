"""I49zj: 本番の流れで、語の位相を誤差で動かす形を数える（持ち主の案の第 2 段、未確定の試し）。数えだけ。

intervention/i49zj_phase_read.py の PhaseRead。覚えた波は作り直さない。聞く前に問う 1 周だけ。
並べる物（解く・属性（席あり））:
  直さない（今までの形） ／ 振幅だけ（毎回・席ごと・.01、I49zg の一番上）
  位相を動かす × 速さ .03・.1・.3 × 合図 隠した語・聞いた組
  位相（速さ .1）＋振幅（毎回・席ごと・.01） × 合図 隠した語・聞いた組
採点は、正しい物と、合わせて。直さない形から 当たりに変わった数・外れに変わった数、位相と振幅の動きも出す。
青空の 3 つ組を先に聞かせる時は、覚えるだけで、位相も振幅も直さない（聞いた組の数えには入れる）。
流れ・引数・記録の書き方は tools/past/i49zg_amp_run.py と同じ（版ごとに書き、出ている版は飛ばす。直さない形だけは比べの元なので数え直す）。
手で置く物: 直す速さ、temp .05。
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

BASE = "直さない"
AMP = dict(rule="毎回", lr=.01, by_seat=True)
VERS = [(BASE, None), ("振幅だけ（毎回・席ごと・.01）", dict(phase_lr=0.0, **AMP))]
for teach in ("隠した語", "聞いた組"):
    for lr in (.03, .1, .3):
        VERS.append(("位相｜合図 %s｜速さ %s" % (teach, ("%g" % lr).lstrip("0")), dict(phase_lr=lr, teach=teach)))
for teach in ("隠した語", "聞いた組"):
    VERS.append(("位相＋振幅｜合図 %s｜速さ .1" % teach, dict(phase_lr=.1, teach=teach, **AMP)))


def run(attr_of, talks, aoz, done=None, save=None):
    import i49k_seat_vec as KV
    from i49zj_phase_read import PhaseRead
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = dict(done or {})
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out.setdefault(start, {})
        base = None
        for name, kw in VERS:
            if name in out[start] and name != BASE:
                continue
            t1 = time.time()
            sv = KV.SeatVec(attr_of, weight="数", bind="位相")
            rel = PhaseRead(sv, **kw) if kw else KV.Unbind(sv, online=True)
            for tri in pre:
                sv.hear(tri)
                if kw:
                    rel.hear_seen(tri)
            recs = ZR.stream(rel, talks, ok)
            d = ZR.summary(recs, 0, base)
            base = recs if base is None else base
            if kw:
                d["動き"] = rel.spread()
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
