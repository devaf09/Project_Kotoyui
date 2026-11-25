"""I49zh: いいとこ取りの形（数えて選ぶ、I49zc）の解く側に、振幅を誤差で直す形（I49zg）を入れて数える。数えだけ。

I49zg で、解く形だけなら振幅を直すと 1〜2 ポイント上がった。表と組んだ時にも積み上がるかを見る。
振幅を誤差で直すのは、今までの決まり（数えて足すだけ）から外れる試し（持ち主の案の第 1 段、未確定）。
並べる物:
  関係の表だけ
  数えて選ぶ ／ 数えて選ぶ＋出た語・上位 3            （I49zc と同じ。比べ役）
  数えて選ぶ・振幅を直す（毎回・席ごと・速さ .01） ／ 同じ＋出た語・上位 3
  数えて選ぶ・振幅を直す（外れた時・席ごと・速さ .01） ／ （毎回・席ごと・速さ .003）   速さと直し方を変えても同じ向きかを見る
採点は、正しい物と、合わせて。関係の表だけから 当たりに変わった数・外れに変わった数、どちらを使ったか、段ごとの当たり、振幅の広がりも出す。
流れ・引数・記録の書き方は tools/past/i49zg_amp_run.py と同じ（版ごとに書き、出ている版は飛ばす。関係の表だけは比べの元なので数え直す）。
手で置く物: 直す速さ、temp .05、上位 3、同じ数の時に表を使う事。
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

BASE = "関係の表だけ"
SAID = dict(pick="出た語", top=3)
A1 = dict(rule="毎回", lr=.01, by_seat=True)
VERS = [(BASE, None, None),
        ("数えて選ぶ", None, dict(pick="そのまま")),
        ("数えて選ぶ＋出た語", None, SAID),
        ("数えて選ぶ・振幅を直す（毎回・席ごと・.01）", A1, dict(pick="そのまま")),
        ("数えて選ぶ・振幅を直す（毎回・席ごと・.01）＋出た語", A1, SAID),
        ("数えて選ぶ・振幅を直す（外れた時・席ごと・.01）", dict(rule="外れた時", lr=.01, by_seat=True), dict(pick="そのまま")),
        ("数えて選ぶ・振幅を直す（毎回・席ごと・.003）", dict(rule="毎回", lr=.003, by_seat=True), dict(pick="そのまま"))]


def run(wave_of, attr_of, talks, aoz, done=None, save=None):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    from i49u_flow_pick import FlowPick
    from i49zb_mix import Mix
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
            table = RL.Relation(wave_of, half=HR.HALF)
            fp = inner = None
            if pick is not None:
                sv = KV.SeatVec(attr_of, weight="数", bind="位相")
                inner = AmpRead(sv, **amp) if amp else KV.Unbind(sv, online=True)
                fp = FlowPick(inner, attr_of, **pick)
            for tri in pre:
                table.learn(tri)
                if inner is not None:
                    inner.sv.hear(tri)
            rel = table if fp is None else Mix(table, fp, upper=None)
            recs = ZR.stream(rel, talks, ok)
            d = ZR.summary(recs, fp.picked if fp is not None else 0, base)
            base = recs if base is None else base
            if fp is not None:
                d["使った答え"] = dict(rel.used)
                d["段ごとの当たり"] = {lv: dict(表=h[0], 解く=h[1], 問い=h[2]) for lv, h in rel.tally.items()}
            if amp:
                d["振幅"] = inner.spread()
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
    save(run(vec_of, full.of, talks, aoz, done, save))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
