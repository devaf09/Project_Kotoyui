"""I49zk: 位相と振幅を一緒に直す解く形（I49zj の一番上）を、いいとこ取りの形（数えて選ぶ）と、出た語の選び直しに重ねて数える。数えだけ。

I49zh では、振幅だけを直した分は、表と組むと消えた。位相も動かした形で同じ事が起きるかを見る。
位相と振幅を誤差で直すのは、今までの決まり（数えて足すだけ）から外れる試し（持ち主の案、未確定）。
並べる物:
  関係の表だけ
  解く形だけ: 直さない ／ 振幅だけ ／ 位相＋振幅 ／ 位相＋振幅＋出た語・上位 3
  数えて選ぶ（表と解く形を段ごとに数えて選ぶ）: 直さない ／ 位相＋振幅 ／ 位相＋振幅＋出た語・上位 3
  位相＋振幅 = 位相の速さ .1、合図 聞いた組、振幅は 毎回・席ごと・.01。
採点は、正しい物と、合わせて。関係の表だけから 当たりに変わった数・外れに変わった数、段ごとの当たり、位相と振幅の動きも出す。
直の比べ（同じ問いで、片方だけ当たった数）も出す: 解く形の 振幅だけ 対 位相＋振幅、数えて選ぶの 直さない 対 位相＋振幅。
流れ・引数は tools/past/i49zh_mix_amp_run.py と同じ。版が 1 つ出るたびに記録を書く（続きからは数えない。軽いので初めから数え直す）。
手で置く物: 直す速さ、temp .05、上位 3、同じ数の時に表を使う事。
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
import i49w_also_right_run as WR  # noqa: E402
import i49za_fold_run as ZR  # noqa: E402

BASE = "関係の表だけ"
AMP = dict(phase_lr=0.0, rule="毎回", lr=.01, by_seat=True)
BOTH = dict(phase_lr=.1, teach="聞いた組", rule="毎回", lr=.01, by_seat=True)
SAID = dict(pick="出た語", top=3)
KEEP = dict(pick="そのまま")
VERS = [(BASE, "表", None, None),
        ("解く形だけ｜直さない", "解く", None, KEEP),
        ("解く形だけ｜振幅だけ", "解く", AMP, KEEP),
        ("解く形だけ｜位相＋振幅", "解く", BOTH, KEEP),
        ("解く形だけ｜位相＋振幅＋出た語", "解く", BOTH, SAID),
        ("数えて選ぶ｜直さない", "数える", None, KEEP),
        ("数えて選ぶ｜位相＋振幅", "数える", BOTH, KEEP),
        ("数えて選ぶ｜位相＋振幅＋出た語", "数える", BOTH, SAID)]
DUELS = [("解く形だけ｜振幅だけ", "解く形だけ｜位相＋振幅"), ("解く形だけ｜直さない", "解く形だけ｜位相＋振幅"),
         ("数えて選ぶ｜直さない", "数えて選ぶ｜位相＋振幅"), ("数えて選ぶ｜直さない", "数えて選ぶ｜位相＋振幅＋出た語")]


def duel(a, b):
    """同じ問いで、b だけ当たった数・a だけ当たった数・z（符号の比べ）。"""
    up = sum(1 for x, y in zip(a, b) if not x[1] and y[1])
    dn = sum(1 for x, y in zip(a, b) if x[1] and not y[1])
    return dict(良くなった=up, 悪くなった=dn, z=round((up - dn) / max(1.0, (up + dn) ** 0.5), 2))


def run(wave_of, attr_of, talks, aoz, save=None):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    from i49u_flow_pick import FlowPick
    from i49zb_mix import Mix
    from i49zj_phase_read import PhaseRead
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out[start], base, keep = {}, None, {}
        for name, kind, read, pick in VERS:
            t1 = time.time()
            table = RL.Relation(wave_of, half=HR.HALF)
            fp = inner = None
            if kind != "表":
                sv = KV.SeatVec(attr_of, weight="数", bind="位相")
                inner = PhaseRead(sv, **read) if read else KV.Unbind(sv, online=True)
                fp = FlowPick(inner, attr_of, **pick)
            for tri in pre:
                if kind != "解く":
                    table.learn(tri)
                if inner is not None:
                    inner.sv.hear(tri)
                    if read:
                        inner.hear_seen(tri)
            rel = table if kind == "表" else fp if kind == "解く" else Mix(table, fp, upper=None)
            recs = ZR.stream(rel, talks, ok)
            d = ZR.summary(recs, fp.picked if fp is not None else 0, base)
            base = recs if base is None else base
            keep[name] = recs
            if kind == "数える":
                d["使った答え"] = dict(rel.used)
                d["段ごとの当たり"] = {lv: dict(表=h[0], 解く=h[1], 問い=h[2]) for lv, h in rel.tally.items()}
            if read:
                d["動き"] = inner.spread()
            d["秒"] = round(time.time() - t1)
            out[start][name] = d
            if save:
                save(out)
            print(start, name, d["正しい物"], d["合わせて"], flush=True)
        out[start]["直の比べ"] = {"%s → %s" % (a, b): duel(keep[a], keep[b]) for a, b in DUELS}
        if save:
            save(out)
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

    save(run(vec_of, full.of, talks, aoz, save))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
