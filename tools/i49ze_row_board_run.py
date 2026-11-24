"""I49ze: 仮の板の語を、足さずに 1 本ずつ並べて、一斉に走査する（持ち主の言葉「並べて一斉に走査もOK」から。未確定の試し）。数えだけ。

I49zd では、語の波を 1 本に足した板は、決まりで書いた「出た語」に届かなかった。足した波との近さでは、似た語にも点が入るためと見た。
ここでは、聞いた語の波を並べて持ち、重さだけを薄れさせる。近さは 重さ × 候補との近さ の一番大きい物。
並べる物（解く・属性（席あり）の答えを、板に一番近い候補に替える）:
  板無し ／ 出た語・上位 3（決まりで書いた比べ役） ／ 足した板・九割以内・席ごと・半分まで 1 組（I49zd の一番上）
  選び方: 九割以内 ／ 上位 3   板: 1 枚 ／ 席ごと   薄れ: 半分まで 1・2・4 組（話をまたぐ） ／ 薄れない（話ごとに空）
採点は、正しい物と、合わせて（I49w の集まり、話と青空）。板無しから 当たりに変わった数・外れに変わった数 も出す。
流れは tools/i49h_tally_run.py と同じ。引数は tools/i49m_attr_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: 薄れる速さ、九割、上位 3。
版が 1 つ出るたびに記録を書く。記録が既に有れば、出ている版は飛ばして続きから数える（板無しだけは比べの元なので数え直す）。
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

VERS = [("板無し", "属性", dict(pick="そのまま")), ("出た語・上位 3", "属性", dict(pick="出た語", top=3)),
        ("足した板｜九割以内｜席ごと｜半分まで 1 組", "語の波", dict(pick="九割以内", top=3, by_seat=True, half=1, carry=True))]
for pick, pname in (("九割以内", "九割以内"), ("上位", "上位 3")):
    for seat, sname in ((False, "1 枚"), (True, "席ごと")):
        for hname, kw in [("半分まで %d 組" % h, dict(half=h, carry=True)) for h in (1, 2, 4)] + [("薄れない", dict())]:
            VERS.append(("並べた板｜%s｜%s｜%s" % (pname, sname, hname), "語の波", dict(pick=pick, top=3, by_seat=seat, board="並べる", **kw)))


def run(wave_of, attr_of, talks, aoz, done=None, save=None):
    import i49k_seat_vec as KV
    from i49u_flow_pick import FlowPick
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = dict(done or {})                                  # done: 前に出た分（途中で止めた時の続き）
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out.setdefault(start, {})
        base = None
        for name, board, kw in VERS:
            if name in out[start] and name != "板無し":
                continue
            t1 = time.time()
            u = KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True)
            for tri in pre:
                u.sv.hear(tri)
            fp = FlowPick(u, wave_of if board == "語の波" else attr_of, **kw)
            recs = ZR.stream(fp, talks, ok)
            out[start][name] = ZR.summary(recs, fp.picked, base)
            base = recs if base is None else base
            out[start][name]["秒"] = round(time.time() - t1)
            if save:
                save(out)
            print(start, name, out[start][name]["正しい物"], out[start][name]["合わせて"], flush=True)
    return out


def main():
    import i49m_attr_base as AB
    i = sys.argv.index("--content")
    content = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    a, vec_of, talks, aoz, t0 = HR.load()
    full = AB.AttrBase(json.load(open(content, encoding="utf-8")))
    def save(out):                                           # 版が 1 つ出るたびに書く（途中で止めても残る）
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
