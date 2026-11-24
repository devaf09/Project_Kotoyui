"""I49za: 行の折り畳みの決まり（ふだんより明るい物だけ残す）を、仮の板に入れる（持ち主の案、未確定）。数えだけ。

持ち主の言葉:「前作った行の折り畳みもつかえるといいんだけどなぁ」
記憶板（I40b）は、上の段に畳む時に「ふだんの見込みより明るい鍵」だけを残し、残りは半分に薄める。同じ決まりを、鍵を属性に替えて板に入れる。
ふだん = 今までに聞いた全部の語の属性の平均（席ごとの板なら、その席の平均）。先に聞かせた青空の 3 つ組も、ふだんに入れる。
並べる物（解く・属性（席あり）の答えを、板に一番近い候補に替える）:
  選び方: 九割以内 ／ 上位 3   板: 1 枚 ／ 席ごと   薄れ: 半分まで 1・2・4 組
  ふだんとの比べ方: 比べない ／ 明るい分だけ ／ 残す・ほかは半分 ／ 引く
採点は、正しい物と、合わせて（I49w の集まり、話と青空）。板無しから 当たりに変わった数・外れに変わった数 も出す。
流れは tools/i49h_tally_run.py と同じ。引数は tools/i49m_attr_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: 薄れる速さ、九割、上位 3、「ほかは半分」の半分。
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

SEATS = ("だれが", "何を", "どうした")
USUAL = (("比べない", None), ("明るい分だけ", "明るい分だけ"), ("残す・ほかは半分", "残す・ほかは半分"), ("引く", "引く"))
VERS = [("板無し", dict(pick="そのまま"))]
for pick, pname in (("九割以内", "九割以内"), ("上位", "上位 3")):
    for seat, sname in ((False, "1 枚"), (True, "席ごと")):
        for h in (1, 2, 4):
            for uname, u in USUAL:
                VERS.append(("%s｜%s｜半分まで %d 組｜%s" % (pname, sname, h, uname),
                             dict(pick=pick, top=3, by_seat=seat, half=h, carry=True, usual=u)))


def stream(fp, talks, ok):
    recs = []
    for tris, words in talks:
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                q = tuple(None if j == si else tri[j] for j in range(3))
                a = fp.ask(si, q, cands)["答え"]
                r = a == tri[si]
                recs.append((si, r, a is not None and (r or a in ok.get((si, q), ()))))
            fp.learn(tri)
        fp.tick()
    return recs


def summary(recs, picked, base=None):
    d = max(1, len(recs))
    out = dict(正しい物=round(sum(r[1] for r in recs) / d, 4), 合わせて=round(sum(r[2] for r in recs) / d, 4), 答えが変わった数=picked,
               席ごと={s: round(sum(r[1] for r in recs if r[0] == i) / max(1, sum(1 for r in recs if r[0] == i)), 4)
                     for i, s in enumerate(SEATS)})
    if base is not None:
        out["当たりに変わった数"] = sum(1 for b, r in zip(base, recs) if not b[1] and r[1])
        out["外れに変わった数"] = sum(1 for b, r in zip(base, recs) if b[1] and not r[1])
    return out


def run(attr_of, talks, aoz):
    import i49k_seat_vec as KV
    from i49u_flow_pick import FlowPick
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out[start], base = {}, None
        for name, kw in VERS:
            u = KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True)
            fp = FlowPick(u, attr_of, **kw)
            for tri in pre:
                u.sv.hear(tri)
                if kw.get("usual"):
                    fp.hear_usual(tri)
            recs = stream(fp, talks, ok)
            out[start][name] = summary(recs, fp.picked, base)
            base = recs if base is None else base
            print(start, name, out[start][name]["正しい物"], out[start][name]["合わせて"], flush=True)
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
