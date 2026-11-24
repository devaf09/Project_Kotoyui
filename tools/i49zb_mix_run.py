"""I49zb: いいとこ取りの形（intervention/i49zb_mix.Mix）を、同じ流れで数える。数えだけ。

並べる物:
  関係の表だけ ／ 解く・属性（席あり）だけ
  表 → 解く                表が上の段で答えられない問いだけ、解く形の答え
  表 → 解く＋出た語         解く形の答えを、上位 3 の中でこの話で一番最近に出た語に替える（I49u）
  表 → 解く＋出た語（九割以内） 同じ事を、一位と並ぶ候補の中だけでする
  表 → 解く＋板            一位と並ぶ候補の中で、席ごとの薄れる板（1 組で半分）に一番近い語（I49z）
  表 → 解く＋出た語→板      出た語が有ればそれ、無ければ板
  解く＋出た語→板          表を使わず、解く形と選び直しだけ
  数えて選ぶ（I49zc）       表に任せる段を手で置かず、段ごとにそれまで多く当たった方を使う。選び直し無し ／ 出た語 ／ 出た語→板
採点は、正しい物と、合わせて（I49w の集まり、話と青空）。関係の表だけから 当たりに変わった数・外れに変わった数、表と解く形のどちらを使ったかの数も出す。
流れは tools/i49h_tally_run.py と同じ。引数は tools/i49m_attr_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: Mix と FlowPick の物（表に任せる段、上位 3、九割、薄れる速さ 1 組）。
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

BOARD = dict(by_seat=True, half=1, carry=True)
VERS = (("関係の表だけ", "表", None),
        ("解く・属性（席あり）だけ", "解く", dict(pick="そのまま")),
        ("表 → 解く", "両方", dict(pick="そのまま")),
        ("表 → 解く＋出た語", "両方", dict(pick="出た語", top=3)),
        ("表 → 解く＋出た語（九割以内）", "両方", dict(pick="出た語・九割以内")),
        ("表 → 解く＋板", "両方", dict(pick="九割以内", **BOARD)),
        ("表 → 解く＋出た語→板", "両方", dict(pick="出た語→板", top=3, **BOARD)),
        ("解く＋出た語→板", "解く", dict(pick="出た語→板", top=3, **BOARD)),
        ("数えて選ぶ", "数える", dict(pick="そのまま")),
        ("数えて選ぶ＋出た語", "数える", dict(pick="出た語", top=3)),
        ("数えて選ぶ＋出た語→板", "数える", dict(pick="出た語→板", top=3, **BOARD)))


def run(wave_of, attr_of, talks, aoz):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    from i49u_flow_pick import FlowPick
    from i49zb_mix import Mix
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out[start], base = {}, None
        for name, kind, kw in VERS:
            table = RL.Relation(wave_of, half=HR.HALF)
            fp = None
            if kind != "表":
                u = KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True)
                fp = FlowPick(u, attr_of, **kw)
            for tri in pre:
                if kind != "解く":
                    table.learn(tri)
                if fp is not None:
                    fp.rel.sv.hear(tri)
            rel = table if kind == "表" else fp if kind == "解く" else Mix(table, fp) if kind == "両方" else Mix(table, fp, upper=None)
            recs = ZR.stream(rel, talks, ok)
            out[start][name] = ZR.summary(recs, fp.picked if fp is not None else 0, base)
            if kind in ("両方", "数える"):
                out[start][name]["使った答え"] = dict(rel.used)
            if kind == "数える":
                out[start][name]["段ごとの当たり"] = {lv: dict(表=h[0], 解く=h[1], 問い=h[2]) for lv, h in rel.tally.items()}
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
    res.update(run(vec_of, full.of, talks, aoz))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
