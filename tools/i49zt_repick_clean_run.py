"""I49zt: 選び直し（もう出た語・薄れる仮の板）を、切り出しを通さない材料で数え直す。

持ち主の問い「材料が悪かった可能性のある実験は過去にある？」から。I49u・I49z・I49zb の選び直しは、崩れた 3 つ組の材料で数えた物だった。
古い切り出しは名前の最後の部分だけを残していたので、同じ語が何度も出て、「もう出た語」が実力より良く見えた恐れが有る。それを確かめる。
材料と話の作り方、並べ替えは tools/i49zp_text_talks_run.py と同じ（材料はリポジトリに入れない）。
形（どれも、答える前に問う）:
  解く形・属性だけ ／ ＋出た語（上位 3） ／ ＋席ごとの薄れる板（九割以内、半分まで 1 組） ／ ＋出た語→板
  数えて選ぶ ／ 数えて選ぶ＋出た語 ／ 数えて選ぶ＋出た語→板
一緒に出す数: 正解が、この話でもう出た語だった問いの割合（同じ席で／どの席でも）と、その内と外での当たり。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --out 記録.json
手で置く物: 上位の数 3、九割、板の薄れる速さ（1 組で半分）、並べ替えの k、後ろ 2 割。
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

import i49zp_text_talks_run as ZP  # noqa: E402

BOARD = dict(by_seat=True, half=1, carry=True)
VERS = (("解く形", "解く", dict(pick="そのまま")),
        ("解く形＋出た語", "解く", dict(pick="出た語", top=3)),
        ("解く形＋席ごとの板", "解く", dict(pick="九割以内", **BOARD)),
        ("解く形＋出た語→板", "解く", dict(pick="出た語→板", top=3, **BOARD)),
        ("数えて選ぶ", "数える", dict(pick="そのまま")),
        ("数えて選ぶ＋出た語", "数える", dict(pick="出た語", top=3)),
        ("数えて選ぶ＋出た語→板", "数える", dict(pick="出た語→板", top=3, **BOARD)))
DUELS = (("解く形", "解く形＋出た語"), ("解く形", "解く形＋席ごとの板"), ("解く形", "解く形＋出た語→板"),
         ("数えて選ぶ", "数えて選ぶ＋出た語"), ("数えて選ぶ", "数えて選ぶ＋出た語→板"))


def seen_flags(seq):
    """問いごとの (正解がこの話の同じ席でもう出た, どの席でも出た)。問いの並びは ZR.stream と同じ。"""
    out = []
    for tris, words in seq:
        had, had_seat = set(), [set(), set(), set()]
        for tri in tris:
            for si in range(3):
                cands = {w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}
                if len(cands) < 2:
                    continue
                out.append((tri[si] in had_seat[si], tri[si] in had))
            for sj, w in enumerate(tri):
                had.add(w)
                had_seat[sj].add(w)
    return out


def run(wave_of, attr_of, talks, ks=(0, 1, 2, 3)):
    import i49e_relation as RL
    import i49h_tally_run as HR
    import i49k_seat_vec as KV
    import i49w_also_right_run as WR
    import i49za_fold_run as ZR
    import i49zk_mix_phase_run as ZK
    from i49u_flow_pick import FlowPick
    from i49zb_mix import Mix
    plain = [(t[0], t[1]) for t in talks]
    ok = WR.ok_sets([t for tris, _ in plain for t in tris])
    out = {}
    for k in ks:
        name_k = "元の順" if k == 0 else "並べ替え %d" % k
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        flags = seen_flags(seq)
        d = dict(問い=len(flags), 正解が同じ席でもう出た=round(sum(f[0] for f in flags) / max(1, len(flags)), 4),
                 正解がどの席でも出た=round(sum(f[1] for f in flags) / max(1, len(flags)), 4))
        keep = {}
        for name, kind, kw in VERS:
            fp = FlowPick(KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True), attr_of, **kw)
            rel = fp if kind == "解く" else Mix(RL.Relation(wave_of, half=HR.HALF), fp, upper=None)
            recs = ZR.stream(rel, seq, ok)
            assert len(recs) == len(flags)
            s = ZR.summary(recs, fp.picked)
            s["後ろ 2 割"] = ZP.tail_score(recs)
            for lab, want in (("正解がもう出た問い", True), ("正解がまだ出ていない問い", False)):
                rs = [r for r, f in zip(recs, flags) if f[1] == want]
                s[lab] = dict(問い=len(rs), 正しい物=round(sum(r[1] for r in rs) / max(1, len(rs)), 4))
            keep[name] = recs
            d[name] = s
            print(name_k, name, s["正しい物"], s["合わせて"], flush=True)
        d["直の比べ"] = {"%s → %s" % (a, b): ZK.duel(keep[a], keep[b]) for a, b in DUELS}
        out[name_k] = d
    return out


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t0 = time.time()
    talks = []
    for f in a.text.split(","):
        talks += ZP.talks_of([l.strip() for l in open(f, encoding="utf-8").read().splitlines()])[0]
    wv = Rw.WordVecs.load(a.wave)

    def wave_of(w):
        v = wv.of(*w)
        return wv.of(w[0], "PROPN") if v is None and w[1] == "NOUN" else v

    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    res = dict(話=len(talks), 三つ組=sum(len(t[0]) for t in talks))
    res.update(run(wave_of, full.of, talks))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("話", "三つ組", "秒")}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
