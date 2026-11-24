"""I49zw: 2 語の組を「広げてから掛ける」束ね方を、切り出しを通さない材料で数える（持ち主がもらった助言から）。

まず数える物（理由を見るため）: 属性の波がどれだけまばらか（属性の総数、語ごとの属性の数）、3 つ組の中の 2 語ずつで
  共通の属性がいくつ有るか、今までの掛け算（周波数ごと）で残る長さ、何も残らない組の割合。
形（どれも、答える前に問う）:
  解く形 ／ 解く形＋今までの束ね（SeatVec(pair=True)） ／ 組だけ・まばら ／ 組だけ・広げる ／ 解く形＋組・広げる（点を 1 対 1 で足す）
  数えて選ぶ ／ 数えて選ぶ・組と（解く側を「解く形＋組・広げる」に替える）
材料と話の作り方、並べ替えは tools/i49zp_text_talks_run.py と同じ（材料はリポジトリに入れない）。
--amp_check を付けると、大きさのそろえ方だけを変えた比べをする（そのまま ／ 属性を 1 ／ 位相だけ。ほかは同じ）:
  組だけ・広げる と 解く形＋組・広げる を、3 通りの大きさで。比べの元は「そのまま」。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --out 記録.json [--dim 8192] [--amp_check]
手で置く物: dim、点を足す重さ 1 対 1、並べ替えの k、後ろ 2 割。
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

NAMES = ("解く形", "解く形＋今までの束ね", "組だけ・まばら", "組だけ・広げる", "解く形＋組・広げる", "数えて選ぶ", "数えて選ぶ・組と")
DUELS = (("解く形", "解く形＋今までの束ね"), ("解く形", "組だけ・広げる"), ("組だけ・まばら", "組だけ・広げる"), ("解く形", "解く形＋組・広げる"),
         ("数えて選ぶ", "数えて選ぶ・組と"))
PAIR_NAMES = ("だれが×何を", "だれが×どうした", "何を×どうした")


def sparse_stats(attr_of, talks, dim):
    import numpy as np
    from i49zw_dense_pair import PairSolve
    p = PairSolve(attr_of, dim=dim)
    words = {w for t in talks for w in t[1]}
    counts = [int(np.count_nonzero(v)) for v in (attr_of(w) for w in words) if v is not None]
    n_attr = max((len(attr_of(w)) for w in words if attr_of(w) is not None), default=0)
    out = dict(属性の総数=n_attr, 属性を持つ語=len(counts), 語ごとの属性の数の平均=round(sum(counts) / max(1, len(counts)), 2),
               語ごとの属性の数の最大=max(counts, default=0), 力の有る所の割合=round(sum(counts) / max(1, len(counts)) / max(1, n_attr), 4))
    for name, (i, j) in zip(PAIR_NAMES, ((0, 1), (0, 2), (1, 2))):
        ss = [s for s in (p.stat(tri[i], tri[j]) for t in talks for tri in t[0]) if s is not None]
        n = max(1, len(ss))
        out[name] = dict(組=len(ss), 共通の属性の平均=round(sum(s["共通の属性"] for s in ss) / n, 2),
                         何も残らない組の割合=round(sum(s["共通の属性"] == 0 for s in ss) / n, 4), 残る長さの平均=round(sum(s["残る長さ"] for s in ss) / n, 4))
    return out


def make(name, wave_of, attr_of, dim):
    import i49e_relation as RL
    import i49h_tally_run as HR
    import i49k_seat_vec as KV
    from i49zb_mix import Mix
    from i49zw_dense_pair import Both, PairSolve

    def solve(pair=False):
        return KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相", pair=pair), online=True)
    if name == "解く形":
        return solve()
    if name == "解く形＋今までの束ね":
        return solve(pair=True)
    if name.startswith("組だけ"):
        return PairSolve(attr_of, mode=name.split("・")[1], dim=dim)
    both = Both(solve(), PairSolve(attr_of, dim=dim))
    if name == "解く形＋組・広げる":
        return both
    return Mix(RL.Relation(wave_of, half=HR.HALF), solve() if name == "数えて選ぶ" else both, upper=None)


AMPS = ("そのまま", "属性を 1", "位相だけ")


def run_amp(attr_of, talks, ks=(0, 1, 2, 3), dim=8192):
    import i49k_seat_vec as KV
    import i49w_also_right_run as WR
    import i49za_fold_run as ZR
    import i49zk_mix_phase_run as ZK
    from i49zw_dense_pair import Both, PairSolve
    plain = [(t[0], t[1]) for t in talks]
    ok = WR.ok_sets([t for tris, _ in plain for t in tris])
    out = {}
    for k in ks:
        name_k = "元の順" if k == 0 else "並べ替え %d" % k
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        d, keep = {}, {}
        for amp in AMPS:
            for name, make in (("組だけ・広げる｜" + amp, lambda: PairSolve(attr_of, dim=dim, amp=amp)),
                               ("解く形＋組・広げる｜" + amp, lambda: Both(KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True),
                                                                      PairSolve(attr_of, dim=dim, amp=amp)))):
                recs = ZR.stream(make(), seq, ok)
                s = ZR.summary(recs, 0)
                s["後ろ 2 割"] = ZP.tail_score(recs)
                keep[name], d[name] = recs, s
                print(name_k, name, s["正しい物"], s["合わせて"], flush=True)
        d["直の比べ"] = {"%s｜そのまま → %s" % (f, amp): ZK.duel(keep[f + "｜そのまま"], keep[f + "｜" + amp])
                      for f in ("組だけ・広げる", "解く形＋組・広げる") for amp in AMPS[1:]}
        out[name_k] = d
    return out


def run(wave_of, attr_of, talks, ks=(0, 1, 2, 3), dim=8192):
    import i49w_also_right_run as WR
    import i49za_fold_run as ZR
    import i49zk_mix_phase_run as ZK
    plain = [(t[0], t[1]) for t in talks]
    ok = WR.ok_sets([t for tris, _ in plain for t in tris])
    out = {}
    for k in ks:
        name_k = "元の順" if k == 0 else "並べ替え %d" % k
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        d, keep = {}, {}
        for name in NAMES:
            recs = ZR.stream(make(name, wave_of, attr_of, dim), seq, ok)
            s = ZR.summary(recs, 0)
            s["後ろ 2 割"] = ZP.tail_score(recs)
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
    ap.add_argument("--dim", type=int, default=8192)
    ap.add_argument("--amp_check", action="store_true")
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
    res = dict(話=len(talks), 三つ組=sum(len(t[0]) for t in talks), dim=a.dim, まばらさ=sparse_stats(full.of, talks, a.dim))
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    res.update(run_amp(full.of, talks, dim=a.dim) if a.amp_check else run(wave_of, full.of, talks, dim=a.dim))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
