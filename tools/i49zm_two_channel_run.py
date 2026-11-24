"""I49zm: 属性の波と、語ごとに固有の位相を、1 本の波に並べて持つ形を数える（持ち主の案、未確定の試し）。数えだけ。

intervention/i49zm_two_channel.py の TwoChannel を、解く形（I49k）の元の波にする。覚え方・答え方は変えない。
並べる物:
  関係の表だけ ／ 解く形・属性だけ（今までの形）
  解く形・属性＋固有: 混ぜ具合 .5・1・2 × 固有の位相 256・1,024 本
  解く形・属性＋語の波（比べ役。固有ではなく、似た語どうしが近い波を並べる）: 混ぜ具合 1
  解く形・属性＋固有（1・1,024 本）で、読む時の振幅を誤差で直す（I49zg。混ぜ具合を数えで決められるかを見る。決まりから外れる試し）
  数えて選ぶ（表と解く形）: 解く側が 属性だけ ／ 属性＋固有（1・1,024 本）
採点は、正しい物と、合わせて。関係の表だけから 当たりに変わった数・外れに変わった数、席ごと。
直の比べ（同じ問いで片方だけ当たった数）: 属性だけ 対 属性＋固有、表だけ 対 属性＋固有、数えて選ぶの 属性だけ 対 属性＋固有。
解く形だけの版では、関係の表の段（聞いた組そのもの・片方だけ借りる・…）ごとの当たりも出す（どの種類の問いで変わったかを見るため。表は段を知るためだけに並べて問う）。
流れ・引数は tools/i49zk_mix_phase_run.py と同じ（--content 付き）。版が 1 つ出るたびに記録を書く。記録はリポジトリの外。
手で置く物: 混ぜ具合、固有の位相の本数、直す速さ（振幅 .01）、temp .05、同じ数の時に表を使う事。
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
import i49zk_mix_phase_run as ZK  # noqa: E402

BASE = "関係の表だけ"
AMP = dict(phase_lr=0.0, rule="毎回", lr=.01, by_seat=True)
ID1 = dict(mix=1.0, n_id=1024)
VERS = [(BASE, "表", None, None), ("解く形｜属性だけ", "解く", None, None)]
for n_id in (256, 1024):
    for mix in (.5, 1.0, 2.0):
        VERS.append(("解く形｜属性＋固有｜混ぜ %g｜%d 本" % (mix, n_id), "解く", dict(mix=mix, n_id=n_id), None))
VERS += [("解く形｜属性＋語の波｜混ぜ 1", "解く", dict(mix=1.0, 語の波=True), None),
         ("解く形｜属性＋固有｜混ぜ 1｜1024 本｜振幅を直す", "解く", ID1, AMP),
         ("数えて選ぶ｜属性だけ", "数える", None, None),
         ("数えて選ぶ｜属性＋固有｜混ぜ 1｜1024 本", "数える", ID1, None)]
DUELS = [("解く形｜属性だけ", "解く形｜属性＋固有｜混ぜ 1｜1024 本"), (BASE, "解く形｜属性＋固有｜混ぜ 1｜1024 本"),
         ("解く形｜属性＋固有｜混ぜ 1｜1024 本", "解く形｜属性＋固有｜混ぜ 1｜1024 本｜振幅を直す"),
         ("数えて選ぶ｜属性だけ", "数えて選ぶ｜属性＋固有｜混ぜ 1｜1024 本")]


class Peek:
    """解く形の答えをそのまま返しつつ、関係の表にも同じ問いをして、段ごとの当たりを数える（表の答えは使わない）。"""

    def __init__(self, table, solve):
        self.table, self.solve, self.tally, self._wait = table, solve, {}, []

    def ask(self, si, tri, cands):
        cands = list(cands)
        lv = self.table.ask(si, tri, cands)["段"]
        o = self.solve.ask(si, tri, cands)
        self._wait.append((si, lv, o["答え"]))
        return o

    def learn(self, tri):
        for si, lv, a in self._wait:
            h = self.tally.setdefault(lv, [0, 0])
            h[0] += a == tri[si]
            h[1] += 1
        self._wait = []
        self.table.learn(tri)
        self.solve.learn(tri)

    def tick(self):
        self.table.tick()
        self.solve.tick()


def run(wave_of, attr_of, talks, aoz, save=None):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    from i49zb_mix import Mix
    from i49zj_phase_read import PhaseRead
    from i49zm_two_channel import TwoChannel
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    dims = {}                                             # 属性の側と語の波の側の長さを、先に材料から調べておく（長さだけ）
    for tri in [t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz]:
        for w in tri:
            for key, f in (("n_attr", attr_of), ("n_wave", wave_of)):
                if key not in dims:
                    v = f(w)
                    if v is not None:
                        dims[key] = len(v)
        if len(dims) == 2:
            break
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out[start], base, keep = {}, None, {}
        for name, kind, two, read in VERS:
            t1 = time.time()
            table = RL.Relation(wave_of, half=HR.HALF)
            inner = None
            if kind != "表":
                if two is None:
                    base_of = attr_of
                else:
                    kw = dict(two)
                    base_of = TwoChannel(attr_of, wave_of=wave_of if kw.pop("語の波", False) else None, n_attr=dims.get("n_attr"), n_wave=dims.get("n_wave"), **kw).of
                sv = KV.SeatVec(base_of, weight="数", bind="位相")
                inner = PhaseRead(sv, **read) if read else KV.Unbind(sv, online=True)
            for tri in pre:
                table.learn(tri)
                if inner is not None:
                    inner.sv.hear(tri)
                    if read:
                        inner.hear_seen(tri)
            rel = table if kind == "表" else Peek(table, inner) if kind == "解く" else Mix(table, inner, upper=None)
            recs = ZR.stream(rel, talks, ok)
            d = ZR.summary(recs, 0, base)
            base = recs if base is None else base
            keep[name] = recs
            if kind == "解く":
                d["段ごとの当たり"] = {lv: dict(解く=h[0], 問い=h[1]) for lv, h in rel.tally.items()}
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
        out[start]["直の比べ"] = {"%s → %s" % (a, b): ZK.duel(keep[a], keep[b]) for a, b in DUELS}
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
