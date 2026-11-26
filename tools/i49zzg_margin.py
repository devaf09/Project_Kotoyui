"""I49zzg: 1 位と 2 位の点の差ごとに、どちらが当たるかを数える（見るだけの数え。揺らぎを育てる案の初めの一歩、未確定の試し）。

持ち主の言葉:「むしろその揺らぎを大きくする方法を考えてほしい」。
今の数えの揺れ（同じ数えで 1 問だけ動く）は、1 位と 2 位の点がほぼ同じ問いで、小数の下の桁が順位を決めている物。
「同じ」と見なす幅を広げて、その中から別の手掛かり（器の向きなど）で選ぶ形を考える前に、幅をどこまで広げて良いかを数える。
  場面: 話を並べ替えて半分に分け、本体（i49_foundation.body）に前半を聞かせ、後半は問うてから聞かせる（「本体が後半も聞く」と同じ）。
  問いごとに、本体が答えに使った点の 1 位と 2 位を取り、差の割合 d =（1 位 − 2 位）÷ 1 位 を出す。
  答えた段（2語・片方だけ借りる・述語側・もう片方・借りる・解く）と d の区切りごとに、問い・1 位が当たり・2 位が当たり を数える。
  1 位と 2 位の当たりが五分に近い区切りが、「揺らしても損をしない幅」の目安になる。
--vessel を付けると（持ち主の言葉「揺らす元を器（内面の波）にする。これがいいね、簡単に試験して」から）:
  器（i49a_inner_vessel.Vessel、属性の数だけの長さ）に、聞いた 3 つ組の語の属性の波（長さ 1）を聞かせ続ける。話の終わりに tick。
  問いのたびに、1 位との差の割合が幅以下の候補の中から、器の向き（whole）に一番近い候補を選び直したら、答えが幾つ変わり、
  当たり外れがどう動くかを、幅ごとに数える。本体の答えと学びは変えない（見るだけ）。
  比べ役: 同じ集まりの中から、器の向きに一番「遠い」候補を選ぶ形。器の向きに意味が有れば、近い方が遠い方より当たるはず。
  オウム返しの確かめ（持ち主の読み「器に直前の単語をそのまま入れたことによる同じ単語のオウム返し」）: 選び直した先が、
  この話の中でもう聞いた語だった時を別に数える（「聞いた語を選んで_…」）。近い方の損が、聞いた語を選んだ時に片寄っていれば、読みの通り。
--gap を付けると（持ち主の案 3-2「速い波と遅い波のズレを使って、連想の探索範囲を制御する」の前提の確かめ）:
  問いのたびに、器の「速い波と遅い波のずれ」を読み、ずれの小さい順に 5 等分して、正解が「この話でもう聞いた語」の割合と、本体の当たりを並べる。
  ずれが大きい時に新しい語が正解になりやすければ、ずれで探す広さを決める案に筋が通る。
手で置いた物: d の区切り（見るための区切りで、選び方には使わない）、場面の作り方（半分に分ける、並べ替えの k）。乱数は使わない。
材料の語や例は出さない。出すのは段の名前と集計の数だけ。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --out 記録.json [--ks 1,2,3]
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

BINS = ((0.0, "0（全く同じ）"), (1e-6, "〜0.000001"), (1e-4, "〜0.0001"), (1e-2, "〜0.01"), (0.05, "〜0.05"), (0.1, "〜0.1"),
        (0.2, "〜0.2"), (0.5, "〜0.5"), (float("inf"), "それより上"))


def bin_of(d):
    return next(name for lim, name in BINS if d <= lim)


def margin(points, cands):
    """点（候補 → 点）から、(1 位, 2 位, 差の割合) を返す。順は、点が大きい順、同じ点なら候補の並びで先の物（本体の選び方と同じ）。"""
    order = sorted(cands, key=lambda c: (-float(points.get(c, 0.0)), cands.index(c)))
    s1, s2 = float(points.get(order[0], 0.0)), float(points.get(order[1], 0.0))
    return order[0], order[1], ((s1 - s2) / abs(s1) if s1 != 0 else 0.0)


def _unit(v):
    import numpy as np
    if v is None:
        return None
    v = np.asarray(v, dtype=np.complex64)
    n = float(np.linalg.norm(v))
    return v / np.float32(n) if n > 0 else None


def vessel_pick(points, cands, lim, state, dir_of, far=False):
    """1 位との差の割合が lim 以下の候補（同点の集まり）の中から、器の向き state に一番近い（far=True なら一番遠い）候補を返す。
    1 位が向きを持たない時、state が無い時、向きを持つ候補が 1 つだけの時は、1 位のまま。"""
    import numpy as np
    order = sorted(cands, key=lambda c: (-float(points.get(c, 0.0)), cands.index(c)))
    s1 = float(points.get(order[0], 0.0))
    if state is None or s1 == 0 or dir_of(order[0]) is None:
        return order[0]
    tie = [c for c in order if (s1 - float(points.get(c, 0.0))) / abs(s1) <= lim and dir_of(c) is not None]
    if len(tie) < 2:
        return order[0]
    near = [float(np.real(np.vdot(state, dir_of(c)))) for c in tie]
    best = min(range(len(tie)), key=lambda i: (near[i], i)) if far else max(range(len(tie)), key=lambda i: (near[i], -i))
    return tie[best]


def count_vessel(body, first, second, attr_of, vessel):
    """器（内面の波）に、聞いた 3 つ組の語の向き（属性の波、長さ 1）を聞かせ続け、問いのたびに、同点の集まりの中から器の向きに近い候補を
    選び直したら答えがどう変わるかを、幅ごとに数える（見るだけ。本体の答えと学びは変えない）。比べ役として、一番遠い候補を選ぶ形も数える。
    → 幅 → dict(問い, 答えが変わった, 当たりに変わった, 外れに変わった, 遠い方_答えが変わった, 遠い方_当たりに変わった, 遠い方_外れに変わった)"""
    dirs = {}

    def dir_of(w):
        if w not in dirs:
            dirs[w] = _unit(attr_of(w))
        return dirs[w]

    def hear(tri):
        for w in tri:
            if dir_of(w) is not None:
                vessel.hear(dir_of(w))

    for tris, _ in first:
        for tri in tris:
            body.learn(tri)
            hear(tri)
        body.tick()
        vessel.tick()
    keys = ("答えが変わった", "当たりに変わった", "外れに変わった")
    out = {name: dict(問い=0, 正解がこの話で聞いた語=0, **{p + h + k: 0 for p in ("", "遠い方_") for h in ("", "聞いた語を選んで_") for k in keys}) for _, name in BINS}
    for tris, words in second:
        heard = set()          # この話の中で、もう聞いた語（オウム返しかどうかを見るため。I49zzg の続き）
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                o = body.ask(si, tuple(None if j == si else tri[j] for j in range(3)), cands)
                if o["答え"] is None or not o.get("点"):
                    continue
                state = vessel.whole()
                for lim, name in BINS:
                    c = out[name]
                    c["問い"] += 1
                    c["正解がこの話で聞いた語"] += tri[si] in heard
                    for far, pre in ((False, ""), (True, "遠い方_")):
                        a = vessel_pick(o["点"], cands, lim, state, dir_of, far=far)
                        if a != o["答え"]:
                            c[pre + "答えが変わった"] += 1
                            c[pre + "当たりに変わった"] += a == tri[si]
                            c[pre + "外れに変わった"] += o["答え"] == tri[si]
                            if a in heard:
                                c[pre + "聞いた語を選んで_答えが変わった"] += 1
                                c[pre + "聞いた語を選んで_当たりに変わった"] += a == tri[si]
                                c[pre + "聞いた語を選んで_外れに変わった"] += o["答え"] == tri[si]
            body.learn(tri)
            hear(tri)
            heard.update(tri)
        body.tick()
        vessel.tick()
    return out


def count_gap(body, first, second, attr_of, vessel):
    """器の「速い波と遅い波のずれ」と、正解の出方の関係を見る（持ち主の案 3-2 の前提。見るだけ）。
    問いのたびに、ずれ、正解がこの話でもう聞いた語か、本体が当たったか、を記録する。
    → 問いごとの (ずれ, 正解が聞いた語, 本体が当たり, この話の中で何個目の 3 つ組か) の並び。"""
    dirs = {}

    def hear(tri):
        for w in tri:
            if w not in dirs:
                dirs[w] = _unit(attr_of(w))
            if dirs[w] is not None:
                vessel.hear(dirs[w])

    for tris, _ in first:
        for tri in tris:
            body.learn(tri)
            hear(tri)
        body.tick()
        vessel.tick()
    rows = []
    for tris, words in second:
        heard = set()
        for n, tri in enumerate(tris):
            gap = float(vessel.want()["速い波と遅い波のずれ"])
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                o = body.ask(si, tuple(None if j == si else tri[j] for j in range(3)), cands)
                rows.append((gap, tri[si] in heard, o["答え"] == tri[si], n))
            body.learn(tri)
            hear(tri)
            heard.update(tri)
        body.tick()
        vessel.tick()
    return rows


def gap_table(rows, parts=5):
    """ずれの小さい順に parts 等分して、組ごとに、問い・ずれの範囲・正解が聞いた語の割合・本体の当たり を出す。
    話の初めの 3 つ組（聞いた語がまだ無い）は、別に数える。"""
    out = {}
    for name, rs in (("話の 2 つ目からの 3 つ組", [r for r in rows if r[3] > 0]), ("話の初めの 3 つ組", [r for r in rows if r[3] == 0])):
        rs = sorted(rs, key=lambda r: r[0])
        tab = []
        for k in range(parts):
            part = rs[len(rs) * k // parts: len(rs) * (k + 1) // parts]
            if part:
                n = len(part)
                tab.append(dict(問い=n, ずれの下=round(part[0][0], 4), ずれの上=round(part[-1][0], 4),
                                正解が聞いた語=round(sum(r[1] for r in part) / n, 4), 本体の当たり=round(sum(r[2] for r in part) / n, 4),
                                聞いた語の時の当たり=round(sum(r[2] for r in part if r[1]) / max(sum(r[1] for r in part), 1), 4),
                                新しい語の時の当たり=round(sum(r[2] for r in part if not r[1]) / max(sum(not r[1] for r in part), 1), 4)))
        out[name] = tab
    return out


def run_gap(wave_of, attr_of, talks, n_attr, ks=(1, 2, 3)):
    import i49_foundation as FD
    import i49a_inner_vessel as IV
    plain = [(t[0], t[1]) for t in talks]
    res, every = {}, []
    for k in ks:
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        rows = count_gap(FD.body(wave_of, attr_of), seq[:len(seq) // 2], seq[len(seq) // 2:], attr_of, IV.Vessel(n_attr))
        every += rows
        res["元の順" if k == 0 else "並べ替え %d" % k] = gap_table(rows)
        print(k, len(rows), flush=True)
    res["合計"] = gap_table(every)
    return res


def run_vessel(wave_of, attr_of, talks, n_attr, ks=(1, 2, 3)):
    import i49_foundation as FD
    import i49a_inner_vessel as IV
    plain = [(t[0], t[1]) for t in talks]
    res = {}
    for k in ks:
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        v = IV.Vessel(n_attr)
        d = count_vessel(FD.body(wave_of, attr_of), seq[:len(seq) // 2], seq[len(seq) // 2:], attr_of, v)
        res["元の順" if k == 0 else "並べ替え %d" % k] = dict(幅ごと=d, 器の見張り=dict(張り付き=int(v.watch["張り付き"].sum()), 死んだ=int(v.watch["死んだ"])))
        print(k, {n: (c["答えが変わった"], c["当たりに変わった"], c["外れに変わった"]) for n, c in d.items()}, flush=True)
    return res


def count(body, first, second):
    """→ 段 → 区切り → dict(問い, 1位が当たり, 2位が当たり)。答えられなかった問い・候補が 1 つの問いは「数えない」に入れる。"""
    for tris, _ in first:
        for tri in tris:
            body.learn(tri)
        body.tick()
    out, skip = {}, 0
    for tris, words in second:
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                o = body.ask(si, tuple(None if j == si else tri[j] for j in range(3)), cands)
                if o["答え"] is None or not o.get("点"):
                    skip += 1
                    continue
                a, b, d = margin(o["点"], cands)
                c = out.setdefault(o["段"], {}).setdefault(bin_of(d), dict(問い=0, 一位が当たり=0, 二位が当たり=0))
                c["問い"] += 1
                c["一位が当たり"] += a == tri[si]
                c["二位が当たり"] += b == tri[si]
            body.learn(tri)
        body.tick()
    return out, skip


def run(wave_of, attr_of, talks, ks=(1, 2, 3)):
    import i49_foundation as FD
    plain = [(t[0], t[1]) for t in talks]
    res = {}
    for k in ks:
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        by, skip = count(FD.body(wave_of, attr_of), seq[:len(seq) // 2], seq[len(seq) // 2:])
        total = {}
        for lv in by.values():
            for name, c in lv.items():
                t = total.setdefault(name, dict(問い=0, 一位が当たり=0, 二位が当たり=0))
                for key in c:
                    t[key] += c[key]
        order = [name for _, name in BINS]
        res["元の順" if k == 0 else "並べ替え %d" % k] = dict(
            数えない問い=skip, 全部の段={n: total[n] for n in order if n in total},
            段ごと={lv: {n: by[lv][n] for n in order if n in by[lv]} for lv in sorted(by)})
        print(k, res["元の順" if k == 0 else "並べ替え %d" % k]["全部の段"], flush=True)
    return res


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--gap", action="store_true")     # 器の速い波と遅い波のずれと、正解の出方の関係を数える
    ap.add_argument("--vessel", action="store_true")  # 器の向きで、同点の集まりの中から選び直した時の変わり方を数える
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
    ks = tuple(int(x) for x in a.ks.split(","))
    if a.gap:
        res = run_gap(wave_of, full.of, talks, full.n_attr, ks=ks)
    else:
        res = run_vessel(wave_of, full.of, talks, full.n_attr, ks=ks) if a.vessel else run(wave_of, full.of, talks, ks=ks)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
