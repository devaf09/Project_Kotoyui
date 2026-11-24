"""I49zx: 介入の板（設計案の 1 つ目の段階）を、切り出しを通さない材料で数える。

本体 = 数えて選ぶ（関係の表 ＋ 波で解く形・属性）。本体は今まで通り、答えた後に全部の 3 つ組を聞く。
介入 = 答えた後に、決めた割合の問いでだけ「正しい語」が届く（教える役は、試しのための代わりとして材料の正解）。
        どの問いで届くかは等間隔（乱数を使わない。問いの番号 × 黄金比の小数部が、割合より小さい問い）。
        届いた分は、介入の板（intervention/i49zx_teach_board.py）にだけ入る。
先に決めた物差し（ロードマップの設計案の 4）:
  教えた組そのものがまた問われた時に当たるか ／ 教えていない似た組で当たるか（広がり） ／ 本体が当たっていたのに板で外れた数（害）
  ／ 板を付けても、本体の答えは 1 つも変わらない事（取り消し。板を外せば元の数に戻る事の確かめ）
材料と話の作り方、並べ替えは tools/i49zp_text_talks_run.py と同じ（材料はリポジトリに入れない）。
--unheard を付けると、本体が聞いていない事を教える場面で数える（I49zzb）:
  話を並べ替えて前半と後半に分け、本体には前半だけを聞かせる（問わない）。後半は問うだけで、本体には聞かせない。
  後半の問いのうち決めた割合でだけ「正しい語」が届き、介入の板に入る。並べる物:
    本体だけ（後半を聞かない） ／ 介入の板を付けた形（割合ごと） ／ 板が答えを持てばいつも板（比べ役） ／ 本体が後半も聞く・板なし（比べ役。同じ事を本体が全部聞けた時）
  鍵の形は --key（今まで・引く・めずらしさ。tools/i49zx_teach_try.py と同じ）。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --out 記録.json [--rates 0.1,0.3,1.0] [--dark 0.001] [--unheard] [--key 今まで] [--ks 1,2]
手で置く物: 介入の割合、暗い側の強さ、板の薄れる速さ・本数（部品の既定）、並べ替えの k（0 と 1）。
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

KINDS = ("教えた組そのもの", "似た組")


def body_of(wave_of, attr_of):
    import i49e_relation as RL
    import i49h_tally_run as HR
    import i49k_seat_vec as KV
    from i49zb_mix import Mix
    return Mix(RL.Relation(wave_of, half=HR.HALF), KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True), upper=None)


def stream(rel, seq, rate, hear=True):
    """→ 問いごとの (本体の答えが当たり, 最後の答えが当たり, 板の種類か None, 板の答えが当たり, 本体の答え)。rate = 介入が届く割合。
    hear=False なら、本体には聞かせない（板を付けた形では、どちらが当たったかの数えだけ進める）。"""
    recs, i = [], 0
    for tris, words in seq:
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                q = tuple(None if j == si else tri[j] for j in range(3))
                a = rel.ask(si, q, cands)["答え"]
                has, same, body_a, board_a = getattr(rel, "last", None) or (False, False, a, None)
                recs.append((body_a == tri[si], a == tri[si], (KINDS[0] if same else KINDS[1]) if has else None, board_a == tri[si], body_a))
                i += 1
                if rate and ((i * ZP.PHI) % 1.0) < rate:
                    rel.teach(si, tri, a)
            if hear:
                rel.learn(tri)
            elif hasattr(rel, "settle"):
                rel.settle(tri)
        if hear:
            rel.tick()
    return recs


def summary(recs, plain):
    n = max(1, len(recs))
    out = dict(問い=len(recs), 本体=round(sum(r[0] for r in recs) / n, 4), 板を付けた形=round(sum(r[1] for r in recs) / n, 4),
               当たりに変わった=sum((not r[0]) and r[1] for r in recs), 外れに変わった_害=sum(r[0] and not r[1] for r in recs),
               本体の答えが板なしと違った数=sum(r[4] != p[4] for r, p in zip(recs, plain)))
    up, dn = out["当たりに変わった"], out["外れに変わった_害"]
    out["z"] = round((up - dn) / max(1.0, (up + dn) ** 0.5), 2)
    for k in KINDS:
        rs = [r for r in recs if r[2] == k]
        m = max(1, len(rs))
        out[k] = dict(問い=len(rs), 本体の当たり=round(sum(r[0] for r in rs) / m, 4), 板の当たり=round(sum(r[3] for r in rs) / m, 4),
                      最後の当たり=round(sum(r[1] for r in rs) / m, 4))
    return out


def run(wave_of, attr_of, talks, rates=(0.1, 0.3, 1.0), ks=(0, 1), dark=1.0 / 1000, dim=8192):
    from i49zx_teach_board import Taught, TeachBoard
    plain_t = [(t[0], t[1]) for t in talks]
    out = {}
    for k in ks:
        seq = [plain_t[i] for i in ZP.order(len(plain_t), k)]
        plain = stream(body_of(wave_of, attr_of), seq, 0)
        d = {"板なし": dict(問い=len(plain), 本体=round(sum(r[0] for r in plain) / max(1, len(plain)), 4))}
        for rate in rates:
            t = Taught(body_of(wave_of, attr_of), TeachBoard(attr_of, dim=dim, dark=dark))
            recs = stream(t, seq, rate)
            s = summary(recs, plain)
            s["教えた数"] = t.board.n
            s["板が答えを持った問いでの数え"] = dict(本体の当たり=t.tally[0], 板の当たり=t.tally[1], 問い=t.tally[2])
            d["介入 %d%%" % round(rate * 100)] = s
            print(k, rate, s["本体"], s["板を付けた形"], s["z"], flush=True)
        out["元の順" if k == 0 else "並べ替え %d" % k] = d
    return out


def run_unheard(wave_of, attr_of, talks, rates=(0.1, 0.3, 1.0), ks=(1, 2), dark=1.0 / 1000, dim=8192, key="今まで"):
    import i49zx_teach_try as TT
    from i49zx_teach_board import Taught
    plain_t = [(t[0], t[1]) for t in talks]
    out = {}
    for k in ks:
        seq = [plain_t[i] for i in ZP.order(len(plain_t), k)]
        first, second = seq[:len(seq) // 2], seq[len(seq) // 2:]
        seats = [[tri[j] for tris, _ in seq for tri in tris] for j in range(3)]

        def heard_body():
            b = body_of(wave_of, attr_of)
            for tris, _ in first:
                for tri in tris:
                    b.learn(tri)
                b.tick()
            return b

        plain = stream(heard_body(), second, 0, hear=False)
        n = max(1, len(plain))
        both = stream(heard_body(), second, 0, hear=True)
        d = {"本体だけ（後半を聞かない）": dict(問い=len(plain), 本体=round(sum(r[0] for r in plain) / n, 4)),
             "本体が後半も聞く・板なし（比べ役）": dict(問い=len(both), 本体=round(sum(r[0] for r in both) / n, 4))}
        for rate in rates:
            t = Taught(heard_body(), TT.board_of(attr_of, key, seats, dark, dim))
            recs = stream(t, second, rate, hear=False)
            s = summary(recs, plain)
            s["教えた数"] = t.board.n
            s["板が答えを持った問い"] = sum(r[2] is not None for r in recs)
            s["板が答えを持てばいつも板（比べ役）"] = round(sum((r[3] if r[2] is not None else r[0]) for r in recs) / n, 4)
            s["板が答えを持った問いでの数え"] = dict(本体の当たり=t.tally[0], 板の当たり=t.tally[1], 問い=t.tally[2])
            d["介入 %d%%" % round(rate * 100)] = s
            print(k, rate, s["本体"], s["板を付けた形"], s["z"], flush=True)
        out["元の順" if k == 0 else "並べ替え %d" % k] = d
    return out


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--rates", default="0.1,0.3,1.0")
    ap.add_argument("--dark", type=float, default=1.0 / 1000)
    ap.add_argument("--unheard", action="store_true")
    ap.add_argument("--key", default="今まで")
    ap.add_argument("--ks", default="1,2")
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
    res = dict(話=len(talks), 三つ組=sum(len(t[0]) for t in talks), 暗い側の強さ=a.dark)
    rates = tuple(float(x) for x in a.rates.split(","))
    if a.unheard:
        res["鍵の形"] = a.key
        res.update(run_unheard(wave_of, full.of, talks, rates=rates, ks=tuple(int(x) for x in a.ks.split(",")), dark=a.dark, key=a.key))
    else:
        res.update(run(wave_of, full.of, talks, rates=rates, dark=a.dark))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
