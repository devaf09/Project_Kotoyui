"""I49zx: 介入の板（設計案の 1 つ目の段階）を、切り出しを通さない材料で数える。

本体 = 数えて選ぶ（関係の表 ＋ 波で解く形・属性）。本体は今まで通り、答えた後に全部の 3 つ組を聞く。
介入 = 答えた後に、決めた割合の問いでだけ「正しい語」が届く（教える役は、試しのための代わりとして材料の正解）。
        どの問いで届くかは等間隔（乱数を使わない。問いの番号 × 黄金比の小数部が、割合より小さい問い）。
        届いた分は、介入の板（intervention/i49zx_teach_board.py）にだけ入る。
先に決めた物差し（ロードマップの設計案の 4）:
  教えた組そのものがまた問われた時に当たるか ／ 教えていない似た組で当たるか（広がり） ／ 本体が当たっていたのに板で外れた数（害）
  ／ 板を付けても、本体の答えは 1 つも変わらない事（取り消し。板を外せば元の数に戻る事の確かめ）
材料と話の作り方、並べ替えは tools/i49zp_text_talks_run.py と同じ（材料はリポジトリに入れない）。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --out 記録.json [--rates 0.1,0.3,1.0] [--dark 0.001]
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


def stream(rel, seq, rate):
    """→ 問いごとの (本体の答えが当たり, 最後の答えが当たり, 板の種類か None, 板の答えが当たり, 本体の答え)。rate = 介入が届く割合。"""
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
            rel.learn(tri)
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


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--rates", default="0.1,0.3,1.0")
    ap.add_argument("--dark", type=float, default=1.0 / 1000)
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
    res.update(run(wave_of, full.of, talks, rates=tuple(float(x) for x in a.rates.split(",")), dark=a.dark))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
