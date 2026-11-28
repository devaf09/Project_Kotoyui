"""I49zzzg: 言われなかった席を前提で埋める時の「割れ方」と、言われた語が前提と「合わない」度合いを、疑問の物差しとして数える
（持ち主の案、ロードマップの追記 18。数えるだけ。未確定の試し）。

持ち主の案: 「前提を話始めに持つのはどうだろうと思ってた、つまり見えない文字で相手の情報や今の自分の情報をもっとく、例えば話し始めに
  『元気？』だけ聞かれたらうっすら主語を自分に持っとく、みたいな。でもたとえば、『歩いてた。』って言われたら前提が多くて迷う、誰が？
  『私、あなた、猫』？みたいな。そういう足りない部分と前提と合わない部分で疑問感を持てないかなって」
ここでの前提: 聞いた行を 1 本の流れとして、席ごとに語を薄めて数えた物（82 節の Hold の「語そのもの」。話の切れ目は教えない）。
1. 足りない（席を隠して、前提で埋める）:
   材料は 3 席とも言う文なので、1 席をわざと隠す（だれが を主に。何を・どうした も同じ形で数える）。
   隠した席を、前提の一番大きい語で埋める。その時の割れ方を 2 通りで出す:
     割れ方（エントロピー）: 前提の数えを割合にした時の、散らばり（ビット）。1 語だけなら 0、同じくらいの語が多いほど大きい。
     上位 2 つの近さ:       2 番目の数え ÷ 1 番目の数え（0〜1）。
   確かめる事: 割れ方が大きい行ほど、埋めた語が外れているか。
     見分け（AUC）: 外れた行の割れ方が、当たった行の割れ方より大きい割合（.5 が当てずっぽう）。
     割れ方の小さい順に 4 つに等分した時の、組ごとの当たり。
   これが立てば、「割れ方」は、答えを見ずに出せる疑問の強さとして使える。隠した席が、疑問の中身（だれが？）。
2. 合わない（言われた語が、前提とぶつかる）:
   席ごとに、言われた語の、前提の中での割合（0〜1）を出し、1 から引いた物を「合わなさ」とする。
   数える事: 席ごとの平均（話の初めの行と、それ以外の行で分ける）。3 席のうち一番合わない席がどれか、の割合。
   「驚き」との違いは、どの席でぶつかったかを持つ所。
材料の向き不向き: この材料は主語を全部言う。隠して当てる形は、言わない主語（会話）の代わりの、作り物の試し。
  会話の材料（MRMP）での「言われない主語が話し手か相手か」は、正解の印が無いので、ここでは数えられない。
手で置いた物: 半減期の格子、割れ方の 2 つの定義、4 つに等分する事、隠す席の選び方。乱数は使わない。材料の語や例は出さない。
引数: --text 材料（カンマ区切り） --content 内容語の板 --out 記録.json [--k 1]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention"), os.path.join(HERE, "..", "intervention", "past")]

import i49zp_text_talks_run as ZP  # noqa: E402
import i49zzw_next_triple_run as W  # noqa: E402
import i49zzzd_hold_run as D  # noqa: E402
import i49zzze_boundary_run as E  # noqa: E402

HALVES = (1, 4, 16)
SPLITS = ("割れ方（エントロピー）", "上位 2 つの近さ")
BINS = 4


def entropy(counts):
    s = sum(counts)
    return -sum(p * math.log2(p) for p in (c / s for c in counts) if p > 0) if s > 0 else 0.0


def fill(held):
    """held: {語: 薄めて数えた回数}。→ (埋める語, 割れ方（エントロピー）, 上位 2 つの近さ)。前提が空なら (None, 0, 0)。"""
    if not held:
        return None, 0.0, 0.0
    top = sorted(held.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return top[0][0], entropy([c for _, c in top]), (top[1][1] / top[0][1] if len(top) > 1 and top[0][1] > 0 else 0.0)


def misfit(held, word):
    """言われた語の、前提の中での割合を 1 から引く。前提が空なら 1。"""
    s = sum(held.values())
    return 1.0 - held.get(word, 0.0) / s if s > 0 else 1.0


def by_bins(rows, bins=BINS):
    """rows: [(割れ方, 当たったか)]。割れ方の小さい順に等分して、組ごとの (数, 当たり, 割れ方の平均)。"""
    rows = sorted(rows, key=lambda r: r[0])
    out = []
    for b in range(bins):
        part = rows[b * len(rows) // bins:(b + 1) * len(rows) // bins]
        out.append(dict(数=len(part), 当たり=sum(1 for r in part if r[1]), 割れ方の平均=sum(r[0] for r in part) / max(len(part), 1)))
    return out


def run(first, second, attr_of, halves=HALVES, dims=None, scale=None, progress=None):
    wave = W.Wave(W.DIMS if dims is None else dims, attr_of, W.SCALE if scale is None else scale)
    hold = D.Hold(wave, halves)
    for tris, _ in first:
        for t in tris:
            hold.hear(tuple(t))
    lack = {(si, h): [] for si in range(3) for h in halves}      # (割れ方 2 つ, 当たったか, 前の行と同じ語か)
    mis = {h: dict(話の初めの行=[0, 0.0, 0.0, 0.0], それ以外の行=[0, 0.0, 0.0, 0.0], 一番合わない席=[0, 0, 0]) for h in halves}
    prev = tuple(first[-1][0][-1]) if first else None
    for ti, (tris, _) in enumerate(second):
        for i, cur in enumerate(tuple(t) for t in tris):
            for h in halves:
                ms = []
                for si in range(3):
                    w, ent, near = fill(hold.n[h][si])
                    lack[(si, h)].append((ent, near, w == cur[si], prev is not None and prev[si] == cur[si]))
                    ms.append(misfit(hold.n[h][si], cur[si]))
                m = mis[h]["話の初めの行" if i == 0 else "それ以外の行"]
                m[0] += 1
                for si in range(3):
                    m[1 + si] += ms[si]
                mis[h]["一番合わない席"][max(range(3), key=lambda s: (ms[s], -s))] += 1
            hold.hear(cur)
            prev = cur
        if progress and ti % progress == 0:
            print("話", ti, "/", len(second), flush=True)
    out_lack = {}
    for (si, h), rows in lack.items():
        d = dict(問い=len(rows), 当たり=sum(1 for r in rows if r[2]), 前の行と同じ語=sum(1 for r in rows if r[3]))
        for k, name in enumerate(SPLITS):
            d[name] = dict(見分け=E.auc([r[k] for r in rows if not r[2]], [r[k] for r in rows if r[2]]), 等分=by_bins([(r[k], r[2]) for r in rows]),
                           前の行と違う語の時の見分け=E.auc([r[k] for r in rows if not r[3]], [r[k] for r in rows if r[3]]))
        out_lack["%s 半減期 %d" % (ZP.SEATS[si], h)] = d
    out_mis = {}
    for h in halves:
        d = {}
        for k in ("話の初めの行", "それ以外の行"):
            n = mis[h][k][0]
            d[k] = dict(行=n, **{s: mis[h][k][1 + si] / max(n, 1) for si, s in enumerate(ZP.SEATS)})
        d["一番合わない席"] = dict(zip(ZP.SEATS, mis[h]["一番合わない席"]))
        out_mis["半減期 %d" % h] = d
    return dict(足りない=out_lack, 合わない=out_mis)


def main():
    import i49m_attr_base as AB
    import i49zzz_talk_memory_run as Z
    ap = argparse.ArgumentParser()
    for k in ("text", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--k", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    talks = []
    for f in a.text.split(","):
        talks += [(t[0], t[1]) for t in ZP.talks_of(Z.read(f))[0]]
    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    seq = [talks[i] for i in ZP.order(len(talks), a.k)]
    res = dict(話=len(talks), 手で置いた値=dict(HALVES=list(HALVES), BINS=BINS), **run(seq[:len(seq) // 2], seq[len(seq) // 2:], full.of, progress=40))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
