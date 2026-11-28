"""I49zzzh の道具: 外の入力を薄める・切って待たせると、器の中で何が変わるかを数える（返事は出さない。ロードマップの追記 19 の「小さく確かめるなら」の 2）。

流れ: 持ち主の文を 1 本の流れとして聞く（話の切れ目は教えない）。行ごとに、遷移の驚き（3 席。I49zzze。遷移の波は流れの全部を聞かせる）を出し、
  Pressure で段（gain）を決め、Shelter を通して器（Hold）に入れる。
比べる 3 つ（同じ流れ、同じ信号）:
  そのまま聞く:        gain はいつも 1（今の仕組み）。
  薄める:              gain を 1／2/3／1/3 で止める（0 にはしない）。
  薄めて、切って待たせる: gain が 0 の時は切って待たせ、0 より上に戻った時に、待った分を薄めて聞く。
数える物:
  段ごとの行の数。話の初めの行と、それ以外の行で分ける（切れ目で閉じるのか、切れ目の後で閉じるのか）。
  待たせた行の数、1 回に続けて待たせた行の数の平均と最大。
  「だれが」を、持った波との共鳴（語そのもの、半減期 1 と 4）で当てた時の当たり（82 節の数え）。入力を薄めると、どれだけ変わるか。
  持つだけの帯（半減期 64 を private にする）を置いても、上の当たりが変わらない事（部品の確かめを、本番の流れでも見る）。
危ない相談の印（gate）は、この道具では渡さない（材料に当たる行が無い。部品の確かめは tests/test_i49zzzh.py）。
手で置いた物: intervention/i49zzzh_shelter.py の物と、信号に遷移の驚き（3 席）を使う事、半減期の格子、持つだけにする帯。乱数は使わない。
引数: --text 材料（カンマ区切り） --content 内容語の板 --out 記録.json [--k 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention"), os.path.join(HERE, "..", "intervention", "past")]

import i49zp_text_talks_run as ZP  # noqa: E402
import i49zzw_next_triple_run as W  # noqa: E402
import i49zzzd_hold_run as D  # noqa: E402
import i49zzze_boundary_run as E  # noqa: E402
import i49zzzh_shelter as S  # noqa: E402

HALVES = (1, 4, 16, 64)
PRIVATE = (64,)
ASK = (1, 4)
MODES = ("そのまま聞く", "薄める", "薄めて、切って待たせる")
SIGNAL = "遷移の驚き（%s）" % E.ALL3


def run(first, second, attr_of, halves=HALVES, private=PRIVATE, ask=ASK, dims=None, scale=None, progress=None):
    flow = W.Wave(W.DIMS if dims is None else dims, attr_of, W.SCALE if scale is None else scale)
    sig_hold = D.Hold(flow, halves)                       # 信号を出すための器（いつもそのまま聞く。3 つの比べで同じ信号にするため）
    sig, press = E.Signals(flow, sig_hold, D.Fast(flow)), S.Pressure()
    sh = {m: S.Shelter(D.Hold(flow, halves), private if m != MODES[0] else ()) for m in MODES}
    seats = set()
    cnt = {m: dict(問い=0, **{"半減期 %d" % h: 0 for h in ask}) for m in MODES}
    level = dict(話の初めの行={}, それ以外の行={})
    runs, cur_run, prev = [], 0, None
    start = len(first)
    talks = [t for t, _ in first] + [t for t, _ in second]
    for ti, tris in enumerate(talks):
        tris = [tuple(t) for t in tris]
        for i, cur in enumerate(tris):
            if ti >= start:                               # 聞く前に「だれが」を当てる
                cands = sorted(seats | {cur[0]}, key=str)
                for m in MODES:
                    cnt[m]["問い"] += 1
                    for h in ask:
                        cnt[m]["半減期 %d" % h] += W.pick(cands, sh[m].hold.word(h, 0, cands)) == cur[0]
            s = sig.before(prev, cur)[SIGNAL] if prev is not None else None
            sig_hold.hear(cur)
            gain = press.step(s)[1] if s is not None else 1.0
            if ti >= start:
                d = level["話の初めの行" if i == 0 else "それ以外の行"]
                d["%.2f" % gain] = d.get("%.2f" % gain, 0) + 1
            sh[MODES[0]].take(cur)
            sh[MODES[1]].set_gain(max(gain, 1.0 / S.STEPS))
            sh[MODES[1]].take(cur)
            sh[MODES[2]].set_gain(gain)
            kind = sh[MODES[2]].take(cur)
            if kind == S.WAIT:
                cur_run += 1
            elif cur_run:
                runs.append(cur_run)
                cur_run = 0
            if prev is not None:
                flow.learn(prev, cur)
            seats.add(cur[0])
            prev = cur
        if progress and ti % progress == 0:
            print("話", ti, "/", len(talks), flush=True)
    return dict(だれが=cnt, 段ごとの行の数=level, 入り方={m: sh[m].log for m in MODES}, 持つだけの帯={m: list(sh[m].private) for m in MODES},
                続けて待たせた行=dict(回=len(runs), 平均=sum(runs) / max(len(runs), 1), 最大=max(runs) if runs else 0),
                待たせたままの行=len(sh[MODES[2]].wait))


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
    res = dict(話=len(talks), 手で置いた値=dict(HALVES=list(HALVES), PRIVATE=list(PRIVATE), STEPS=S.STEPS, FAST=S.FAST, SLOW=S.SLOW, 信号=SIGNAL),
               **run(seq[:len(seq) // 2], seq[len(seq) // 2:], full.of, progress=40))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
