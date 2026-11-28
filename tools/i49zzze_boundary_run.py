"""I49zzze: 話の切れ目で、器の中の信号は本当に立つか（数えるだけ。持ち主とのメモ、ロードマップの追記 17 の順の 1。未確定の試し）。

追記 17 の案（AI メモ 1）: 「速い波（半減期 1〜4）と中間の波（半減期 8〜32）の乖離（速い波と遅い波のずれ、または急激な驚き sur のスパイク）が
  連続して大きくなった瞬間を『話の切れ目（場面転換）』と自己判定させます。」
作業側の突っ込み: 使う前に、その信号が本物の切れ目で本当に立つかを数える。しきい値は手で置かず、数えから決める。
ここでは、聞いた行を 1 本の流れとして（82 節の Hold）、行ごとに信号を出し、「話の初めの行」と「それ以外の行」で比べる。
信号（どれも、大きいほど切れ目らしい、という向きにそろえる）:
  人が替わった:                 前の行と だれが が違う（0／1。比べ役）。
  持った波と合わない（席、半減期）: その行の語の波と、持った波の向きの近さ（cos）の低さ。3 席を足した物も。
  速い波と遅い波のずれ（a と b）:   その行を聞いた後の、半減期 a の波と b の波の向きの近さの低さ（3 席を足す）。
  遷移の驚き（席）:               前の行から遷移の波で引いた時の、本物の語の点の低さ。3 席を足した物も。
数える物:
  見分け（AUC）: 切れ目の行の信号が、切れ目でない行の信号より大きい割合（同じ時は半分）。.5 が当てずっぽう、1 が完全。しきい値は要らない。
  しきい値を数えから決めた時の当たり: 前半の後ろ半分（波が育った後）で F1 が一番良いしきい値を選び、後半で 適合率・再現率・F1 を数える。
遷移の波の聞かせ方は 2 通り（両方数える）:
  話の中だけ:   今までと同じ。話の切れ目をまたぐ行の組は聞かせない（切れ目を教えている事になる。遷移の驚きには有利な作り）。
  流れの全部:   切れ目をまたぐ組も聞かせる（切れ目を教えない）。
注意: しきい値を選ぶ所では、前半の本物の切れ目を使う（ここは教えている）。教えずに決める形は、まだ無い。
手で置いた物: 半減期の格子、ずれを見る帯の組、信号の定義、しきい値を選ぶ所の区切り（前半の後ろ半分）。乱数は使わない。材料の語や例は出さない。
引数: --text 材料（カンマ区切り） --content 内容語の板 --out 記録.json [--k 1] [--sample 0（全部の話。>0 なら後半の話を等間隔でその数だけ）]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention"), os.path.join(HERE, "..", "intervention", "past")]

import i49zp_text_talks_run as ZP  # noqa: E402
import i49zzw_next_triple_run as W  # noqa: E402
import i49zzzd_hold_run as D  # noqa: E402

HALVES = (1, 4, 16, 64)
PAIRS = ((1, 16), (4, 16), (4, 64))
LEARN = ("話の中だけ", "流れの全部")
ALL3 = "3 席"


def cos(a, b):
    n = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    return float(np.real(np.vdot(a, b))) / n if n > 0 else 0.0


class Signals:
    """行ごとの信号。before(prev, cur) を cur を聞く前に、after() を cur を聞いた後に呼ぶ。"""

    def __init__(self, wave, hold, fast):
        self.wave, self.hold, self.fast = wave, hold, fast

    def before(self, prev, cur):
        out = {"人が替わった": float(prev[0] != cur[0])}
        keys = [self.fast.key(j, cur[j]) for j in range(3)]
        for h in self.hold.halves:
            vals = [-cos(keys[j], self.hold.v[h][j]) for j in range(3)]
            for j, s in enumerate(ZP.SEATS):
                out["持った波と合わない（%s、半減期 %d）" % (s, h)] = vals[j]
            out["持った波と合わない（%s、半減期 %d）" % (ALL3, h)] = sum(vals)
        ks = [self.fast.key(j, prev[j]) for j in range(3)]          # 1 語だけの点なので、表を行列にせずに直に引く（聞くたびに表を作り直さないため）
        sur = [-float(D.scores_k(self.wave, si, ks, [cur[si]])[0]) for si in range(3)]
        for si, s in enumerate(ZP.SEATS):
            out["遷移の驚き（%s）" % s] = sur[si]
        out["遷移の驚き（%s）" % ALL3] = sum(sur)
        return out

    def after(self):
        return {"速い波と遅い波のずれ（半減期 %d と %d）" % (a, b): sum(-cos(self.hold.v[a][j], self.hold.v[b][j]) for j in range(3))
                for a, b in PAIRS if a in self.hold.v and b in self.hold.v}


def auc(pos, neg):
    """pos の値が neg の値より大きい割合（同じ時は半分）。"""
    if not pos or not neg:
        return None
    vals = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, total = 0.0, 0, 0.0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j][0] == vals[i][0]:
            j += 1
        mid = (i + j + 1) / 2.0                      # 同じ値の順位の平均（1 から）
        total += mid * sum(x[1] for x in vals[i:j])
        i = j
    return (total - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def prf(values, labels, th):
    tp = sum(1 for v, l in zip(values, labels) if v > th and l)
    fp = sum(1 for v, l in zip(values, labels) if v > th and not l)
    fn = sum(1 for v, l in zip(values, labels) if v <= th and l)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(適合率=p, 再現率=r, F1=2 * p * r / (p + r) if p + r else 0.0)


def choose(values, labels):
    """F1 が一番良いしきい値（値と値の間）。同じなら低い方。"""
    us = sorted(set(values))
    if len(us) < 2:
        return us[0] if us else 0.0
    best = max(((a + b) / 2.0 for a, b in zip(us, us[1:])), key=lambda th: (prf(values, labels, th)["F1"], -th))
    return best


def stream(talks, attr_of, across=False, halves=HALVES, start=0, dims=None, scale=None, progress=None):
    """話を順に聞きながら、行ごとの信号を出す。→ [(話の番号, 行の番号, 切れ目か, {信号})]。start より前の話は、聞くだけ。
    初めの話の初めの行（前の行が無い）は出さない。"""
    wave = W.Wave(W.DIMS if dims is None else dims, attr_of, W.SCALE if scale is None else scale)
    hold, fast = D.Hold(wave, halves), D.Fast(wave)
    sig, out, prev = Signals(wave, hold, fast), [], None
    for ti, tris in enumerate(talks):
        tris = [tuple(t) for t in tris]
        last = prev
        for i, cur in enumerate(tris):
            s = sig.before(prev, cur) if prev is not None and ti >= start else None
            hold.hear(cur)
            if s is not None:
                s.update(sig.after())
                out.append((ti, i, i == 0, s))
            prev = cur
        if across and last is not None:
            wave.learn(last, tris[0])
        for a, b in zip(tris, tris[1:]):
            wave.learn(a, b)
        if progress and ti % progress == 0:
            print("話", ti, "/", len(talks), flush=True)
    return out


def summarize(rows, cut):
    """rows: stream の出した物。cut: この話の番号より前を、しきい値を選ぶ所に使う。"""
    names = list(rows[0][3]) if rows else []
    cal, test = [r for r in rows if r[0] < cut], [r for r in rows if r[0] >= cut]
    out = {}
    for n in names:
        pos, neg = [r[3][n] for r in test if r[2]], [r[3][n] for r in test if not r[2]]
        th = choose([r[3][n] for r in cal], [r[2] for r in cal]) if cal else None
        out[n] = dict(見分け=auc(pos, neg), 切れ目の行の平均=float(np.mean(pos)) if pos else None, それ以外の行の平均=float(np.mean(neg)) if neg else None,
                      しきい値=th, **(prf([r[3][n] for r in test], [r[2] for r in test], th) if th is not None else {}))
    return dict(行=len(test), 切れ目の行=sum(1 for r in test if r[2]), しきい値を選んだ行=len(cal), 信号=out)


def run(first, second, attr_of, halves=HALVES, dims=None, scale=None, progress=None):
    talks = [t for t, _ in first] + [t for t, _ in second]
    start, cut = len(first) // 2, len(first)
    return {name: summarize(stream(talks, attr_of, name == LEARN[1], halves, start, dims, scale, progress), cut) for name in LEARN}


def main():
    import i49m_attr_base as AB
    import i49zzz_talk_memory_run as Z
    ap = argparse.ArgumentParser()
    for k in ("text", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    talks = []
    for f in a.text.split(","):
        talks += [(t[0], t[1]) for t in ZP.talks_of(Z.read(f))[0]]
    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    seq = [talks[i] for i in ZP.order(len(talks), a.k)]
    first, second = seq[:len(seq) // 2], seq[len(seq) // 2:]
    if a.sample:
        second = second[:a.sample]            # 流れなので、間を抜かずに初めから取る
    res = dict(話=len(talks), 手で置いた値=dict(DIMS=W.DIMS, SCALE=W.SCALE, HALVES=list(HALVES), PAIRS=[list(p) for p in PAIRS]),
               **run(first, second, full.of, progress=20))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for name in LEARN:
        top = sorted(res[name]["信号"].items(), key=lambda kv: -(kv[1]["見分け"] or 0))[:6]
        print(name, res[name]["行"], res[name]["切れ目の行"], [(n, round(d["見分け"], 3), round(d.get("F1", 0), 3)) for n, d in top])
    print(json.dumps(res, ensure_ascii=False))          # 最後の 1 行: 数だけ（材料の語は入らない）。図は、この行を保存した物から作る


if __name__ == "__main__":
    main()
