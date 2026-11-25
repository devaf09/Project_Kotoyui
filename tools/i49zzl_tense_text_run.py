"""I49zzl: 時制（状況）の波を、本物の文（青空文庫など）で数える（持ち主の案、未確定の試し）。

I49zzk の作り方を本物の文に掛ける。文の終わり（述語の在る所）の切り出しを全部数える（分ける道具は使わない）。
  文: 行を 。！？!? で切った物。かぎかっこなどは除く。
  終わりの切り出し: 文の最後の TAIL 文字の、続いた切り出しの全部（1 文字から）。時制や丁寧さの印は文の終わりに出る。
2 つの席の作り方を並べる:
  1. 介入で席を作る（`--seats`）: 持ち主や介入が「時の語」として教えた語だけが席になる。文の中に有れば、その席に 1 を足す。
     長い語から先に探し、見つけた所は消す（「今まで」の中の「今」を数えない）。
  2. 全部の語を席にする（`--auto`）: 文の終わりより前の、1〜2 文字の切り出しのうち、多く出た SEATS2 個を席にする（分ける道具は使わない）。
数える物:
  1 で: 時の語の有る文の割合、席ごとに強い終わりの切り出し、独立に動く向きの数、
       並べ替えた前半で学び、後半で「終わりの切り出しの波から時の語を当てる」割合（一番多い時の語を言い続けた時と比べる）。
  2 で: 独立に動く向きの数、強い向きごとの終わりの切り出し、1 の波どうしの近さと 2 の波どうしの近さの合い方（同じ切り出しで比べる）。
手で置いた物: 同じ文を 1 回だけ数える事、時の語より後ろだけを終わりとして見る事（その時に頭の助詞の字を落とす事）、TAIL、SEATS2、数える切り出しの下限 MIN、平均を引く事、時の語の一覧（1 の時）。
材料の文は出さない。出すのは数と、文の終わりの短い切り出し（語尾）だけ。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

TAIL = 6       # 文の終わりの何文字を見るか（手で置いた値）
SEATS2 = 2000  # 2 で席にする切り出しの数（手で置いた値）
MIN = 30       # 数える終わりの切り出しの、出た回数の下限（手で置いた値）
PHI = 0.6180339887498949
CUT = re.compile(r"[。！？!?]+")
DROP = re.compile(r"[「」『』（）()〔〕［］\[\]【】〈〉《》\s　…‥・、，,.．―─－ー〜～\"'“”‘’]+$")
STRIP = re.compile(r"[「」『』（）()〔〕［］\[\]【】〈〉《》\s　\"'“”‘’]")


def sentences(lines):
    for line in lines:
        for s in CUT.split(STRIP.sub("", line)):
            s = DROP.sub("", s)
            if len(s) >= 2:
                yield s


def spans(text):
    return [text[i:j] for i in range(len(text)) for j in range(i + 1, len(text) + 1)]


def tail_of(s, seats=()):
    """文の終わりの TAIL 文字。seats を渡した時は、最後の時の語より後ろだけを見る（時の語そのものを終わりに数えない）。"""
    cut = 0
    for w in seats:
        i = s.rfind(w)
        if i >= 0:
            cut = max(cut, i + len(w))
    return DROP.sub("", s[cut:].lstrip("、，, 　はもにでがをのと"))[-TAIL:] if cut else s[-TAIL:]


def distinct(sents):
    """同じ文は 1 回だけ数える（芝居の決まり文句などが何度も出て、向きを引っ張るのを避ける）。初めて出た順。"""
    seen = {}
    for s in sents:
        seen.setdefault(s, None)
    return list(seen)


def seats_in(s, seats):
    """文の中の時の語（長い語から先に探し、見つけた所は消す）。"""
    out = []
    for w in sorted(seats, key=len, reverse=True):
        n = s.count(w)
        if n:
            out += [w] * n
            s = s.replace(w, "\0")
    return out


def golden_order(n):
    return sorted(range(n), key=lambda i: ((i + 1) * PHI) % 1.0)


class Table:
    """終わりの切り出し × 席 の回数。"""

    def __init__(self):
        self.count = {}
        self.seat = {}

    def add(self, unit, seat):
        s = self.seat.setdefault(seat, len(self.seat))
        c = self.count.setdefault(unit, {})
        c[s] = c.get(s, 0) + 1

    def waves(self, min_count=None):
        """回数が MIN 以上の切り出しの、平均を引いた波（行が切り出し）。"""
        mc = MIN if min_count is None else min_count
        units = [u for u, c in self.count.items() if sum(c.values()) >= mc]
        m = np.zeros((len(units), len(self.seat)), dtype=np.float32)
        for i, u in enumerate(units):
            for s, n in self.count[u].items():
                m[i, s] = n
        tot = m.sum(axis=0)
        n = m.sum(axis=1, keepdims=True)
        w = m / np.where(n > 0, n, 1) - (tot / max(tot.sum(), 1))[None, :]
        return units, w.astype(np.float32), n[:, 0]


def directions(w, k=10):
    if w.size == 0:
        return [], None
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    share = s / s.sum() if s.sum() > 0 else s
    return [round(float(x), 4) for x in share[:k]], (u, s, vt)


def top_by_seat(units, w, names, n=8):
    out = {}
    for name, j in names.items():
        order = np.argsort(-w[:, j], kind="stable")[:n]
        out[name] = [[units[i], round(float(w[i, j]), 4)] for i in order]
    return out


def guess(units, w, seat_names, held):
    """held: [(終わり, [時の語...]), ...]。終わりの切り出しの波を足し、一番強い席を当てる。"""
    ix = {u: i for i, u in enumerate(units)}
    major = {}
    ok = base = n = 0
    for _, ws in held:
        for x in ws:
            major[x] = major.get(x, 0) + 1
    top = max(major, key=major.get) if major else None
    for tail, ws in held:
        v = np.zeros(w.shape[1], dtype=np.float32)
        for sp in spans(tail):
            if sp in ix:
                v += w[ix[sp]]
        n += 1
        base += int(top in ws)
        if np.abs(v).sum() > 0:
            ok += int(seat_names[int(np.argmax(v))] in ws)
    return dict(問い=n, 当たり=round(ok / max(n, 1), 4), 一番多い時の語を言い続ける=round(base / max(n, 1), 4), 一番多い時の語=top)


def run_seats(sents, seats):
    order = golden_order(len(sents))
    half = len(order) // 2
    t = Table()
    with_seat = 0
    per = {}
    held = []
    for k, i in enumerate(order):
        s = sents[i]
        ws = seats_in(s, seats)
        if not ws:
            continue
        with_seat += 1
        for x in ws:
            per[x] = per.get(x, 0) + 1
        tail = tail_of(s, ws)
        if not tail:
            continue
        if k < half:
            for x in ws:
                for sp in spans(tail):
                    t.add(sp, x)
        else:
            held.append((tail, ws))
    units, w, n = t.waves()
    names = {name: j for name, j in t.seat.items()}
    share, _ = directions(w)
    seat_names = {j: name for name, j in t.seat.items()}
    return dict(文=len(sents), 時の語の有る文=with_seat, 割合=round(with_seat / max(len(sents), 1), 4),
                時の語ごとの文=dict(sorted(per.items(), key=lambda x: -x[1])), 数えた切り出し=len(units),
                独立に動く向き=share, 席ごとに強い終わりの切り出し=top_by_seat(units, w, names),
                時の語を当てる=guess(units, w, seat_names, held)), (units, w)


def run_auto(sents, k_seats=None):
    k_seats = SEATS2 if k_seats is None else k_seats
    freq = {}
    for s in sents:
        body = s[:-TAIL]
        for i in range(len(body)):
            for j in (i + 1, i + 2):
                if j <= len(body):
                    g = body[i:j]
                    freq[g] = freq.get(g, 0) + 1
    seat_list = sorted(freq, key=lambda g: (-freq[g], g))[:k_seats]
    seat_ix = {g: i for i, g in enumerate(seat_list)}
    tc = {}
    for s in sents:
        for x in set(spans(tail_of(s))):
            tc[x] = tc.get(x, 0) + 1
    unit_list = sorted((x for x, c in tc.items() if c >= (MIN if MIN else 1)), key=lambda x: (-tc[x], x))
    unit_ix = {x: i for i, x in enumerate(unit_list)}
    nu, ns = len(unit_list), len(seat_list)
    flat = np.zeros(nu * ns, dtype=np.float64)
    buf = []
    for s in sents:
        body = s[:-TAIL]
        hit = {seat_ix[body[i:j]] for i in range(len(body)) for j in (i + 1, i + 2) if j <= len(body) and body[i:j] in seat_ix}
        us = [unit_ix[x] for x in spans(tail_of(s)) if x in unit_ix]
        if hit and us:
            buf.append((np.array(us, dtype=np.int64)[:, None] * ns + np.array(sorted(hit), dtype=np.int64)[None, :]).ravel())
        if len(buf) >= 4096:
            flat += np.bincount(np.concatenate(buf), minlength=nu * ns)
            buf = []
    if buf:
        flat += np.bincount(np.concatenate(buf), minlength=nu * ns)
    t = Table()
    t.seat = {g: i for i, g in enumerate(seat_list)}
    m = flat.reshape(nu, ns)
    for i, x in enumerate(unit_list):
        nz = np.nonzero(m[i])[0]
        t.count[x] = {int(j): float(m[i, j]) for j in nz}
    units, w, n = t.waves()
    share, svd = directions(w)
    dirs = []
    if svd is not None:
        u, s, vt = svd
        names = {j: g for g, j in t.seat.items()}
        for d in range(min(5, len(s))):
            col = u[:, d] * s[d]
            pos = np.argsort(-col, kind="stable")[:8]
            neg = np.argsort(col, kind="stable")[:8]
            sp = np.argsort(-vt[d], kind="stable")[:8]
            sn = np.argsort(vt[d], kind="stable")[:8]
            dirs.append(dict(取り分=share[d], 正の側の終わり=[units[i] for i in pos], 負の側の終わり=[units[i] for i in neg],
                             正の側の席=[names[i] for i in sp], 負の側の席=[names[i] for i in sn]))
    return dict(席=len(t.seat), 数えた切り出し=len(units), 独立に動く向き=share, 強い向き=dirs), (units, w)


def agreement(a, b):
    """同じ切り出しどうしで、1 の波の近さと 2 の波の近さの相関（右上の三角）。"""
    ua, wa = a
    ub, wb = b
    ib = {u: i for i, u in enumerate(ub)}
    both = [(i, ib[u]) for i, u in enumerate(ua) if u in ib]
    if len(both) < 3:
        return dict(共通の切り出し=len(both), 相関=None)

    def cosm(m):
        n = np.linalg.norm(m, axis=1)
        x = m / np.where(n > 0, n, 1)[:, None]
        return x @ x.T

    ca = cosm(wa[[i for i, _ in both]])
    cb = cosm(wb[[j for _, j in both]])
    iu = np.triu_indices(len(both), 1)
    return dict(共通の切り出し=len(both), 相関=round(float(np.corrcoef(ca[iu], cb[iu])[0, 1]), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)      # UTF-8 の文字の並び（1 行 1 発話など）。カンマで複数
    ap.add_argument("--out", required=True)
    ap.add_argument("--seats", default="今,さっき,昨日,明日,これから,いつも,ずっと,今まで,もう,まだ,今日,先程")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    lines = []
    for f in a.text.split(","):
        lines += open(f, encoding="utf-8", errors="replace").read().splitlines()
    if a.limit:
        lines = lines[:a.limit]
    all_sents = list(sentences(lines))
    sents = distinct(all_sents)
    seats = [x for x in a.seats.split(",") if x]
    r1, w1 = run_seats(sents, seats)
    res = dict(行=len(lines), 切った文=len(all_sents), 同じ文を除いた文=len(sents), 手で置いた値=dict(TAIL=TAIL, SEATS2=SEATS2, MIN=MIN), 時の語=seats, 介入で席を作る=r1)
    if a.auto:
        r2, w2 = run_auto(sents)
        res["全部の語を席にする"] = r2
        res["1と2の近さの合い方"] = agreement(w1, w2)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
