"""I49zzzf: 話のテーマ波と、話の単位の束ねで、動きを引く。「だれが × 前の動き」を結び付けた鍵でも引く
（持ち主のメモ、ロードマップの追記 17 のメモ・メモ 2 と、順の 2〜4。未確定の試し）。

持ち主のメモ: 「現在保持している『だれがの波』 × 直前の文脈の波を鍵にして次の『どうした』を引きこむことで、結びついた関係のある属性、
  行動空間が読み込まれる（絞られる）かも。」
持ち主のメモ 2: 「各話の最初の 1〜2 文で出た内容語（だれが・何を・どうした）の属性波を足し合わせたものを『この話のテーマ波』として保持し、
  話題の切れ目を検知するまで減衰させない。」
82 節の読み: 遷移の表は「隣の行」しか覚えていない。「この話に一緒に出る物」を引く所が無い。
ここで足す部品:
  話の束ね（Together）: 遷移の波（I49zzw の Wave）と同じ形の表を、隣の行の組ではなく、同じ話の中の全部の行の組で作る。
  テーマ波:           話の初めの THEME 行（2）の語の波を、席ごとに足した物。薄めない。話の束ねを、この波で引く。
  結び付けた鍵（Bound）: 前の行の だれが の波と どうした の波を、位相を足して 1 つの鍵にし（掛け算）、その鍵に次の動きを覚えさせる。
動きの引き方（広い候補。その席で聞いた事の有る語の全部）:
  隣の表（前の行から）:               今までの形。比べ役。
  話の束ね（テーマ波）:               話の初めの 2 行の波で、話の束ねを引く。
  話の束ね（ここまでの全部の行）:      その話でここまでに出た全部の行の波で引く（比べ役。テーマを初めの 2 行に絞る意味が有るか）。
  隣の表＋話の束ね（テーマ波）:        2 つの点を足す（同じ式の点。重みは 1:1。手で置いた）。
  順位を足す（隣の表と話の束ね）:      2 つの点を、それぞれ候補の中での順位に直してから足す（点の大きさの違いに左右されない。重みは 1:1 のまま）。
  結び付けた鍵:                       だれが × 前の動き の鍵で引く。
  隣の表＋結び付けた鍵:               2 つの点を足す（同じく 1:1）。
話の切れ目（テーマ波の「話の初め」を、何で知るか）: --boundary で選ぶ。
  教える:   本物の切れ目を使う（上限。これで効かなければ、切れ目を見つけても効かない）。
  見つける: I49zzze の信号（--signal）が、前半で選んだしきい値を超えた行を、話の初めとする（遷移の波は別に、流れの全部を聞かせた物を使う）。
            しきい値を選ぶ所は、前半の本物の切れ目を使う。話の束ねを作る時の話の区切りは、どちらでも本物を使う（ここは教えたまま）。
数える物:
  相手に返す: 後半の問いで、次の行の動きの当たり（引き方ごと）。
  自分で続ける: 後半の話から等間隔に --free 個。初めの THEME 行は本物を渡し、その後を --steps 回続ける（出した動きは返さない。だれが は前の行の人、
    何を は本体。本体が無い時は遷移の波）。
  場面の物差し（82 節の物は弱かったので作り直した）: 出した動き（と物）が、その話の本物の続きに出て来る物である割合。
    比べ: 同じ数えを、別の話（後半で 1 つ後ろの話）の続きで数えた物（たまたまの重なり）。
出た文は材料の語を使うので、`--out` はリポジトリに入れない。
手で置いた物: THEME=2、点を足す時の重み 1:1、広い候補の作り方、続ける回数、場面の物差しの定義、信号の名前。乱数は使わない。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --out 記録.json
      [--k 1] [--sample 400] [--free 12] [--steps 8] [--show 2] [--boundary 教える] [--signal 名前]
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
import i49zzx_reply_triple_run as X  # noqa: E402
import i49zzzd_hold_run as D  # noqa: E402
import i49zzze_boundary_run as E  # noqa: E402

THEME = 2
NEAR, TOG, TOG_ALL, BND = "隣の表（前の行から）", "話の束ね（テーマ波）", "話の束ね（ここまでの全部の行）", "結び付けた鍵"
NARROW = "順位を足す（隣の表と話の束ね）"
NAMES = (NEAR, TOG, TOG_ALL, NEAR + "＋" + TOG, NARROW, BND, NEAR + "＋" + BND)
MODES = ("教える", "見つける")
SIGNAL = "遷移の驚き（%s）" % E.ALL3


def learn_together(table, tris, key=None):
    """話の束ね: 同じ話の中の、全部の行の組（自分自身は除く）を聞かせる。
    行の組ごとに table.learn(a, b) を呼ぶのと同じ結果を、席ごとに語の波を先に足しておいて、まとめて作る。key は波の置き場（Fast.key）。"""
    key = key or table._key
    n = len(tris)
    if n < 2:
        return
    keys = [[key(j, t[j]) for j in range(3)] for t in tris]
    total = [np.sum([k[j] for k in keys], axis=0) for j in range(3)]
    for b, kb in zip(tris, keys):
        for si in range(3):
            table.cnt[si][b[si]] = table.cnt[si].get(b[si], 0) + n - 1
            table.total[si] += n - 1
            for j in range(3):
                k = (total[j] - kb[j]).astype(np.complex64)
                d = table.m.setdefault((si, j), {})
                d[b[si]] = d[b[si]] + k if b[si] in d else k.copy()
                table.all[(si, j)] = table.all[(si, j)] + k if (si, j) in table.all else k.copy()


class Bound:
    """（前の行の だれが × どうした）を結び付けた鍵 → 次の動き。式は Wave.scores と同じ（似た鍵の数えが届く。いつもの分を引く）。"""

    def __init__(self, wave):
        self.wave, self.m, self.all = wave, {}, None
        self.cnt, self.total, self._mat = {}, 0, None

    def key(self, prev):
        return (self.wave._key(0, prev[0]) * self.wave._key(2, prev[2])).astype(np.complex64)

    def learn(self, prev, nxt):
        k, v = self.key(prev), nxt[2]
        self.m[v] = self.m[v] + k if v in self.m else k.copy()
        self.all = self.all + k if self.all is not None else k.copy()
        self.cnt[v] = self.cnt.get(v, 0) + 1
        self.total += 1
        self._mat = None

    def scores(self, prev, cands):
        out = np.zeros(len(cands), dtype=np.float64)
        if self.all is None:
            return out
        if self._mat is None:
            ws = list(self.m)
            self._mat = ({w: i for i, w in enumerate(ws)}, np.stack([self.m[w] for w in ws]))
        ix, M = self._mat
        k = np.conj(self.key(prev))
        near = float(np.real(k @ self.all))
        if near <= 0:
            return out
        rows = np.array([ix.get(c, -1) for c in cands])
        has = rows >= 0
        if has.any():
            out[has] = np.real(M[rows[has]] @ k) / near
        return out - np.array([self.cnt.get(c, 0) / max(self.total, 1) for c in cands])


def ranks(sc):
    """点 → 順位（一番大きい点が 0。同じ点は、同じ順位の平均）。"""
    sc = np.asarray(sc, dtype=np.float64)
    order = np.argsort(-sc, kind="stable")
    r = np.empty(len(sc), dtype=np.float64)
    r[order] = np.arange(len(sc))
    for v in np.unique(sc):
        same = sc == v
        if same.sum() > 1:
            r[same] = r[same].mean()
    return r


def theme_of(fast, lines):
    """行の並び → 席ごとに語の波を足した物（薄めない）。"""
    return [np.sum([fast.key(j, t[j]) for t in lines], axis=0).astype(np.complex64) for j in range(3)]


def verb_scores(fast, fast_t, bound, prev, talk, c2):
    """→ {引き方: 点（候補ごと）}。talk は、その話の初めから前の行まで（切れ目の見方で変わる）。"""
    near = fast.many(2, [[fast.key(j, prev[j]) for j in range(3)]], c2)[0]
    tog = fast_t.many(2, [theme_of(fast_t, talk[:THEME]), theme_of(fast_t, talk)], c2)
    bnd = bound.scores(prev, c2)
    narrow = -(ranks(near) + ranks(tog[0]))
    return {NEAR: near, TOG: tog[0], TOG_ALL: tog[1], NEAR + "＋" + TOG: near + tog[0], NARROW: narrow, BND: bnd, NEAR + "＋" + BND: near + bnd}


def pick_new(c2, sc, said):
    fresh = [i for i, c in enumerate(c2) if c not in said]
    return W.pick([c2[i] for i in fresh], [float(sc[i]) for i in fresh]) if fresh else W.pick(c2, [float(x) for x in sc])


def run(first, second, attr_of, body=None, sample=400, free=12, steps=8, show=2, boundary=MODES[0], signal=SIGNAL, dims=None, scale=None, progress=None):
    dims, scale = W.DIMS if dims is None else dims, W.SCALE if scale is None else scale
    wave, together = W.Wave(dims, attr_of, scale), W.Wave(dims, attr_of, scale)
    fast, fast_t, bound = D.Fast(wave), D.Fast(together), Bound(wave)
    heard = [set(), set(), set()]
    find = boundary == MODES[1]
    if find:                                    # 切れ目を見つける時の信号は、流れの全部を聞かせた別の波で出す
        flow = W.Wave(dims, attr_of, scale)
        hold = D.Hold(flow, E.HALVES)
        sig, cal, th, last = E.Signals(flow, hold, D.Fast(flow)), [], None, None

    def learn(tris, words=None):
        for t in tris:
            for si in range(3):
                heard[si].add(t[si])
        for a, b in zip(tris, tris[1:]):
            wave.learn(a, b)
            bound.learn(a, b)
        learn_together(together, tris, fast_t.key)
        if body is not None:
            for t in tris:
                if words is not None:
                    for si in range(3):
                        cs = sorted({w for w in words if w not in {t[j] for j in range(3) if j != si}} | {t[si]}, key=str)
                        if len(cs) >= 2:
                            body.ask(si, tuple(None if j == si else t[j] for j in range(3)), cs)
                body.learn(t)
            body.tick()

    def signal_of(prev, cur):
        s = sig.before(prev, cur) if prev is not None else None
        hold.hear(cur)
        if s is not None:
            s.update(sig.after())
        return None if s is None else s[signal]

    for ti, (tris, _) in enumerate(first):
        tris = [tuple(t) for t in tris]
        if find:
            for i, cur in enumerate(tris):
                v = signal_of(last, cur)
                if v is not None and ti >= len(first) // 2:
                    cal.append((v, i == 0))
                if i == 0 and last is not None:
                    flow.learn(last, cur)
                elif i:
                    flow.learn(tris[i - 1], cur)
                last = cur
        learn(tris)
    if find:
        th = E.choose([v for v, _ in cal], [l for _, l in cal])

    def seat(si, hist):
        return sorted(heard[si] | {t[si] for t in hist}, key=str)

    total = sum(len(t) - 1 for t, _ in second)
    every = max(total // sample, 1) if sample else 1
    free_every = max(len(second) // max(free, 1), 1)
    cnt = dict(問い=0, **{n: 0 for n in NAMES})
    found = dict(行=0, 切れ目の行=0, 見つけた=0, 当たり=0)
    scene = {}                                  # 引き方 → [出した数, 動きが本物の続きに有る, 物が本物の続きに有る, 動きが別の話の続きに有る, 物が別の話の続きに有る]
    runs, q, seg = [], 0, []
    for ti, (tris, words) in enumerate(second):
        tris = [tuple(t) for t in tris]
        for i, cur in enumerate(tris):
            if find:
                v = signal_of(last, cur)
                hit = v is not None and v > th
                found["行"] += 1
                found["切れ目の行"] += i == 0
                found["見つけた"] += hit
                found["当たり"] += hit and i == 0
                if hit:
                    seg = []
                if i == 0 and last is not None:
                    flow.learn(last, cur)
                elif i:
                    flow.learn(tris[i - 1], cur)
                last = cur
            if i and q % every == 0:
                talk = seg if find and seg else tris[:i] if not find else [tris[i - 1]]
                c2 = seat(2, tris[:i])
                sc = verb_scores(fast, fast_t, bound, tris[i - 1], talk, c2)
                cnt["問い"] += 1
                for n in NAMES:
                    cnt[n] += W.pick(c2, [float(x) for x in sc[n]]) == cur[2]
            q += bool(i)
            seg.append(cur)
        if ti % free_every == 0 and ti // free_every < free and len(tris) > THEME:
            other = [tuple(t) for t in second[(ti + 1) % len(second)][0]]
            real_v, real_o = {t[2] for t in tris[THEME:]}, {t[1] for t in tris[THEME:]}
            oth_v, oth_o = {t[2] for t in other[THEME:]}, {t[1] for t in other[THEME:]}
            for n in NAMES:
                hist, lines = list(tris[:THEME]), []
                s = scene.setdefault(n, [0, 0, 0, 0, 0])
                for _ in range(steps):
                    c1, c2 = seat(1, hist), seat(2, hist)
                    v = pick_new(c2, verb_scores(fast, fast_t, bound, hist[-1], hist, c2)[n], {t[2] for t in hist})
                    fb = W.pick(c1, [float(x) for x in fast.many(1, [[fast.key(j, hist[-1][j]) for j in range(3)]], c1)[0]])
                    o = X.fill(body, 1, (hist[-1][0], None, v), c1, fb)[0] if body is not None else fb
                    r = (hist[-1][0], o, v)
                    s[0] += 1
                    s[1] += v in real_v
                    s[2] += o in real_o
                    s[3] += v in oth_v
                    s[4] += o in oth_o
                    hist.append(r)
                    lines.append("%sが%sを%s。" % (r[0][0], r[1][0], r[2][0]))
                if len(runs) < show * len(NAMES):
                    runs.append(dict(話=ti, 引き方=n, 初めの行=["%sが%sを%s。" % tuple(w[0] for w in t) for t in tris[:THEME]], 続き=lines))
        learn(tris, words)
        if progress and ti % progress == 0:
            print("話", ti, "/", len(second), "問い", cnt["問い"], flush=True)
    return dict(切れ目=boundary, どうした=cnt, 場面=scene, 自分で続ける=runs, 広い候補の語の数=[len(s) for s in heard],
                **(dict(切れ目を見つける=dict(信号=signal, しきい値=th, **found)) if find else {}))


def main():
    import i49_foundation as FD
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    import i49zzz_talk_memory_run as Z
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    for k, v in (("k", 1), ("sample", 400), ("free", 12), ("steps", 8), ("show", 2)):
        ap.add_argument("--" + k, type=int, default=v)
    ap.add_argument("--boundary", default=MODES[0], choices=MODES)
    ap.add_argument("--signal", default=SIGNAL)
    a = ap.parse_args()
    t0 = time.time()
    talks = []
    for f in a.text.split(","):
        talks += [(t[0], t[1]) for t in ZP.talks_of(Z.read(f))[0]]
    wv = Rw.WordVecs.load(a.wave)

    def wave_of(w):
        v = wv.of(*w)
        return wv.of(w[0], "PROPN") if v is None and w[1] == "NOUN" else v

    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    seq = [talks[i] for i in ZP.order(len(talks), a.k)]
    res = dict(話=len(talks), 手で置いた値=dict(DIMS=W.DIMS, SCALE=W.SCALE, THEME=THEME),
               **run(seq[:len(seq) // 2], seq[len(seq) // 2:], full.of, FD.body(wave_of, full.of), a.sample, a.free, a.steps, a.show, a.boundary, a.signal, progress=20))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "自分で続ける"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
