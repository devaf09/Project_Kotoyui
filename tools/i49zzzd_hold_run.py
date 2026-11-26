"""I49zzzd: 「だれが」を手書きの決めではなく、持った波で選ぶ。持った波で動きも引いて、場面が保たれるかを見る
（持ち主の案、ロードマップの追記 16 の案 1-1 と、81 節で見えた「話題をつなぎ止める所が無い」の確かめ。未確定の試し）。

持ち主の案 1-1: 「『だれが』の席（保持）: 内面波の中〜遅い帯域（半減期 16〜1024）が持つ『この話でよく出てきた人物の属性波』に近い候補
  （正の共鳴）を優先する。」
今までの形（75 節・78 節）は、「その話でもう出た人」という集まりを手で作って、その中から選んでいた。話の切れ目も手で教えていた。
ここでは、聞いた行を 1 本の流れとして、席ごとに波を足し込み、半減期で薄めて持つ（Hold）。話の切れ目は教えない。集まりも作らない。
だれが の選び方（広い候補。その席で聞いた事の有る語の全部から選ぶ）:
  手書き（話の中の人から、遷移の波）: 今までの形。比べ役。
  遷移の波だけ:                       前の行だけから引く。比べ役。
  持った波との共鳴（属性の波）:       候補の語の属性の波と、持った波の内積が大きい語。似た人にも届く。
  持った波との共鳴（語そのもの）:     語ごとに薄めて数えた回数が大きい語。（語ごとに直交した波を持たせた時の極限。）
  持った波で遷移を引く:               遷移の波を、前の行の語の波ではなく、持った波で引く。半減期を短くすると「遷移の波だけ」に近づく。
どうした の選び方（広い候補）: 遷移の波だけ／持った波で遷移を引く。それぞれ、その話で出た動きを除く形も数える（除く集まりは手書き。案 1-2 は別に確かめる）。
半減期は行の数で数える（HALVES。持ち主の帯域の案から、手で置いた格子）。
自分で続ける（--free 個の話で、初めの 1 行から --steps 回）: だれが は初めの人のまま、どうした を上の形で選び（出した動きは返さない）、何を は本体で埋める。
  場面の物差し（数えるだけ）: 出した動きが、初めの行の動きと「同じ話に出た事が有る」割合（聞いた話の中で）。隣の動き同士の同じ割合。
  比べ: 本物の話の、同じ位置の動きで数えた割合。
場面は今までと同じ（話を並べ替えて半分に分け、前半を聞かせる。後半の話で試し、試した後に聞かせる）。
出た文は材料の語を使うので、`--out` はリポジトリに入れない。
手で置いた物: 半減期の格子、場面の作り方、広い候補の作り方、「出た動きを除く」の集まり、場面の物差しの定義。乱数は使わない。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --out 記録.json [--k 1] [--sample 0（全部）] [--free 30] [--steps 10] [--show 2]
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

HALVES = (1, 4, 16, 64, 256, 1024)
HAND, LAST = "手書き（話の中の人から、遷移の波）", "遷移の波だけ"
ATTR, WORD, HELD = "持った波との共鳴（属性の波）", "持った波との共鳴（語そのもの）", "持った波で遷移を引く"
NEW = "（出た動きを除く）"


def name_of(kind, h):
    return "%s 半減期 %d" % (kind, h)


class Hold:
    """聞いた行を、席ごとに足し込み、半減期（行の数）で薄めて持つ。消さない。"""

    def __init__(self, wave, halves=HALVES):
        self.wave, self.halves = wave, tuple(halves)
        self.v = {h: [np.zeros(wave.dims, dtype=np.complex64) for _ in range(3)] for h in self.halves}
        self.n = {h: [{}, {}, {}] for h in self.halves}

    def fork(self):
        o = Hold.__new__(Hold)
        o.wave, o.halves = self.wave, self.halves
        o.v = {h: [x.copy() for x in v] for h, v in self.v.items()}
        o.n = {h: [dict(d) for d in n] for h, n in self.n.items()}
        return o

    def hear(self, tri):
        keys = [self.wave._key(j, tri[j]) for j in range(3)]
        for h in self.halves:
            f = np.float32(0.5 ** (1.0 / h))
            for j in range(3):
                self.v[h][j] = (self.v[h][j] * f + keys[j]).astype(np.complex64)
                d = self.n[h][j]
                for w in d:
                    d[w] *= float(f)
                d[tri[j]] = d.get(tri[j], 0.0) + 1.0

    def attr(self, h, j, cands):
        v = self.v[h][j]
        return [float(np.real(np.vdot(self.wave._key(j, c), v))) for c in cands]

    def word(self, h, j, cands):
        d = self.n[h][j]
        return [d.get(c, 0.0) for c in cands]


def scores_k(wave, si, ks, cands):
    """Wave.scores と同じ式。ただし、前の行の語の波の代わりに、渡した波 ks[j] で引く。"""
    out = [0.0] * len(cands)
    for j in range(3):
        d = wave.m.get((si, j))
        if not d:
            continue
        near = float(np.real(np.vdot(ks[j], wave.all[(si, j)])))
        if near <= 0:
            continue
        for i, c in enumerate(cands):
            base = wave.cnt[si].get(c, 0) / max(wave.total[si], 1)
            out[i] += (float(np.real(np.vdot(ks[j], d[c]))) / near if c in d else 0.0) - base
    return out


class Fast:
    """scores_k と同じ式を、幾つもの引き方についてまとめて計算する（表を行列にして持つ。聞いたら作り直す）。"""

    def __init__(self, wave):
        self.wave, self._t, self._at, self._keys = wave, {}, None, [{}, {}, {}]

    def _table(self, si, j):
        if self._at != tuple(self.wave.total):
            self._t, self._at = {}, tuple(self.wave.total)
        if (si, j) not in self._t:
            d = self.wave.m.get((si, j)) or {}
            ws = list(d)
            self._t[(si, j)] = ({w: i for i, w in enumerate(ws)}, np.stack([d[w] for w in ws]) if ws else None)
        return self._t[(si, j)]

    def key(self, j, w):
        k = self._keys[j]
        if w not in k:
            k[w] = self.wave._key(j, w)
        return k[w]

    def many(self, si, K, cands):
        """K: 引き方ごとの、席 3 つの波。→ 点（引き方 × 候補）。"""
        wave, out = self.wave, np.zeros((len(K), len(cands)), dtype=np.float64)
        base = np.array([wave.cnt[si].get(c, 0) / max(wave.total[si], 1) for c in cands])
        for j in range(3):
            ix, M = self._table(si, j)
            if M is None:
                continue
            Kj = np.conj(np.stack([k[j] for k in K]))
            near = np.real(Kj @ wave.all[(si, j)]).astype(np.float64)
            rows = np.array([ix.get(c, -1) for c in cands])
            has = rows >= 0
            vals = np.zeros((len(K), len(cands)), dtype=np.float64)
            if has.any():
                vals[:, has] = np.real(Kj @ M[rows[has]].T)
            ok = near > 0
            out[ok] += vals[ok] / near[ok, None] - base[None, :]
        return out

    def attr(self, hold, j, cands):
        """持った波との共鳴（属性の波）を、半減期の全部について。→ 点（半減期 × 候補）。"""
        C = np.stack([self.key(j, c) for c in cands])
        return np.real(np.conj(C) @ np.stack([hold.v[h][j] for h in hold.halves]).T).T


def _new(c2, sc, verbs):
    fresh = [i for i, c in enumerate(c2) if c not in verbs]
    return W.pick([c2[i] for i in fresh], [sc[i] for i in fresh]) if fresh else W.pick(c2, sc)


def picks(fast, hold, prev, hist, seat):
    """→ ({選び方: だれが}, {選び方: どうした})。seat[si] は、その席に置いて良い語（並べた物）。"""
    c0, c2 = seat[0], seat[2]
    K = [[fast.key(j, prev[j]) for j in range(3)]] + [hold.v[h] for h in hold.halves]
    s0, s2 = fast.many(0, K, c0), fast.many(2, K, c2)
    people, verbs = {t[0] for t in hist}, {t[2] for t in hist}
    ix = [i for i, c in enumerate(c0) if c in people]
    who = {HAND: W.pick([c0[i] for i in ix], [s0[0][i] for i in ix]) if ix else prev[0], LAST: W.pick(c0, s0[0])}
    act = {LAST: W.pick(c2, s2[0]), LAST + NEW: _new(c2, s2[0], verbs)}
    at = fast.attr(hold, 0, c0)
    for n, h in enumerate(hold.halves):
        who[name_of(ATTR, h)] = W.pick(c0, at[n])
        who[name_of(WORD, h)] = W.pick(c0, hold.word(h, 0, c0))
        who[name_of(HELD, h)] = W.pick(c0, s0[n + 1])
        act[name_of(HELD, h)] = W.pick(c2, s2[n + 1])
        act[name_of(HELD, h) + NEW] = _new(c2, s2[n + 1], verbs)
    return who, act


def act_new(fast, hold, prev, hist, c2, h=None):
    """自分で続ける時の動き: 前の行から（h が無い時）か、持った波（半減期 h）から引く。出た動きは返さない。"""
    k = [fast.key(j, prev[j]) for j in range(3)] if h is None else hold.v[h]
    return _new(c2, fast.many(2, [k], c2)[0], {t[2] for t in hist})


def run(first, second, attr_of, body=None, halves=HALVES, sample=0, free=30, steps=10, show=2, dims=None, scale=None):
    wave = W.Wave(W.DIMS if dims is None else dims, attr_of, W.SCALE if scale is None else scale)
    hold, fast = Hold(wave, halves), Fast(wave)
    heard_seat = [set(), set(), set()]
    mates = {}                                  # 動き → 同じ話に出た事の有る動き

    def learn(tris, words=None):
        for t in tris:
            for si in range(3):
                heard_seat[si].add(t[si])
        vs = {t[2] for t in tris}
        for v in vs:
            mates.setdefault(v, set()).update(vs)
        for a, b in zip(tris, tris[1:]):
            wave.learn(a, b)
        if body is not None:
            for t in tris:
                if words is not None:
                    for si in range(3):
                        cs = sorted({w for w in words if w not in {t[j] for j in range(3) if j != si}} | {t[si]}, key=str)
                        if len(cs) >= 2:
                            body.ask(si, tuple(None if j == si else t[j] for j in range(3)), cs)
                body.learn(t)
            body.tick()

    def seats(hist):
        return [sorted(heard_seat[si] | {t[si] for t in hist}, key=str) for si in range(3)]

    def together(a, b):
        return b in mates.get(a, ())

    for tris, _ in first:
        tris = [tuple(t) for t in tris]
        for t in tris:
            hold.hear(t)
        learn(tris)
    total = sum(len(t) - 1 for t, _ in second)
    every = max(total // sample, 1) if sample else 1
    free_every = max(len(second) // max(free, 1), 1)
    cnt = [dict(問い=0), dict(問い=0)]
    scene = {}                                  # 選び方 → 位置 → [数, 初めの動きと同じ話, 隣の動きと同じ話]
    runs, q, said_same = [], 0, {}

    def mark(name, step, with_start, with_prev):
        s = scene.setdefault(name, {}).setdefault(step, [0, 0, 0])
        s[0] += 1
        s[1] += bool(with_start)
        s[2] += bool(with_prev)

    for ti, (tris, words) in enumerate(second):
        tris = [tuple(t) for t in tris]
        start = hold.fork()
        for i in range(1, len(tris)):
            hold.hear(tris[i - 1])
            if q % every == 0:
                who, act = picks(fast, hold, tris[i - 1], tris[:i], seats(tris[:i]))
                for c, d, si in ((cnt[0], who, 0), (cnt[1], act, 2)):
                    c["問い"] += 1
                    for n, w in d.items():
                        c[n] = c.get(n, 0) + (w == tris[i][si])
            q += 1
        hold.hear(tris[-1])
        if body is not None and ti % free_every == 0 and ti // free_every < free:
            for step in range(1, min(len(tris), steps + 1)):
                mark("本物の話", step, together(tris[0][2], tris[step][2]), together(tris[step - 1][2], tris[step][2]))
            names = [(LAST + NEW, None)] + [(name_of(HELD, x) + NEW, x) for x in halves]
            for name, half in names:
                h, hist, lines = start.fork(), [tris[0]], []
                h.hear(tris[0])
                for step in range(1, steps + 1):
                    st = seats(hist)
                    v = act_new(fast, h, hist[-1], hist, st[2], half)
                    o, _ = X.fill(body, 1, (tris[0][0], None, v), st[1], W.pick(st[1], wave.scores(1, hist[-1], st[1])))
                    r = (tris[0][0], o, v)
                    mark(name, step, together(tris[0][2], v), together(hist[-1][2], v))
                    said_same[name] = said_same.get(name, 0) + (r in hist)
                    hist.append(r)
                    h.hear(r)
                    lines.append("%sが%sを%s。" % (r[0][0], r[1][0], r[2][0]))
                if len(runs) < show * len(names):
                    runs.append(dict(話=ti, 選び方=name, 初めの行="%sが%sを%s。" % tuple(w[0] for w in tris[0]), 続き=lines))
        learn(tris, words)
        if ti % 20 == 0:
            print("話", ti, "/", len(second), "問い", cnt[0]["問い"], flush=True)
    return dict(だれが=cnt[0], どうした=cnt[1], 場面=scene, 同じ物をもう一度出した数=said_same, 自分で続ける=runs,
                広い候補の語の数=[len(s) for s in heard_seat])


def main():
    import i49_foundation as FD
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    import i49zzz_talk_memory_run as Z
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    for k, v in (("k", 1), ("sample", 0), ("free", 30), ("steps", 10), ("show", 2)):
        ap.add_argument("--" + k, type=int, default=v)
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
    res = dict(話=len(talks), 手で置いた値=dict(DIMS=W.DIMS, SCALE=W.SCALE, HALVES=list(HALVES)),
               **run(seq[:len(seq) // 2], seq[len(seq) // 2:], full.of, FD.body(wave_of, full.of), HALVES, a.sample, a.free, a.steps, a.show))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("だれが", "どうした", "場面", "同じ物をもう一度出した数", "広い候補の語の数", "秒")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
