"""I49zzw: 前の 3 つ組から、次の 3 つ組を出せるか（持ち主の案 3-1 を、3 つ組がもう有る材料で確かめる。未確定の試し）。

持ち主の案:「会話で必要なのは「相手の発話の3つ組 (s1, o1, v1) を受け取り、応答の3つ組 (s2, o2, v2) を出す」のを、across_pairs(61節の)で作った
  遷移波と同じ構造をつかって …」
材料は「A が B を C する」の文が並んだテキスト（tools/i49zp_text_talks_run.py と同じ切り方。話の中の続いた 2 行を、前の 3 つ組 → 次の 3 つ組 と見る）。
場面は今までと同じ: 話を並べ替えて半分に分け、前半を聞かせ、後半は問うてから聞かせる。
問い: 前の 3 つ組だけを見せて、次の 3 つ組の席ごとに、話の中の語から選ぶ（次の 3 つ組の語は 1 つも見せない）。
比べる形:
  引き継ぎ:     前の 3 つ組の、同じ席の語をそのまま答える（同じ人の話が続く、を当てるだけの比べ役）。
  遷移の数え:   （前の席, 前の語）→ 次の席の語 の回数。点 = Σ 前の席について（その前の語の後に c が出た割合 − c のいつもの割合）。
  遷移の波:     同じ数えを、語の位相（語板の属性から始める。i49zzt_triple と同じ写し方）で持つ。似た前の語の数えが届く。
  どの形も、点が全部 0 の時は「引き継ぎ」で答える形も数える（＋引き継ぎ）。
分けて数える: 次の語が、前の 3 つ組の同じ席の語と同じ（続き）／違う（替わる）。
手で置いた物: 次元、属性を位相へ写す大きさ、場面の作り方。乱数は使わない。材料の語や例は出さない。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --out 記録.json [--ks 1,2,3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49zp_text_talks_run as ZP  # noqa: E402
import i49zzt_triple as T  # noqa: E402

DIMS = 4096
SCALE = 4.0
KINDS = ("全部", "前と同じ語が続く", "前と違う語に替わる")


class Count:
    """（前の席, 前の語）→ 次の席の語 の回数。"""

    def __init__(self):
        self.n = {}        # (次の席, 前の席, 前の語) → {次の語: 回数}
        self.base = [{}, {}, {}]
        self.total = [0, 0, 0]

    def learn(self, prev, nxt):
        for si in range(3):
            self.base[si][nxt[si]] = self.base[si].get(nxt[si], 0) + 1
            self.total[si] += 1
            for j in range(3):
                d = self.n.setdefault((si, j, prev[j]), {})
                d[nxt[si]] = d.get(nxt[si], 0) + 1

    def scores(self, si, prev, cands):
        out = []
        for c in cands:
            s = 0.0
            for j in range(3):
                d = self.n.get((si, j, prev[j]))
                if d:
                    s += d.get(c, 0) / sum(d.values()) - self.base[si].get(c, 0) / max(self.total[si], 1)
            out.append(s)
        return out


class Wave(T._Tables):
    """同じ数えを、前の語の位相（属性から始める）で持つ。似た前の語の数えが届く。"""

    def __init__(self, dims, attr_of, scale):
        super().__init__(dims, attr_of, scale)
        self.m = {}        # (次の席, 前の席) → {次の語: 波}
        self.all = {}      # (次の席, 前の席) → 全部の次の語の波の足し算（いつもの分を引くため）
        self.cnt = [{}, {}, {}]
        self.total = [0, 0, 0]

    def _key(self, j, w):
        r = self.row(j, w)
        return np.exp(1j * self.th[j][r]).astype(np.complex64)

    def learn(self, prev, nxt):
        for si in range(3):
            self.cnt[si][nxt[si]] = self.cnt[si].get(nxt[si], 0) + 1
            self.total[si] += 1
            for j in range(3):
                k = self._key(j, prev[j])
                d = self.m.setdefault((si, j), {})
                d[nxt[si]] = d[nxt[si]] + k if nxt[si] in d else k.copy()
                self.all[(si, j)] = self.all[(si, j)] + k if (si, j) in self.all else k.copy()

    def scores(self, si, prev, cands):
        out = [0.0] * len(cands)
        for j in range(3):
            d = self.m.get((si, j))
            if not d:
                continue
            k = self._key(j, prev[j])
            near = float(np.real(np.vdot(k, self.all[(si, j)])))       # 似た前の語を、どれだけ聞いたか
            if near <= 0:
                continue
            for i, c in enumerate(cands):
                if c in d:
                    out[i] += float(np.real(np.vdot(k, d[c]))) / near - self.cnt[si].get(c, 0) / max(self.total[si], 1)
                else:
                    out[i] -= self.cnt[si].get(c, 0) / max(self.total[si], 1)
        return out


def pick(cands, scores, fallback=None):
    if fallback is not None and not any(s != 0 for s in scores):
        return fallback
    return cands[max(range(len(cands)), key=lambda i: (scores[i], -i))]


def count(first, second, attr_of, dims=None, scale=None):
    models = {"遷移の数え": Count(), "遷移の波": Wave(DIMS if dims is None else dims, attr_of, SCALE if scale is None else scale)}

    def hear(tris):
        for a, b in zip(tris, tris[1:]):
            for m in models.values():
                m.learn(a, b)

    for tris, _ in first:
        hear(tris)
    names = ["引き継ぎ"] + [n + x for n in models for x in ("", "＋引き継ぎ")]
    out = {si: {k: dict(問い=0, 当てずっぽう=0.0, **{n: 0 for n in names}) for k in KINDS} for si in range(3)}
    for tris, words in second:
        cands = sorted(words, key=str)
        for a, b in zip(tris, tris[1:]):
            for si in range(3):
                ans = {"引き継ぎ": a[si]}
                for n, m in models.items():
                    sc = m.scores(si, a, cands)
                    ans[n] = pick(cands, sc)
                    ans[n + "＋引き継ぎ"] = pick(cands, sc, a[si])
                for k in ("全部", "前と同じ語が続く" if b[si] == a[si] else "前と違う語に替わる"):
                    c = out[si][k]
                    c["問い"] += 1
                    c["当てずっぽう"] += 1 / len(cands)
                    for n in names:
                        c[n] += ans[n] == b[si]
        hear(tris)
    return out


def run(wave_of, attr_of, talks, ks=(1, 2, 3), dims=None, scale=None):
    plain = [(t[0], t[1]) for t in talks]
    res, total = {}, {}
    for k in ks:
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        d = count(seq[:len(seq) // 2], seq[len(seq) // 2:], attr_of, dims, scale)
        d = {ZP.SEATS[si]: v for si, v in d.items()}
        res["元の順" if k == 0 else "並べ替え %d" % k] = d
        for seat, kinds in d.items():
            for kind, c in kinds.items():
                t = total.setdefault(seat, {}).setdefault(kind, {})
                for key, v in c.items():
                    t[key] = round(t.get(key, 0) + v, 1)
        print(k, {seat: (v["全部"]["問い"], v["全部"]["引き継ぎ"], v["全部"]["遷移の数え＋引き継ぎ"], v["全部"]["遷移の波＋引き継ぎ"]) for seat, v in d.items()}, flush=True)
    res["合計"] = total
    return res


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--ks", default="1,2,3")
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
    res = dict(話=len(talks), 手で置いた値=dict(DIMS=DIMS, SCALE=SCALE), **run(wave_of, full.of, talks, ks=tuple(int(x) for x in a.ks.split(","))))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["合計"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
