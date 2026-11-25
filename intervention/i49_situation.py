"""I49 状況波の部品: 文の終わりから、会話の状況（時・間柄・言葉づかい・やり取り）の波を作って使う（未確定の試しをまとめた物）。

考え: 状況は会話の中でしばらく続く。なら、前の文の終わりと今の文の終わりは結び付く。
  時の語も、分ける道具も、手で打つ語も使わない。文の終わりの切り出し（1 文字から TAIL 文字まで全部）を数えて足すだけ。
ここに有る物:
  文を切る: sentences, tail_of, spans, distinct, golden_order
  組を作る: pairs_of（同じ発話の中 / 行をまたぐ）, runs_of（空の行を場の切れ目として、続きごとの発話）, across_pairs（1 個前の発話 → 今の発話）
  波: Assoc（今の文の終わりの切り出し × 前の文の終わりの切り出し。平均を引いた割合）, overlap（比べ役。同じ切り出しの数）
  書き替え: joins（つなぎ目の表）, rewrite（行の終わりだけを、状況に合う物へ替える）
  まとめ: SituationWave（材料の行から上の全部を作り、合い方・並べ替え・書き替えを 1 つの口で使う）
  読むための印: marks（丁寧・過去・問い・古風。手で決めた物で、学習には使わない）
数えた結果は docs/i49_inner_vessel.md の 60〜63 節。状況波の仮の名前: 時の波（場に付き、薄れる）、間柄の波（話し手と相手の組）、
  言葉づかいの波、やり取りの波（次の話し手で反転する）。
分かっている弱い所: 効くのは文の終わりの TAIL 文字だけ。打ち消しを守らない。語の途中で切る事が有る。本体（数えて選ぶ）とはまだつないでいない。
手で置いた物: TAIL、SEATS、MIN、JOIN、SEEN、同じ組を 1 回だけ数える事、平均を引く事、印の決め方。
乱数は使わない。float32。
"""
from __future__ import annotations

import re

import numpy as np

TAIL = 6       # 文の終わりの何文字を見るか（手で置いた値）
SEATS = 1000   # 席にする前の文の終わりの切り出しの数（手で置いた値）
MIN = 30       # 数える今の文の終わりの切り出しの、出た回数の下限（手で置いた値）
JOIN = 2       # つなぎ目の左側として見る文字の数（手で置いた値）
SEEN = 3       # つなぎ目が材料に出た回数の下限（手で置いた値）
PHI = 0.6180339887498949
CUT = re.compile(r"[。！？!?]+")
DROP = re.compile(r"[「」『』（）()〔〕［］\[\]【】〈〉《》\s　…‥・、，,.．―─－ー〜～\"'“”‘’]+$")
STRIP = re.compile(r"[「」『』（）()〔〕［］\[\]【】〈〉《》\s　\"'“”‘’]")
POLITE = ("ます", "です", "ました", "でした", "ません", "でしょう", "ましょう", "ございま")
TRAIL = "よねわなのかぞさぜ"
OLD = ("ござ", "まする", "おりま", "じゃ", "ぬ", "申し")
MODES = ("同じ話し手が続ける", "次の話し手が受ける")


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


def golden_order(n):
    return sorted(range(n), key=lambda i: ((i + 1) * PHI) % 1.0)


def marks(tail, old=False):
    """読むための印（丁寧・過去・問い。old=True なら古風も）。手で決めた物で、学習には使わない。"""
    core = tail.rstrip(TRAIL)
    m = dict(丁寧=any(p in tail for p in POLITE), 過去=core.endswith("た"), 問い=tail.endswith(("か", "かい", "かね", "の")))
    if old:
        m["古風"] = any(x in tail for x in OLD)
    return m


def pairs_of(lines):
    """→ (同じ行の中の組, 行をまたぐ組)。組は (前の文の終わり, 今の文の終わり)。同じ組は 1 回だけ。"""
    inside, across = {}, {}
    last = None
    for line in lines:
        ss = [tail_of(s) for s in sentences([line])]
        if not ss:
            last = None
            continue
        if last is not None:
            across.setdefault((last, ss[0]), None)
        for a, b in zip(ss, ss[1:]):
            inside.setdefault((a, b), None)
        last = ss[-1]
    return list(inside), list(across)


def runs_of(lines):
    """空の行を続きの切れ目として、続きごとの発話の並びにする。発話は (初めの文の終わり, 最後の文の終わり)。"""
    runs, cur = [], []
    for line in lines:
        ss = [tail_of(x) for x in sentences([line])]
        if not line.strip():
            if cur:
                runs.append(cur)
            cur = []
        elif ss:
            cur.append((ss[0], ss[-1]))
    if cur:
        runs.append(cur)
    return runs


def across_pairs(runs):
    """1 個前の発話の最後の文の終わり → 今の発話の初めの文の終わり（続きの切れ目はまたがない。同じ組は 1 回）。"""
    across = {}
    for r in runs:
        for i in range(1, len(r)):
            across.setdefault((r[i - 1][1], r[i][0]), None)
    return across


class Assoc:
    """今の文の終わりの切り出し × 前の文の終わりの切り出し（席）の回数から、平均を引いた波を作る。"""

    def __init__(self, train, seats=None, min_count=None):
        seats = SEATS if seats is None else seats
        mc = MIN if min_count is None else min_count
        pc, cc = {}, {}
        for p, c in train:
            for x in set(spans(p)):
                pc[x] = pc.get(x, 0) + 1
            for x in set(spans(c)):
                cc[x] = cc.get(x, 0) + 1
        self.seats = sorted(pc, key=lambda x: (-pc[x], x))[:seats]
        self.units = sorted((x for x, n in cc.items() if n >= mc), key=lambda x: (-cc[x], x))
        self.six = {x: i for i, x in enumerate(self.seats)}
        self.uix = {x: i for i, x in enumerate(self.units)}
        nu, ns = len(self.units), len(self.seats)
        flat = np.zeros(nu * ns, dtype=np.float64)
        buf = []
        for p, c in train:
            us, ps = self.ids(c, self.uix), self.ids(p, self.six)
            if len(us) and len(ps):
                buf.append((us[:, None] * ns + ps[None, :]).ravel())
            if len(buf) >= 4096:
                flat += np.bincount(np.concatenate(buf), minlength=nu * ns)
                buf = []
        if buf:
            flat += np.bincount(np.concatenate(buf), minlength=nu * ns)
        m = flat.reshape(nu, ns)
        tot = m.sum(axis=0)
        n = m.sum(axis=1, keepdims=True)
        self.w = (m / np.where(n > 0, n, 1) - (tot / max(tot.sum(), 1))[None, :]).astype(np.float32) if nu and ns else np.zeros((nu, ns), dtype=np.float32)

    @staticmethod
    def ids(tail, ix):
        return np.array(sorted({ix[x] for x in spans(tail) if x in ix}), dtype=np.int64)

    def score(self, prev, cur):
        us, ps = self.ids(cur, self.uix), self.ids(prev, self.six)
        return float(self.w[np.ix_(us, ps)].sum()) if len(us) and len(ps) else 0.0


def overlap(prev, cur):
    return len(set(spans(prev)) & set(spans(cur)))


def joins(sents):
    """つなぎ目の表: 文の終わりを k 文字（1〜TAIL）で切った時の、左の JOIN 文字 → 右の終わり → 出た回数。
    分ける道具は使わない。「この 2 文字の後に、この終わりが続いた事が有る」を数えるだけ。"""
    out = {}
    for s in sents:
        for k in range(1, TAIL + 1):
            if len(s) >= k + JOIN:
                d = out.setdefault(s[-k - JOIN:-k], {})
                e = s[-k:]
                d[e] = d.get(e, 0) + 1
    return out


def rewrite(sent, probe, assoc, table, n=1):
    """行の終わりだけを書き替える。切る所（終わりの 1〜TAIL 文字）ごとに、左の JOIN 文字の後に材料で SEEN 回以上続いた終わりへ替え、
    状況（probe）との合い方が一番高い物を取る。元の行と同じ物は除く。→ 上から n 個の (書き替えた行, 点)。"""
    best = {}
    for k in range(1, TAIL + 1):
        if len(sent) < k + JOIN:
            continue
        for e, c in table.get(sent[-k - JOIN:-k], {}).items():
            new = sent[:-k] + e
            if c < SEEN or new == sent:
                continue
            sc = sum(assoc.score(p, tail_of(new)) for p in probe)
            if new not in best or sc > best[new]:
                best[new] = sc
    return sorted(best.items(), key=lambda x: (-x[1], x[0]))[:n]


class SituationWave:
    """材料の行から状況波を作り、1 つの口で使う。
    lines: 1 行 1 発話。空の行は場の切れ目（無ければ、全部が 1 つの続き）。
    mode: 「同じ話し手が続ける」（同じ発話の中の続いた 2 文）か「次の話し手が受ける」（1 個前の発話 → 今の発話）。
    状況は「前の文がこう終わっていた」という終わりの並び（prev_tails）で渡す。"""

    def __init__(self, lines):
        inside, _ = pairs_of(lines)
        self.assoc = {MODES[0]: Assoc(inside), MODES[1]: Assoc(list(across_pairs(runs_of(lines))))}
        self.table = joins(distinct(list(sentences(lines))))

    def fit(self, prev_tails, cur, mode=MODES[0]):
        """今の文の終わり cur が、状況とどれだけ合うか（点の足し算）。"""
        a = self.assoc[mode]
        return sum(a.score(p, tail_of(cur)) for p in prev_tails)

    def rank(self, prev_tails, cands, mode=MODES[0]):
        """候補の終わりを、状況と合う順に並べる。"""
        return sorted(cands, key=lambda c: (-self.fit(prev_tails, c, mode), c))

    def rewrite(self, sent, prev_tails, mode=MODES[0], n=1):
        """行の終わりだけを、状況に合う物へ書き替える。→ 上から n 個の (書き替えた行, 点)。"""
        return rewrite(sent, tuple(prev_tails), self.assoc[mode], self.table, n=n)
