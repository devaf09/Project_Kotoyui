"""I49zzy 形を組む所の直し: 語幹の終わりの字で流れを選び、前の字を見て流れの表記を選ぶ（I42 の逆引きに足す部品。未確定の試し）。

元: 73 節で「壊れるのは語幹の形を戻す所（五段の音便など）」と見た。確かめると、訳は 2 つだった。
  1. 流れの選び方: その語でその意味を聞いた事が無い時、語の類（動詞）の表から「た」を取る。語幹が「む・ぶ・ぐ」で終わる語には「だ」が要る。
  2. 流れの表記: 「だ」の表記を、次に付く語だけで引くので、文の終わりの「だ」（断定）の形「に・な」と混ざる（読ん + に）。
直し（どちらも数えるだけ。辞書も活用の表も使わない）:
  EndFlow     （語の類, 語幹の終わりの 1 字, 意味の組）→ 流れ の回数。その語の表が無い時に使う。選び方は 2 通り:
                細かい表を先（終わりの字の表の割合で選ぶ。語の類の表の割合を 1 回ぶん足す）／掛け合わせ（更に語の類の表の割合を掛ける）。
  PrevSurface （流れの語, 前の表記の終わりの 1 字）→ 表記 の回数。I42 の表（流れの語, 次の語）と、割合を掛け合わせて選ぶ。
  build       語幹・意味の組 → 形（語幹の表記 ＋ 流れの表記）。fix=False なら I42 のまま。
  mix / EndFlow.mixed / stem_mixed  広い表と細かい表を、数えの多さを重みにして混ぜる形（κ。追記 16 の案 2）。
手で置いた物: κ、終わりの 1 字を見る事、前の 1 字を見る事、割合を掛ける事、細かい表に広い表の割合を 1 回ぶん足す事。乱数は使わない。
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "past"))

import i42_fused_flow as F  # noqa: E402


def pick(c1, c2, prior=True):
    """2 つの数え（広い表 c1、細かい表 c2）から 1 つ選ぶ。細かい表は、広い表の割合を 1 回ぶん足してから割合にする（少ない数えに振り回されない）。
    点 = 広い表の割合 × 細かい表の割合。候補は広い表に出た物。広い表が空なら細かい表の一番多い物。どちらも空なら None。"""
    if not c1:
        return max(sorted(c2, key=str), key=lambda y: c2[y]) if c2 else None
    n1, n2 = sum(c1.values()), sum((c2 or {}).values())
    return max(sorted(c1, key=str), key=lambda y: (c1[y] / n1 if prior else 1.0) * (((c2 or {}).get(y, 0) + c1[y] / n1) / (n2 + 1)))


def mix(levels, kappa):
    """広い表から細かい表へ並べた数え（levels）を、数えの多さを重みにして混ぜる（持ち主がもらった案。ロードマップの追記 16 の案 2）。
    割合(y) = (その段の回数(y) + κ × 1 つ広い段の割合(y)) / (その段の総回数 + κ)。一番広い段は、そのままの割合。空の段は飛ばす。
    下限を置かず、1〜2 回しか出ていない段は広い段の割合が強く効き、何度も出た段はその段の数えが勝つ。→ {y: 割合}（空なら {}）。"""
    p = {}
    for c in levels:
        if not c:
            continue
        n = sum(c.values())
        if not p:
            p = {y: v / n for y, v in c.items()}
        else:
            p = {y: (c.get(y, 0) + kappa * p.get(y, 0.0)) / (n + kappa) for y in set(c) | set(p)}
    return p


def best(p):
    return max(sorted(p, key=str), key=lambda y: p[y]) if p else None


class EndFlow:
    def __init__(self, train):
        self.t = defaultdict(Counter)
        for its in train:
            for sy, lem, fl, _, fs in its:
                pos = F._pos(sy)
                self.t[(F.wclass(lem, pos), lem[-1:], F.meaning(fl, fs, pos))][fl] += 1

    def flow(self, rev, lem, m, pos="VERB", quote=False, prior=False):
        """その語の表が無い時だけ、語の類の表と、終わりの字の表の割合を掛けて選ぶ。語の類の表が無ければ I42 の逆引きの表。"""
        if not rev.stem.get((lem, m)):
            k = F.wclass(lem, pos)
            c1 = rev.ctab.get((k, m))
            if c1:
                return pick(c1, self.t.get((k, lem[-1:], m)), prior=prior)
        return rev.table(lem, m, pos, quote)

    def mixed(self, rev, lem, m, pos="VERB", quote=False, kappa=3.0):
        """語の類の表 → 終わりの字の表 → その語の表 を、κ で混ぜて選ぶ。どの表も空なら I42 の逆引きの表。"""
        k = F.wclass(lem, pos)
        fl = best(mix([rev.ctab.get((k, m)), self.t.get((k, lem[-1:], m)), rev.stem.get((lem, m))], kappa))
        return rev.table(lem, m, pos, quote) if fl is None else fl


def stem_mixed(surf, lem, fl, pos="VERB", kappa=3.0):
    """語幹の形を戻す所も、広い決まりから細かい決まりへ κ で混ぜる。I42 は、細かい表が 1 回でも有ればそれを取る。
    段: 終わりの 1 字 → 終わりの 2 字 → 活用の種類と 1 字 → 活用の種類と 2 字 → その語そのもの。書き替え（終わりの字 → 替える字）で持つ。"""
    f = fl[0] if fl else F.END
    k = surf.kind(lem, pos)
    own = Counter()
    for x, v in (surf.exact.get((lem, f)) or {}).items():
        cp = os.path.commonprefix([lem, x])
        own[(lem[len(cp):], x[len(cp):])] += v
    levels = [surf.rule.get(key) for key in ((lem[-1:], f), (lem[-2:], f), (k, lem[-1:], f), (k, lem[-2:], f))] + [own]
    levels = [Counter({r: v for r, v in c.items() if lem.endswith(r[0])}) if c else None for c in levels]
    r = best(mix(levels, kappa))
    return lem if r is None else lem[: len(lem) - len(r[0])] + r[1]


class PrevSurface:
    def __init__(self, train, fsurf):
        self.fsurf = fsurf
        self.t = defaultdict(Counter)
        for its in train:
            for _, _, fl, stem, fs in its:
                for i, (l, x) in enumerate(zip(fl, fs)):
                    self.t[(l, (stem if i == 0 else fs[i - 1])[-1:])][x] += 1

    def form(self, fl, stem):
        out, prev = [], stem
        for i, l in enumerate(fl):
            n = fl[i + 1] if i + 1 < len(fl) else F.END
            x = pick(self.fsurf.t.get((l, n)) or self.fsurf.t.get((l, "*")), self.t.get((l, prev[-1:])))
            x = l if x is None else x
            out.append(x)
            prev = x
        return "".join(out)


class Former:
    """I42 の部品と、上の 2 つをまとめて、語幹・意味の組から形を組む。"""

    def __init__(self, its):
        self.fused, self.surf, self.fsurf = F.Fused(its), F.Surface(its), F.FlowSurface(its)
        self.rev = F.Reverse(its, self.fused)
        self.end, self.prev = EndFlow(its), PrevSurface(its, self.fsurf)

    def build(self, lem, m, pos="VERB", quote=True, fix=True):
        """fix: False = I42 のまま。True か「細かい表を先」= 流れは終わりの字の表を先に見る。「掛け合わせ」= 流れも、語の類の表の割合と掛けて選ぶ。
        ("混ぜる", κ, 語幹の形も混ぜるか) = 流れを mix で選ぶ（その語の表も混ぜる）。3 つ目が True なら、語幹の形も stem_mixed で戻す。"""
        m = tuple(sorted(m))
        if isinstance(fix, tuple):
            _, kappa, stem_too = fix
            fl = self.end.mixed(self.rev, lem, m, pos, quote, kappa)
            if fl is None:
                return None
            stem = stem_mixed(self.surf, lem, fl, pos, kappa) if stem_too else self.surf.form(lem, fl, pos)
            return stem + self.prev.form(fl, stem)
        fl = self.end.flow(self.rev, lem, m, pos, quote, prior=fix == "掛け合わせ") if fix else self.rev.table(lem, m, pos, quote)
        if fl is None:
            return None
        stem = self.surf.form(lem, fl, pos)
        return stem + (self.prev.form(fl, stem) if fix else self.fsurf.form(fl, ("台詞:はい",) if quote else None))
