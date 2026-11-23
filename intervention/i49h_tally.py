"""I49h: 関係の表の「下がる順」を、手で置かずに、段ごとの当たり外れの数で決める（算数の試しの知見 6 の応用、未確定）。

もとの関係の表（I49e の Relation）は、2 語 → 片方だけ借りる → 述語側 → もう片方 → 借りる の順（手で置いた順）に、数のある最初の段で答える。
ここでは、問われるたびに 5 つの段それぞれの答えを控えておき、後でその 3 つ組を聞いた時（learn）に、段ごと・席ごとに当たり・外れを数える。
答え方は 2 通り:
  選ぶ      数のある段のうち、これまでの当たりの割合がいちばん高い段の答え。同じ割合なら、もとの順で先の段。まだ 1 度も答え合わせしていない段は、もとの順の扱い（割合 0 と同じに置かない）。
  合わせる   数のある段ぜんぶの点を、段ごとに合計 1 にそろえ、当たりの割合を掛けて足す。いちばん大きい候補。
手で置く物: 当たりの割合 = 当たり ÷ (当たり + 外れ) をそのまま使う事、まだ答え合わせの無い段をもとの順で扱う事。切る線・上位何個は置かない。
記録: もとの rec に加えて、段ごと・席ごとの当たり・外れ（tally_levels）。
"""
from __future__ import annotations

from i49e_relation import LEVELS, NEAR, Relation, SLOTS

MODES = ("選ぶ", "合わせる")


class TallyRelation(Relation):
    def __init__(self, vec_of, half=1024.0, mode="選ぶ"):
        super().__init__(vec_of, half)
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.hit = [{lv: 0 for lv in LEVELS} for _ in SLOTS]
        self.miss = [{lv: 0 for lv in LEVELS} for _ in SLOTS]
        self._wait = {}                                   # (隠す席, 見せた 2 語) → 段 → その段の答え

    def rate(self, si, lv):
        n = self.hit[si][lv] + self.miss[si][lv]
        return None if n == 0 else self.hit[si][lv] / n

    def ask(self, si, tri, cands):
        cands = list(cands)
        near = NEAR[si]
        other = next(j for j in range(3) if j not in (si, near))
        per = {}
        for lv in LEVELS:
            s = self._level(lv, si, tri, cands)
            if any(v > 0 for v in s.values()):
                per[lv] = (s, max(cands, key=lambda c: (s[c], -cands.index(c))))
        self._wait[(si, tri[near], tri[other])] = {lv: b for lv, (_, b) in per.items()}
        out = dict(段="答えられない", 点={}, 答え=None)
        if per:
            rated = [lv for lv in per if self.rate(si, lv) is not None]
            if self.mode == "選ぶ" or not rated:
                first = next(lv for lv in LEVELS if lv in per)       # もとの順で先の段
                pick = first
                if rated and self.rate(si, first) is not None:
                    pick = max(rated, key=lambda lv: (self.rate(si, lv), -LEVELS.index(lv)))
                out = dict(段=pick, 点=per[pick][0], 答え=per[pick][1])
            else:
                tot = {c: 0.0 for c in cands}
                for lv in rated:
                    s, z = per[lv][0], sum(per[lv][0].values())
                    for c in cands:
                        tot[c] += self.rate(si, lv) * s[c] / z
                if any(v > 0 for v in tot.values()):
                    out = dict(段="合わせる", 点=tot, 答え=max(cands, key=lambda c: (tot[c], -cands.index(c))))
                else:                                     # 答え合わせ済みの段の割合がぜんぶ 0 の時は、もとの順
                    first = next(lv for lv in LEVELS if lv in per)
                    out = dict(段=first, 点=per[first][0], 答え=per[first][1])
        self.rec.append((self.now, si, (tri[near], tri[other]), out["段"], out["答え"]))
        return out

    def learn(self, tri):
        t = tuple(tri)
        for si in range(3):                               # 問われて待っていた答えを、ここで答え合わせする
            near = NEAR[si]
            other = next(j for j in range(3) if j not in (si, near))
            w = self._wait.pop((si, t[near], t[other]), None)
            if w:
                for lv, ans in w.items():
                    if ans == t[si]:
                        self.hit[si][lv] += 1
                    else:
                        self.miss[si][lv] += 1
        super().learn(t)

    def tally_levels(self):
        return {SLOTS[si]: {lv: dict(当たり=self.hit[si][lv], 外れ=self.miss[si][lv]) for lv in LEVELS} for si in range(3)}
