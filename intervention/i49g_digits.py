"""I49g: 数え歌を文字で見て、並びの決まりを数える（桁を教えずに見つける案 A、未確定）。

数を文字の並びとして見る（"47" は "4" と "7"）。意味は教えない。数え歌の隣どうし（x の次が y）を、右からそろえて 1 文字ずつ比べ、
「その場所の字がどうなったか」を数える。
  どうなったか: 同じ ／ 次（1 文字だけの数の並び 0→1→…→9 での次）／ そのどちらでもなければ、なった字そのもの。左に字が無い所は「無」。
  手がかり: (自分の字, 右隣の字) → 右隣の字だけ → 自分の字だけ。数のある最初の手がかりで決める（細かい方から）。右端の字の右隣は「端」。
聞いた事の無い数の「次」は、場所ごとにこの決まりを当てはめて作る。決まりが無い・次の字を知らない時は作れない（None）。
手で置く物: 右からそろえる事、手がかりの順、いちばん数の多い答えを使う事（同じ数なら作れないとする）。
記録: 作るたびに、場所ごとにどの手がかりで決めたか。
"""
from __future__ import annotations

from collections import Counter

NONE, EDGE = "無", "端"
KEYS = ("自分と右隣", "右隣", "自分")


class CharSong:
    def __init__(self):
        self.step = {}                                    # 1 文字の数の並び: 字 → 次の字
        self.rule = [dict() for _ in KEYS]                # 手がかりの種類 → (場所, 手がかり) → Counter(どうなったか)
        self.rec = []

    @staticmethod
    def _chars(w, n):
        return [NONE] * (n - len(w)) + list(w)

    def _keys(self, p, own, right):
        return ((p, own, right), (p, right), (p, own))

    def hear(self, seq):
        seq = list(seq)
        for x, y in zip(seq, seq[1:]):
            if len(x) == 1 and len(y) == 1:
                self.step[x] = y
        for x, y in zip(seq, seq[1:]):
            n = max(len(x), len(y)) + 1                   # 左に 1 つ「無」を足して見る（字が増える場合のため。作る時と同じ）
            cx, cy = self._chars(x, n), self._chars(y, n)
            for i in range(n):
                p = n - 1 - i                             # 右から数えた場所
                right = cx[i + 1] if i + 1 < n else EDGE
                what = "同じ" if cy[i] == cx[i] else "次" if self.step.get(cx[i]) == cy[i] else cy[i]
                for k, key in enumerate(self._keys(p, cx[i], right)):
                    self.rule[k].setdefault(key, Counter())[what] += 1

    def next(self, w):
        """w の次の数を作る。作れなければ None。"""
        n = len(w) + 1                                    # 左に 1 つ「無」を足して見る（字が増える場合のため）
        cx, out, used = self._chars(w, n), [], []
        for i in range(n):
            p = n - 1 - i
            right = cx[i + 1] if i + 1 < n else EDGE
            what = None
            for k, key in enumerate(self._keys(p, cx[i], right)):
                c = self.rule[k].get(key)
                if c:
                    top = c.most_common(2)
                    if len(top) == 1 or top[0][1] > top[1][1]:
                        what = top[0][0]
                    used.append(KEYS[k])
                    break
            if what is None:
                self.rec.append((w, None, used))
                return None
            ch = cx[i] if what == "同じ" else self.step.get(cx[i]) if what == "次" else what
            if ch is None:
                self.rec.append((w, None, used))
                return None
            out.append(ch)
        y = "".join(c for c in out if c != NONE)
        self.rec.append((w, y, used))
        return y
