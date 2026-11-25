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
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

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


class CharSum:
    """足し算の事実を文字で見て、どの手がかりで答えの字が決まるかを数える（桁を教えずに見つける案 B、未確定）。

    事実 (A, B, C) を右からそろえて 1 文字ずつ見る。場所ごとに、自分の字の組 (A の字, B の字) と、右隣の字の組を手がかりにする。
    数える物（聞いた事実だけから。数える順は 3 段）:
      1. 右端の表: 右端の字の組 → 答えの右端の字。
      2. 右隣の組の働き: 右端でない場所で、自分の字の組が右端の表に有る時、答えの字が「表のとおり（同じ）」か「表の字の次」か「ほか」かを、右隣の字の組ごとに数える。
         （「次」になる右隣の組が、繰り上がりの起きる組。名前は付けない。）
      3. 残りの組: (自分の字の組, 右隣の組の働き) → 答えの字。左に字が無い所（無）を含む組は、ここで覚える。
      2 と 3 は、分かった所から順に広げる: 3 で「右隣が何もしない時の字」が分かった組を使って、まだ働きの分からない右隣の組を 2 で数え直す。増えなくなるまで。
    答え方: 右から 1 文字ずつ。右端は 1 の表。ほかは、3 に有ればその字。無ければ、1 の表の字を、2 の働き（同じ・次）でずらす。どれも無ければ作れない（None）。
    字の「次」は、数え歌の 1 文字の並び（CharSong.step）を借りる。
    手で置く物: 右からそろえる事、手がかりを「自分の字の組」と「右隣の字の組」にする事、数える順、いちばん数の多い答えを使う事。
    """

    def __init__(self, step):
        self.step = step                                  # 字 → 次の字
        self.facts = []
        self.rec = []
        self._t = None

    def learn(self, tri):
        self.facts.append(tuple(tri))
        self._t = None

    @staticmethod
    def _rows(a, b, c=None):
        n = max(len(a), len(b), len(c) if c else 0) + 1
        pad = lambda w: [NONE] * (n - len(w)) + list(w)
        A, B, C = pad(a), pad(b), pad(c) if c is not None else [None] * n
        rows, right = [], EDGE
        for i in range(n - 1, -1, -1):                    # 右から
            own = (A[i], B[i])
            rows.append((own, right, C[i]))
            right = own
        return rows

    @staticmethod
    def _top(c):
        return max(c, key=c.get) if c else None

    def tables(self):
        if self._t is not None:
            return self._t
        rows = [self._rows(*f) for f in self.facts]
        t0, work, rest = {}, {}, {}
        for r in rows:
            own, _, ch = r[0]
            t0.setdefault(own, Counter())[ch] += 1

        def base(own):                                    # その組の「右隣が何もしない時」の字
            b = self._top(t0.get(own))
            return b if b is not None else self._top(rest.get((own, "同じ")))

        while True:                                       # 分かった所から順に広げる（増えなくなるまで）
            new = {}
            for r in rows:
                for own, right, ch in r[1:]:
                    if right in work:
                        continue
                    b = base(own)
                    if b is not None:
                        what = "同じ" if ch == b else "次" if self.step.get(b) == ch else "ほか"
                        new.setdefault(right, Counter())[what] += 1
            if not new:
                break
            work.update(new)
            rest = {}
            for r in rows:
                for own, right, ch in r[1:]:
                    w = self._top(work.get(right))
                    if w is not None:
                        rest.setdefault((own, w), Counter())[ch] += 1
        self._t = (t0, work, rest)
        return self._t

    def ask(self, a, b):
        """A＋B の答えを 1 文字ずつ作る。作れなければ None。"""
        t0, work, rest = self.tables()
        out, how = [], []
        for own, right, _ in self._rows(a, b):
            if right == EDGE:
                ch = self._top(t0.get(own))
                how.append("右端の表")
            else:
                w = self._top(work.get(right))
                ch = self._top(rest.get((own, w))) if w is not None else None
                how.append("残りの組")
                if ch is None and w in ("同じ", "次"):
                    base = self._top(t0.get(own))
                    ch = base if w == "同じ" else self.step.get(base)
                    how[-1] = "表をずらす"
            if ch is None:
                self.rec.append((a, b, None, how))
                return None
            out.append(ch)
        y = "".join(c for c in reversed(out) if c != NONE)
        self.rec.append((a, b, y, how))
        return y
