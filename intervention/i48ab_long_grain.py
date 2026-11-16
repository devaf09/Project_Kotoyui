"""I48ab: 長いグレイン（文 → 要点 → 文）。語の名前ではなく、語板の属性を周波数に置いた波で数える（足すだけ。勾配の学びはしない）。

1. 入口の波（順方向）
   文の語ごとに、語板の属性の割合を周波数の帯に置く（内容語は語板 v2 の帯、機能語は機能語の板。属性の名前 → 周波数は
   f = (属性の番号 × STEP) mod D。属性の数が D を越えると重なる）。語の位置 k は周波数のシフト（f + k × SHIFT mod D）で重ね、
   1 本の長い波（complex64、長さ D）にする。語の名前そのものは入れない。
   要点の読み出し: 位置 k のシフトを戻した波と、候補の語の属性の波との近さ（cos）でいちばん近い語を選ぶ。読めた語の波をほかの位置から
   引いて読み直す（浮かび上がらせ）も試す。
   要点 = 内容語の原形の並び ＋ 意味の印（過去・否定・丁寧・進行・問い）。
2. 逆回し（要点 → 文）
   隣り合う内容語 i・i+1 の間の詰め物（助詞・語尾など、内容語でない語の表記の並び）を、2 語の属性の波の積
   （周波数の和 (f_a + f_b·MIX) mod D の帯。I48w の和の周波数と同じ考え）と、それぞれの語の波（別のずれ）の重ねから、
   多項の数え上げ（足すだけ、加算の平滑 α）で選ぶ。文末（最後の内容語の後ろ）は、最後の語の波と、意味の印の波（印ごとの周波数）で選ぶ。
   内容語の表記（活用）は、本の数え（原形・次の詰め物の頭 → 表記）から。無い原形は、原形の最後の字と次の詰め物の頭からの書き換えの数え。
3. 基準: 語そのものを鍵にした表（(原形 i, 原形 i+1) → 詰め物。無ければ (原形 i) → 、無ければいちばん多い詰め物）。

手で置く物: CONTENT（内容語の品詞）、FUNC_DEP（内容語の品詞でも詰め物にする係り）、MARKS（機能語の板の機能 → 意味の印）、
帯ごとに上位 TOPK の属性を使うこと、STEP・SHIFT・MIX（周波数の置き方の歩幅。D と互いに素の奇数）、α。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

import numpy as np

CONTENT = ("NOUN", "PROPN", "PRON", "NUM", "VERB", "ADJ", "ADV", "DET")
FUNC_DEP = ("fixed", "aux")
MARKS = {"テンス": "過去", "否定": "否定", "丁寧": "丁寧", "アスペクト": "進行"}   # 機能語の板の「機能」 → 意味の印（か・？ は問い）
TOPK = 4
ALPHA = 0.1


def read_conllu(path, limit=None):
    out, cur, meta = [], [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur:
                    out.append(dict(id=meta.get("sent_id", str(len(out))), text=meta.get("text", ""), toks=cur))
                    if limit and len(out) >= limit:
                        return out
                cur, meta = [], {}
                continue
            if line.startswith("#"):
                m = re.match(r"# (\w+) = (.*)", line)
                if m:
                    meta[m.group(1)] = m.group(2)
                continue
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:
                continue
            cur.append(dict(form=c[1], lemma=c[2], upos=c[3], xpos=c[4], dep=c[7]))
    if cur:
        out.append(dict(id=meta.get("sent_id", str(len(out))), text=meta.get("text", ""), toks=cur))
    return out


def is_content(t):
    return t["upos"] in CONTENT and t["dep"] not in FUNC_DEP


def split(sent):
    """文 → (内容語の並び [(原形, 表記, 品詞)], 詰め物の並び（内容語 i の後ろ、最後は文末まで）, 頭の詰め物)。"""
    T = sent["toks"]
    ci = [i for i, t in enumerate(T) if is_content(t)]
    if not ci:
        return [], [], ""
    head = "".join(t["form"] for t in T[:ci[0]])
    cont = [(T[i]["lemma"], T[i]["form"], T[i]["upos"]) for i in ci]
    gaps = []
    for k, i in enumerate(ci):
        j = ci[k + 1] if k + 1 < len(ci) else len(T)
        gaps.append("".join(t["form"] for t in T[i + 1:j]))
    return cont, gaps, head


class Func:
    def __init__(self, func_boards):
        self.by = defaultdict(list)
        for w in func_boards:
            self.by[w["word"]].append(w)

    def board(self, t):
        c = self.by.get(t["lemma"]) or self.by.get(t["form"]) or []
        for w in c:
            if t["xpos"] in w["attrs"].get("品詞", {}):
                return w
        return c[0] if c else None

    def marks(self, toks):
        m = set()
        for t in toks:
            if is_content(t):
                continue
            if t["form"] in ("？", "?") or (t["lemma"] == "か" and t["xpos"].endswith("終助詞")):
                m.add("問い")
            w = self.board(t)
            for f in (w["attrs"].get("機能", {}) if w else {}):
                if f in MARKS:
                    m.add(MARKS[f])
        return m


class AttrSpace:
    """属性 → 周波数。語 → (周波数の配列, 重みの配列)。"""

    def __init__(self, content_boards, func, D, bands=("B", "C", "D", "E", "G"), topk=TOPK):
        self.D, self.func, self.bands, self.topk = D, func, bands, topk
        self.step = self._odd_coprime(int(D * 0.6180339887))
        self.shift = self._odd_coprime(int(D * 0.4142135623))
        self.mix = self._odd_coprime(int(D * 0.7320508075))
        self.cb = {}
        for w in content_boards:
            if w["name"] not in self.cb or w.get("freq", 0) > self.cb[w["name"]].get("freq", 0):
                self.cb[w["name"]] = w
        self.idx = {}
        self._cache = {}

    def _odd_coprime(self, x):
        x |= 1
        while math.gcd(x, self.D) != 1:
            x += 2
        return x

    def f_of(self, name):
        if name not in self.idx:
            self.idx[name] = len(self.idx)
        return (self.idx[name] * self.step) % self.D

    def attrs(self, lemma, upos, xpos="", form=""):
        key = (lemma, upos, xpos)
        if key in self._cache:
            return self._cache[key]
        out = [("品詞:" + upos, 1.0)]
        if upos in CONTENT:
            w = self.cb.get(lemma)
            if w is not None:
                for b in self.bands:
                    A = w["attrs"].get(b, {})
                    for a, v in sorted(A.items(), key=lambda z: -z[1])[: self.topk]:
                        out.append((b + "|" + a, float(v)))
        else:
            w = self.func.board(dict(lemma=lemma, form=form or lemma, xpos=xpos))
            if w is not None:
                for b, A in w["attrs"].items():
                    for a, v in A.items():
                        out.append(("機能語|" + b + ":" + a, float(v)))
            else:
                out.append(("記号:" + (form or lemma), 1.0))
        f = np.array([self.f_of(a) for a, _ in out], dtype=np.int64)
        v = np.array([x for _, x in out], dtype=np.float32)
        v /= max(1e-9, float(np.linalg.norm(v)))
        self._cache[key] = (f, v)
        return f, v

    def tok(self, t):
        return self.attrs(t["lemma"], t["upos"], t["xpos"], t["form"])

    # ---- 1. 入口の波と、位置ごとの読み出し ----
    def encode(self, toks):
        """語 k の属性の周波数を k × SHIFT だけずらして重ねる（周波数のシフト。I48j・I48o のスライドと同じ考え）。"""
        X = np.zeros(self.D, dtype=np.complex64)
        for k, t in enumerate(toks):
            f, v = self.tok(t)
            np.add.at(X, (f + k * self.shift) % self.D, v)
        return X

    def read(self, X, k):
        return np.real(np.roll(X, -k * self.shift)).astype(np.float32)

    def raw(self, f, v):
        y = np.zeros(self.D, dtype=np.float32)
        np.add.at(y, f, v)
        return y

    def decode(self, X, n, M, lex_raw, rounds=2):
        """位置 0..n-1 の語を、候補の語の属性の波 M（行は長さ 1）との近さで読む。rounds 回、ほかの位置で読んだ語の波を引いて読み直す
        （浮かび上がらせ: 重なった波から、読めた語を差し引く）。返り値: 読みの回ごとの候補の番号の並び。"""
        R0 = np.stack([self.read(X, k) for k in range(n)])
        g = np.argmax(R0 @ M.T, axis=1)
        out = [g.copy()]
        for _ in range(rounds):
            Xr = np.zeros(self.D, dtype=np.float32)
            for k in range(n):
                Xr += np.roll(lex_raw[g[k]], k * self.shift)
            Rr = np.stack([np.roll(Xr, -k * self.shift) for k in range(n)])
            R = R0 - Rr + lex_raw[g]
            g = np.argmax(R @ M.T, axis=1)
            out.append(g.copy())
        return out

    def dense(self, f, v):
        y = np.zeros(self.D, dtype=np.float32)
        np.add.at(y, f, v)
        n = float(np.linalg.norm(y))
        return y / n if n > 0 else y

    # ---- 2. 逆回しの特徴（2 語の波の積の帯 ＋ それぞれの波 ＋ 印の帯） ----
    def pair(self, a, b, marks=()):
        fa, va = a
        if b is None:                                       # 文末: 最後の語の波 ＋ 意味の印の波
            fs = [fa]
            vs = [va]
            for m in sorted(marks):
                fs.append(np.array([(self.f_of("印:" + m) + self.shift) % self.D]))
                vs.append(np.array([1.0], dtype=np.float32))
            fs.append(np.array([(self.f_of("印:文末") + self.shift) % self.D]))
            vs.append(np.array([1.0], dtype=np.float32))
            return np.concatenate(fs), np.concatenate(vs)
        fb, vb = b
        fx = ((fa[:, None] + fb[None, :] * self.mix) % self.D).ravel()
        vx = (va[:, None] * vb[None, :]).ravel()
        return (np.concatenate([fx, fa, (fb + self.shift) % self.D]),
                np.concatenate([vx, va * 0.5, vb * 0.5]).astype(np.float32))


class PairGrain:
    """詰め物の選び方の数え上げ（多項、足すだけ）。クラスは学びの上位 n_cls の詰め物と「その他」。"""

    def __init__(self, D, n_cls=300):
        self.D, self.n_cls = D, n_cls
        self.cls, self.W, self.prior = None, None, None

    def fit(self, rows, counts):
        """rows: (f, v, 詰め物) を順に出す物（一度だけ読む）。counts: 詰め物の数（クラスを決める）。"""
        self.cls = [y for y, _ in counts.most_common(self.n_cls)]
        ci = {y: i for i, y in enumerate(self.cls)}
        other = len(self.cls)
        self.cls.append(None)
        self.W = np.zeros((len(self.cls), self.D), dtype=np.float64)
        pr = np.zeros(len(self.cls))
        for f, v, y in rows:
            i = ci.get(y, other)
            np.add.at(self.W[i], f, v)
            pr[i] += 1
        self.prior = np.log((pr + 1) / (pr.sum() + len(pr)))
        tot = self.W.sum(1, keepdims=True)
        self.logp = np.log((self.W + ALPHA) / (tot + ALPHA * self.D))
        self.logp[other] = -1e9                             # 「その他」は言わない
        return self

    def predict(self, f, v, k=1, prior_w=1.0):
        s = prior_w * self.prior + self.logp[:, f] @ v
        o = np.argsort(-s)[:k]
        return [self.cls[i] for i in o]


class WordTable:
    def __init__(self):
        self.t2, self.t1, self.t0 = defaultdict(Counter), defaultdict(Counter), Counter()

    def fit(self, rows):
        """rows: [(原形 i, 原形 i+1 または印の組, 詰め物)]"""
        for a, b, y in rows:
            self.t2[(a, b)][y] += 1
            self.t1[a][y] += 1
            self.t0[y] += 1
        return self

    def predict(self, a, b):
        for c in (self.t2.get((a, b)), self.t1.get(a)):
            if c:
                return c.most_common(1)[0][0]
        return self.t0.most_common(1)[0][0]


class Surface:
    """内容語の表記（活用）: (原形, 次の詰め物の頭の字) → 表記。無い原形は (原形の最後の字, 品詞, 頭の字) → (削る字数, 足す字)。"""

    def __init__(self):
        self.t, self.r = defaultdict(Counter), defaultdict(Counter)

    @staticmethod
    def _head(g):
        return g[:1]

    def add(self, lemma, form, upos, nxt):
        h = self._head(nxt)
        self.t[(lemma, h)][form] += 1
        n = 0
        while n < min(len(lemma), len(form)) and lemma[n] == form[n]:
            n += 1
        self.r[(lemma[-1:], upos, h)][(len(lemma) - n, form[n:])] += 1

    def form(self, lemma, upos, nxt):
        c = self.t.get((lemma, self._head(nxt)))
        if c:
            return c.most_common(1)[0][0]
        c = self.r.get((lemma[-1:], upos, self._head(nxt)))
        if c:
            cut, add = c.most_common(1)[0][0]
            return lemma[: len(lemma) - cut] + add
        return lemma


def norm(s):
    return re.sub(r"[\s　]", "", s)
