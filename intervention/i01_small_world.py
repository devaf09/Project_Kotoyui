"""介入の研究 実験I1: 小さな世界で「既存と違う」の自覚から語板を分け、介入の問いに答える

持ち主の案（2026-11-04、Sin-CNN3 の学び 51・62 が元）:
  1. 語板 = 1 語（1 つの意味）= 1 台の光の器。出口の明るさ = 属性（色・形・味…）の割合。対応のない属性は光が届かない（拡散）。
  2. 文の中の語の属性が、既にある語板と「一定以上違う」と自覚したら、同じ文字列でも新しい語板を作る（リンゴ = 果物／人、緑で渋いトマト）。
  3. 介入 = 「赤いトマトを緑に塗ったら渋くなる？」のような問い。自覚で不自然な関係（色 → 味）を否定し、残りの属性を比べて
     「未熟 → 渋い」を見つける。
最初の段階として手で与えるもの（持ち主の判断 2026-11-04）:
  - 語と属性の対応表（「赤い」= 色:赤 など）。動作の語は種類を含む（「手を振った」= する:手を振る + 種類:人）。
  - 動詞が変える観点（塗る → 色だけ、熟させる → 状態だけ）。
  - 分けるかどうかのしきい値（固定）。原因の比べはデジタル（光にするのは 2 回目から）。
方法:
  - 手で作った 9 つの意味（下の SENSES）から、文を型で作る。教え 50 文 / 意味、試し 20 文 / 意味。順番は seed で混ぜる。
  - 文を 1 つずつ読む。文の中の名詞の属性を、同じ語の語板の明るさと比べる。
      観点が分かっている（光が届いている）属性のうち、暗い（その観点の最大の明るさの 20% 未満）ものの割合 = 違い。
      いちばん違いの小さい語板に入れる。違いがしきい値以上なら、新しい語板を作る。分かっていない観点の属性は違いに数えない（新しい知識として足す）。
      ただし、文の属性が語板の分かっている観点とまったく重ならず、語板が観点を 2 つ以上知っているなら、違い = 0.5（「人の語板に味の話」）。
  - 光の器: 実験09 の器（sincnn3.LightDevice と同じ式）を小さく（縦 32 画素・板 8 枚）。正解 = その語板に入った属性の割合。
    文を読むたびに 30 回、新しい語板には 150 回学ぶ。--mode digital は光を使わず、数えた割合そのもの。
  - 物差し:
      (1) 分かれ方: 語ごとの語板の数、教えの文の純度（語板の多数派の意味と同じ意味の割合）。
      (2) 振り分け: 試しの文（意味が属性から決まるもの）を、正しい意味の語板に入れられるか。
      (3) 違和感: 試しの文の属性を 1 つ、その語のどの意味にもない値に変えた「おかしな文」と、元の文を、違いの大きさで見分ける（AUROC）。
      (4) 介入の問い: 「X を Y に塗ったら Z くなる？」「X を熟させたら Z くなる？」「… Z い まま？」（はい／いいえ）。
          原因の効き目は、同じ語の語板どうしの差（観点ごとの明るさの差）と、1 枚の語板の中の差（色が 2 つあるのに味は 1 つ、など）から出す。
            否定: 観点 v が違うのに味が変わらない組が 1 つでもあれば、v は味の原因ではないと否定する。残った観点の効き目 = 1。
            最小二乗: 味の差 ≈ Σ 効き目_v × v の差（Sin-CNN3 実験63 と同じ）。
            相関だけ: 変えた後の姿に近い語板の味をそのまま読む（介入を考えない）。
          聞かれた語の組は使わない（1 語抜き）ものを主の数字にする。
      (5) 見たままの問い: 「X は Z い？」。
  - 比べる相手（同じ文で学んだ）: GRU・小さなトランスフォーマー（次の語当て）。問いと答えを文として足して学ぶ:
      obs = 見たままの問いだけ、int = それに加えて、聞かれる語以外の語の介入の問いと答え。1 語抜きで 4 回学ぶ。
結果: results/i01.jsonl（1 行 = 1 つの設定と seed）
実行: CUDA_VISIBLE_DEVICES= python intervention/i01_small_world.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------- 小さな世界
VIEWS = {
    "種類": ["果物", "野菜", "人"],
    "色": ["赤", "緑", "黄", "白"],
    "形": ["丸い", "長い"],
    "味": ["甘い", "酸っぱい", "渋い"],
    "状態": ["完熟", "未熟"],
    "する": ["手を振る", "話す", "歩く"],
    "される": ["食べる", "切る"],
}
SLOTS = [(v, x) for v, xs in VIEWS.items() for x in xs]
SLOT = {s: i for i, s in enumerate(SLOTS)}

# 語（トークン）→ 属性（手で与える対応表）
COLOR_TOK = {"赤": "赤い", "緑": "緑の", "黄": "黄色い", "白": "白い"}
PAINT_TOK = {"赤": "赤に", "緑": "緑に", "黄": "黄色に", "白": "白に"}
TASTE_TOK = {"甘い": "甘い", "酸っぱい": "酸っぱい", "渋い": "渋い"}
TASTE_NARU = {"甘い": "甘く", "酸っぱい": "酸っぱく", "渋い": "渋く"}
LEX = {}
for c, t in COLOR_TOK.items():
    LEX[t] = [("色", c)]
for t in ("丸い", "長い"):
    LEX[t] = [("形", t)]
for t in TASTE_TOK:
    LEX[t] = [("味", t)]
LEX["甘くて"] = [("味", "甘い")]
LEX["完熟の"] = [("状態", "完熟")]
LEX["未熟な"] = [("状態", "未熟")]
LEX["果物"] = [("種類", "果物")]
LEX["野菜"] = [("種類", "野菜")]
LEX["手を振った"] = [("する", "手を振る"), ("種類", "人")]
LEX["話した"] = [("する", "話す"), ("種類", "人")]
LEX["歩いた"] = [("する", "歩く"), ("種類", "人")]
LEX["食べた"] = [("される", "食べる")]
LEX["切った"] = [("される", "切る")]

SENSES = {   # 意味 → (語, 属性 {観点: [値]})。色が 2 つある意味は、1 文に 1 つだけ出す。ripe トマトの味は 2 つを 1 文に一緒に出す。
    "リンゴ/果物": ("リンゴ", {"種類": ["果物"], "色": ["赤", "緑"], "形": ["丸い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる", "切る"]}),
    "リンゴ/人": ("リンゴ", {"種類": ["人"], "する": ["手を振る", "話す", "歩く"]}),
    "トマト/完熟": ("トマト", {"種類": ["野菜"], "色": ["赤"], "形": ["丸い"], "味": ["甘い", "酸っぱい"], "状態": ["完熟"], "される": ["食べる", "切る"]}),
    "トマト/未熟": ("トマト", {"種類": ["野菜"], "色": ["緑"], "形": ["丸い"], "味": ["渋い"], "状態": ["未熟"], "される": ["切る"]}),
    "イチゴ/完熟": ("イチゴ", {"種類": ["果物"], "色": ["赤"], "形": ["丸い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる"]}),
    "イチゴ/未熟": ("イチゴ", {"種類": ["果物"], "色": ["白"], "形": ["丸い"], "味": ["酸っぱい"], "状態": ["未熟"], "される": ["切る"]}),
    "バナナ/完熟": ("バナナ", {"種類": ["果物"], "色": ["黄"], "形": ["長い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる", "切る"]}),
    "バナナ/未熟": ("バナナ", {"種類": ["果物"], "色": ["緑"], "形": ["長い"], "味": ["渋い"], "状態": ["未熟"], "される": ["切る"]}),
    "レモン": ("レモン", {"種類": ["果物"], "色": ["黄"], "形": ["丸い"], "味": ["酸っぱい"], "状態": ["完熟"], "される": ["切る"]}),
}
WORDS = sorted({w for w, _ in SENSES.values()})
FRUIT_WORDS = ["イチゴ", "トマト", "バナナ", "リンゴ"]       # 介入の問いを作る語（状態の違う 2 つの意味、またはリンゴの色）


def make_sentence(sense, rng):
    """意味から 1 文（トークンの列）を作る。属性は 2〜4 個。人は動作 1 つ（+ 種類:人）。"""
    w, at = SENSES[sense]
    if "する" in at:
        return [w, "が", rng.choice(["手を振った", "話した", "歩いた"]), "。"]
    views = [v for v in ("色", "状態", "形", "種類", "味", "される") if v in at]
    pick = rng.sample(views, rng.randint(2, min(4, len(views))))
    pre = []
    for v in ("状態", "色", "形"):
        if v in pick:
            x = rng.choice(at[v])
            pre.append({"色": COLOR_TOK.get(x), "状態": "完熟の" if x == "完熟" else "未熟な", "形": x}[v])
    if "味" in pick:
        pred = ["は"] + (["甘くて", "酸っぱい"] if len(at["味"]) == 2 else [at["味"][0]])
    elif "される" in pick:
        pred = ["を", "食べた" if rng.choice(at["される"]) == "食べる" else "切った"]
    else:
        pred = ["は", at["種類"][0], "だ"]
    if "種類" in pick and pred[1] != at["種類"][0]:
        pred = pred + ["。", w, "は", at["種類"][0], "だ"]            # 種類は 2 つ目の短い文で言う
    return pre + [w] + pred + ["。"]


def attrs_of(toks, word):
    """文から、語 word の属性 [(観点, 値)] を取り出す（手で与えた対応表。1 文 = 1 つの物の話）。"""
    out = []
    for t in toks:
        out += LEX.get(t, [])
    return sorted(set(out))


def make_world(seed, n_train=50, n_test=20):
    rng = random.Random(seed)
    train, test = [], []
    for s in SENSES:
        for _ in range(n_train):
            train.append((s, make_sentence(s, rng)))
        for _ in range(n_test):
            test.append((s, make_sentence(s, rng)))
    rng.shuffle(train)
    return train, test


def determined(word, attrs):
    """属性から、その語の意味が 1 つに決まるか（決まるならその意味）。"""
    cands = []
    for s, (w, at) in SENSES.items():
        if w != word:
            continue
        if all(x in at.get(v, []) for v, x in attrs):
            cands.append(s)
    return cands[0] if len(cands) == 1 else None


# ---------------------------------------------------------------- 光の器（実験09 の Devices を小さく）
class Devices(torch.nn.Module):
    def __init__(self, W, n=32, plates=8, gap=16.0, seed=0):
        super().__init__()
        self.n, self.plates, self.L = n, plates, 2 * n
        fx = torch.fft.fftfreq(self.L)
        self.register_buffer("H", torch.exp(1j * gap * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * fx) ** 2)).to(torch.complex64))
        g = torch.Generator().manual_seed(seed)
        self.phase = torch.nn.Parameter(math.pi * (torch.rand(W, n, plates, generator=g) < 0.5).float())
        y = torch.arange(n, dtype=torch.float32)
        self.register_buffer("inp", (torch.exp(1j * 2 * math.pi * y / n) / math.sqrt(n)).to(torch.complex64))

    def brightness(self, idx):
        ph = self.phase[idx]
        u = torch.nn.functional.pad(self.inp.expand(len(idx), -1), (0, self.L - self.n))
        for k in range(self.plates):
            u = torch.cat([u[:, :self.n] * torch.exp(1j * ph[:, :, k]), u[:, self.n:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * self.H, dim=-1, norm="ortho")
        return torch.fft.fft(u[:, :self.n], dim=-1, norm="ortho").abs() ** 2


VAL0 = 3


class Boards:
    """語板の集まり。counts = 入った属性の数（学ぶ正解のもと）、light = 光の器の出口（mode=light のとき読むのはこちら）。"""

    def __init__(self, mode, theta, seed, cap=40, steps=30, new_steps=150, dark=0.2):
        self.mode, self.theta, self.dark = mode, theta, dark
        self.word, self.counts, self.members, self.occ = [], [], [], []     # occ = 入った文の属性と意味（分け直しに使う）
        self.steps, self.new_steps = steps, new_steps
        if mode == "light":
            self.D = Devices(cap, seed=seed)
            self.opt = torch.optim.Adam(self.D.parameters(), lr=0.02)
        self._cache = None

    def target(self):
        q = torch.zeros(len(self.counts), self.D.n - VAL0 + 1)
        for r, c in enumerate(self.counts):
            tot = c.sum()
            q[r, :len(SLOTS)] = torch.tensor(c / tot, dtype=torch.float32)
        return q

    def fit(self, steps):
        if self.mode != "light" or not self.counts:
            return
        idx = torch.arange(len(self.counts))
        q = self.target()
        for _ in range(steps):
            Wb = self.D.brightness(idx)[:, VAL0:]
            p = torch.cat([Wb, (1 - Wb.sum(1)).clamp(min=1e-9)[:, None]], 1)
            loss = -(q * torch.log(p + 1e-9)).sum(1).sum()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self._cache = None

    def light(self):
        """語板ごとの属性の明るさ (B, len(SLOTS))。"""
        if self.mode == "digital":
            return np.array([c / c.sum() for c in self.counts])
        if self._cache is None:
            with torch.no_grad():
                self._cache = self.D.brightness(torch.arange(len(self.counts)))[:, VAL0:VAL0 + len(SLOTS)].numpy()
        return self._cache

    def tv(self):
        if self.mode == "digital":
            return 0.0
        L = self.light()
        return float(np.mean([0.5 * np.abs(L[r] - c / c.sum()).sum() for r, c in enumerate(self.counts)]))

    def view_dist(self, r, v, L=None):
        L = self.light() if L is None else L
        ids = [SLOT[(v, x)] for x in VIEWS[v]]
        b = L[r, ids]
        return b, b.sum()

    def known(self, r, v, L):
        b, s = self.view_dist(r, v, L)
        return s >= 0.03 * L[r].sum()

    def diff(self, r, attrs, L=None):
        """自覚の違い: 分かっている観点の属性のうち、暗いものの割合（と、合った数）。"""
        L = self.light() if L is None else L
        known, darkn = 0, 0
        for v, x in attrs:
            if not self.known(r, v, L):
                continue
            b, _ = self.view_dist(r, v, L)
            known += 1
            if b[VIEWS[v].index(x)] < self.dark * b.max():
                darkn += 1
        if known == 0:                                             # 重なる観点がまったく無い: 育った語板（観点 2 つ以上）なら「違うかもしれない」
            nk = sum(self.known(r, v, L) for v in VIEWS)
            return (0.5 if nk >= 2 else 0.0), 0
        return darkn / known, known - darkn

    def route(self, word, attrs):
        """いちばん違いの小さい同じ語の語板（違い、番号）。無ければ (1.0, None)。"""
        L = self.light() if self.counts else None
        best = (1.0, 0, 0, None)
        for r, w in enumerate(self.word):
            if w != word:
                continue
            d, m = self.diff(r, attrs, L)
            key = (d, -m, -self.counts[r].sum(), r)
            if best[3] is None or key < (best[0], -best[1], -best[2], best[3]):
                best = (d, m, self.counts[r].sum(), r)
        return best[0], best[3]

    def read(self, word, attrs, sense):
        d, r = self.route(word, attrs)
        if r is None or d >= self.theta:
            r = len(self.counts)
            self.word.append(word)
            self.counts.append(np.zeros(len(SLOTS)))
            self.members.append(Counter())
            self.occ.append([])
            if self.mode == "light":
                g = torch.Generator().manual_seed(1000 + r)
                self.D.phase.data[r] = math.pi * (torch.rand(self.D.n, self.D.plates, generator=g) < 0.5).float()
            new = True
        else:
            new = False
        for v, x in attrs:
            self.counts[r][SLOT[(v, x)]] += 1
        self.members[r][sense] += 1
        self.occ[r].append((attrs, sense))
        self.fit(self.new_steps if new else self.steps)
        return r, new

    def n_differ(self, r1, r2, L):
        """2 枚の語板で、両方に光が届いていて、値の割合が半分以上違う観点の数（と、両方に届いている観点の数）。"""
        dif = both = 0
        for v in VIEWS:
            if self.known(r1, v, L) and self.known(r2, v, L):
                both += 1
                b1, s1 = self.view_dist(r1, v, L)
                b2, s2 = self.view_dist(r2, v, L)
                dif += 0.5 * np.abs(b1 / s1 - b2 / s2).sum() >= 0.5
        return dif, both

    def consolidate(self, max_differ=1):
        """見直し: 同じ語で、違う観点が 1 つ以下の語板は、同じ物の幅（色違いのリンゴ、動作の違う人）として 1 枚にまとめる。"""
        merged = 0
        while True:
            L = self.light()
            best = None
            for r1 in range(len(self.counts)):
                for r2 in range(r1 + 1, len(self.counts)):
                    if self.word[r1] != self.word[r2]:
                        continue
                    dif, both = self.n_differ(r1, r2, L)
                    if both >= 2 and dif <= max_differ and (best is None or dif < best[0]):
                        best = (dif, r1, r2)
            if best is None:
                break
            _, r1, r2 = best
            if self.counts[r2].sum() > self.counts[r1].sum() and self.mode == "light":
                self.D.phase.data[r1] = self.D.phase.data[r2]          # 大きいほうの器の板を残す
            self.counts[r1] = self.counts[r1] + self.counts[r2]
            self.members[r1] = self.members[r1] + self.members[r2]
            self.occ[r1] = self.occ[r1] + self.occ[r2]
            for lst in (self.word, self.counts, self.members, self.occ):
                del lst[r2]
            if self.mode == "light":
                n = len(self.counts)
                self.D.phase.data[r2:n] = self.D.phase.data[r2 + 1:n + 1].clone()
                self.opt = torch.optim.Adam(self.D.parameters(), lr=0.02)
                self.fit(self.new_steps)
            self._cache = None
            merged += 1
        return merged

    def split_check(self, views=("色", "状態", "味", "形", "種類")):
        """2 つの関係を持つ語板を分ける: 2 つ以上の観点にそれぞれ 2 つの値があり、片方の観点の値で文を分けると
        ほかの観点も一緒に分かれる（違う観点が 2 つ以上）なら、2 枚にする（持ち主の「2 つの光の経路にまとまる」の手作りの代わり）。"""
        split = 0
        r = 0
        while r < len(self.counts):
            L = self.light()
            multi = []
            for v in views:
                if not self.known(r, v, L):
                    continue
                b, sm = self.view_dist(r, v, L)
                srt = np.sort(b / sm)
                if srt[-2] >= 0.2:
                    multi.append((srt[-2], v))
            if len(multi) < 2:
                r += 1
                continue
            _, v = max(multi)
            b, _ = self.view_dist(r, v, L)
            x1, x2 = [VIEWS[v][i] for i in np.argsort(b)[-2:]]
            g1 = [o for o in self.occ[r] if (v, x1) in o[0]]
            g2 = [o for o in self.occ[r] if (v, x2) in o[0]]
            rest = [o for o in self.occ[r] if o not in g1 and o not in g2]
            c1, c2 = (sum((np.eye(len(SLOTS))[[SLOT[a] for a in o[0]]].sum(0) for o in g) ) for g in (g1, g2))
            dif = 0
            for u in VIEWS:
                ids = [SLOT[(u, x)] for x in VIEWS[u]]
                s1, s2 = c1[ids].sum(), c2[ids].sum()
                if s1 > 0 and s2 > 0:
                    dif += 0.5 * np.abs(c1[ids] / s1 - c2[ids] / s2).sum() >= 0.5
                elif max(s1 / max(len(g1), 1), s2 / max(len(g2), 1)) >= 0.5:   # 片方にだけいつもある観点（人の動作など）
                    dif += 1
            if dif < 2:
                r += 1
                continue
            for o in rest:                                          # どちらとも言えない文は、違いの小さいほうへ
                d1 = sum(c1[SLOT[a]] == 0 for a in o[0] if c1[[SLOT[(a[0], x)] for x in VIEWS[a[0]]]].sum() > 0)
                d2 = sum(c2[SLOT[a]] == 0 for a in o[0] if c2[[SLOT[(a[0], x)] for x in VIEWS[a[0]]]].sum() > 0)
                (g1 if (d1, -len(g1)) <= (d2, -len(g2)) else g2).append(o)
            n = len(self.counts)
            self.word.append(self.word[r])
            for gi, (idx, g) in enumerate(((r, g1), (n, g2))):
                cnt = np.zeros(len(SLOTS))
                for at, _ in g:
                    for a in at:
                        cnt[SLOT[a]] += 1
                mem = Counter(sn for _, sn in g)
                if idx == n:
                    self.counts.append(cnt); self.members.append(mem); self.occ.append(list(g))
                else:
                    self.counts[idx], self.members[idx], self.occ[idx] = cnt, mem, list(g)
            if self.mode == "light":
                gg = torch.Generator().manual_seed(2000 + n)
                self.D.phase.data[n] = math.pi * (torch.rand(self.D.n, self.D.plates, generator=gg) < 0.5).float()
                self.opt = torch.optim.Adam(self.D.parameters(), lr=0.02)
                self.fit(self.new_steps)
            self._cache = None
            split += 1
        return split


# ---------------------------------------------------------------- 介入
CAUSE_VIEWS = ["色", "状態", "形"]


def view_norm(B, r, v, L):
    b, s = B.view_dist(r, v, L)
    return b / s if s > 0 else None


LAST_B = None
VERB_RULE = True                                                    # T1: 動詞が変える観点の規則（塗る = 色）を外すときは False
DIST = None                                                         # I14: 観点ごとの差の測り方を差し替える（None = 明るさを観点ごとに正規化した差、デジタル）


def pairs(B, exclude_word=None):
    """原因を比べる組: (語, Δ{観点}, Δ味)。同じ語・人でない語板どうし + 1 枚の中（観点に 2 つの値、味は 1 つ）。"""
    L = B.light()
    out = []
    food = [r for r, w in enumerate(B.word) if w != exclude_word and B.known(r, "味", L)
            and all(B.known(r, v, L) for v in CAUSE_VIEWS)]
    for i, r1 in enumerate(food):
        for r2 in food[i + 1:]:
            if B.word[r1] != B.word[r2]:
                continue
            if DIST is None:
                d = {v: 0.5 * np.abs(view_norm(B, r1, v, L) - view_norm(B, r2, v, L)).sum() for v in CAUSE_VIEWS}
                dt = 0.5 * np.abs(view_norm(B, r1, "味", L) - view_norm(B, r2, "味", L)).sum()
            else:
                d = {v: DIST(B, r1, r2, v) for v in CAUSE_VIEWS}
                dt = DIST(B, r1, r2, "味")
            out.append((B.word[r1], d, dt, "語板どうし"))
    inner = [r for r, w in enumerate(B.word) if w != exclude_word and B.known(r, "味", L)]
    for r in inner:                                                # 1 枚の中: 2 つの値がある観点（光の届かない観点は同じ物なので差 0）
        multi = {v: B.known(r, v, L) and np.sort(view_norm(B, r, v, L))[-2] >= B.dark * view_norm(B, r, v, L).max()   # 2 番目の値も暗くない
                 for v in CAUSE_VIEWS + ["味"]}
        if multi["味"]:
            continue
        for v in CAUSE_VIEWS:
            if multi[v] and not any(multi[u] for u in CAUSE_VIEWS if u != v):
                out.append((B.word[r], {u: (1.0 if u == v else 0.0) for u in CAUSE_VIEWS}, 0.0, "1 枚の中"))
    return out


def effects(P, how):
    """観点ごとの味への効き目 (0〜1)。"""
    if how == "corr":
        return {v: 1.0 for v in CAUSE_VIEWS}
    if not P:
        return {v: 0.0 for v in CAUSE_VIEWS}
    if how == "deny":
        e = {}
        for v in CAUSE_VIEWS:
            refuted = any(d[v] >= 0.5 and dt < 0.2 for _, d, dt, _ in P)
            support = any(d[v] >= 0.5 and dt >= 0.5 for _, d, dt, _ in P)
            e[v] = 1.0 if (support and not refuted) else 0.0
        return e
    X = np.array([[d[v] for v in CAUSE_VIEWS] for _, d, _, _ in P])
    y = np.array([dt for _, _, dt, _ in P])
    a = np.linalg.lstsq(X, y, rcond=None)[0]
    return {v: float(np.clip(a[i], 0, 1)) for i, v in enumerate(CAUSE_VIEWS)}


def questions():
    """介入の問いと見たままの問い。(種類, 語, トークン, 正解, 付帯)"""
    Q = []
    for w in FRUIT_WORDS:
        ss = [s for s, (ww, _) in SENSES.items() if ww == w and "する" not in SENSES[s][1]]
        for s in ss:                                               # 塗る: どの色に塗っても味は変わらない
            at = SENSES[s][1]
            for c0 in at["色"]:
                for c1 in VIEWS["色"]:
                    if c1 == c0:
                        continue
                    for z in VIEWS["味"]:
                        if z in at["味"]:
                            toks, ans = [COLOR_TOK[c0], w, "を", PAINT_TOK[c1], "塗ったら", z, "まま", "？"], True
                        else:
                            toks, ans = [COLOR_TOK[c0], w, "を", PAINT_TOK[c1], "塗ったら", TASTE_NARU[z], "なる", "？"], False
                        Q.append(dict(kind="塗る", word=w, toks=toks, ans=ans, src=[("色", c0)], change=("色", c1), taste=z,
                                      form="まま" if z in at["味"] else "なる"))
        un = [s for s in ss if SENSES[s][1]["状態"] == ["未熟"]]
        ri = [s for s in ss if SENSES[s][1]["状態"] == ["完熟"]]
        if un and ri:                                              # 熟させる: 完熟の味になる
            a0, a1 = SENSES[un[0]][1], SENSES[ri[0]][1]
            c0 = a0["色"][0]
            for z in VIEWS["味"]:
                if z in a0["味"]:
                    toks, ans, form = [COLOR_TOK[c0], w, "を", "熟させたら", z, "まま", "？"], z in a1["味"], "まま"
                else:
                    toks, ans, form = [COLOR_TOK[c0], w, "を", "熟させたら", TASTE_NARU[z], "なる", "？"], z in a1["味"], "なる"
                Q.append(dict(kind="熟す", word=w, toks=toks, ans=ans, src=[("色", c0)], change=("状態", "完熟"), taste=z, form=form))
    for s, (w, at) in SENSES.items():                             # 見たまま
        if "味" not in at:
            continue
        for c0 in at["色"]:
            for z in VIEWS["味"]:
                Q.append(dict(kind="見たまま", word=w, toks=[COLOR_TOK[c0], w, "は", z, "？"], ans=z in at["味"], src=[("色", c0)], taste=z))
    return Q


def answer(B, q, eff):
    """語板と効き目から、はい／いいえ。"""
    L = B.light()
    _, r0 = B.route(q["word"], q["src"])
    if r0 is None:
        return False
    t0 = view_norm(B, r0, "味", L)
    if q["kind"] == "見たまま":
        t = t0
    else:
        v, x = q["change"]
        cand = [(B.diff(r, [(v, x)], L)[0], -B.diff(r, [(v, x)], L)[1], r) for r, w in enumerate(B.word)
                if w == q["word"] and B.known(r, "味", L)]
        cand.sort()
        r1 = cand[0][2] if cand and cand[0][0] < 0.5 and cand[0][1] < 0 else None   # 変えた後の姿の語板（無ければ元のまま）
        if r1 is None:
            t = t0
        else:
            dv = {u: 0.5 * np.abs(view_norm(B, r0, u, L) - view_norm(B, r1, u, L)).sum() if B.known(r0, u, L) and B.known(r1, u, L) else 0.0
                  for u in CAUSE_VIEWS}
            if VERB_RULE:
                c = min(1.0, eff[v] * dv[v])                        # 動詞が変える観点だけ（手で与えた規則）
            else:                                                   # T1: 規則なし = 動詞がどの観点を変えるか知らないので、元と変えた後で違う観点は全部変わるとみなす
                c = min(1.0, max(eff[u] * dv[u] for u in CAUSE_VIEWS))
            t1 = view_norm(B, r1, "味", L)
            t = t0 if t1 is None or t0 is None else (1 - c) * t0 + c * t1
    if t is None:
        return False
    return bool(t[VIEWS["味"].index(q["taste"])] >= 0.5 * t.max())


# ---------------------------------------------------------------- 比べる相手（次の語当て）
class GRULM(torch.nn.Module):
    def __init__(self, V, d=64, h=128):
        super().__init__()
        self.e, self.g, self.o = torch.nn.Embedding(V, d), torch.nn.GRU(d, h, batch_first=True), torch.nn.Linear(h, V)

    def forward(self, x):
        return self.o(self.g(self.e(x))[0])


class TfLM(torch.nn.Module):
    def __init__(self, V, d=64, layers=2, heads=4, T=40):
        super().__init__()
        self.e, self.p = torch.nn.Embedding(V, d), torch.nn.Parameter(torch.zeros(T, d))
        lay = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True)
        self.t, self.o = torch.nn.TransformerEncoder(lay, layers), torch.nn.Linear(d, V)

    def forward(self, x):
        T = x.shape[1]
        m = torch.nn.Transformer.generate_square_subsequent_mask(T)
        return self.o(self.t(self.e(x) + self.p[:T], mask=m, is_causal=True))


def train_lm(kind, lines, vocab, seed, steps=3000):
    torch.manual_seed(seed)
    V, T = len(vocab), max(len(l) for l in lines) + 1
    X = torch.full((len(lines), T), 0, dtype=torch.long)
    for i, l in enumerate(lines):
        X[i, :len(l)] = torch.tensor([vocab[t] for t in l])
    M = (X != 0).float()
    model = GRULM(V) if kind == "gru" else TfLM(V, T=T)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randint(0, len(lines), (64,), generator=g)
        x, m = X[b], M[b]
        lo = model(x[:, :-1])
        loss = (torch.nn.functional.cross_entropy(lo.transpose(1, 2), x[:, 1:], reduction="none") * m[:, 1:]).sum() / m[:, 1:].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def lm_answer(model, vocab, q):
    toks = ["<s>", "問"] + q["toks"] + ["答"]
    x = torch.tensor([[vocab[t] for t in toks]])
    with torch.no_grad():
        lo = model(x)[0, -1]
    return bool(lo[vocab["はい"]] > lo[vocab["いいえ"]])


def qa_line(q):
    return ["<s>", "問"] + q["toks"] + ["答", "はい" if q["ans"] else "いいえ", "</s>"]


# ---------------------------------------------------------------- 物差し
def auroc(pos, neg):
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else 0.5 if p == n else 0.0
    return s / (len(pos) * len(neg))


def corrupt(sense, toks, rng):
    """属性を 1 つ、その語のどの意味にもない値に変えた、おかしな文。作れなければ None。"""
    w = SENSES[sense][0]
    union = defaultdict(set)
    for s, (ww, at) in SENSES.items():
        if ww == w:
            for v, xs in at.items():
                union[v] |= set(xs)
    opts = []
    for i, t in enumerate(toks):
        if t in COLOR_TOK.values():
            opts += [(i, ct) for c, ct in COLOR_TOK.items() if c not in union["色"]]
        if t in ("丸い", "長い"):
            opts += [(i, x) for x in VIEWS["形"] if x not in union["形"]]
        if t in TASTE_TOK or t == "甘くて":
            opts += [(i, x) for x in VIEWS["味"] if x not in union["味"]]
    if not opts:
        if "する" not in union:                                  # 人でない語に動作をさせる
            return [w, "が", rng.choice(["手を振った", "話した", "歩いた"]), "。"]
        return None
    i, nt = rng.choice(opts)
    return toks[:i] + [nt] + toks[i + 1:]


def run(a):
    t0 = time.perf_counter()
    train, test = make_world(a.seed, n_train=a.n_train)
    B = Boards(a.mode, a.theta, a.seed)
    merged = splits = 0
    for i, (s, toks) in enumerate(train):
        B.read(SENSES[s][0], attrs_of(toks, SENSES[s][0]), s)
        if (i + 1) % 50 == 0:
            if a.split:
                splits += B.split_check()
            if a.consolidate:
                merged += B.consolidate()
    if a.split:
        splits += B.split_check()
    if a.consolidate:
        merged += B.consolidate()
    if a.reread:                                                    # 読み直し: 最後の語板で、教えの文をもう一度振り分けて数え直す
        new_counts = [np.zeros(len(SLOTS)) for _ in B.counts]
        new_mem = [Counter() for _ in B.counts]
        new_occ = [[] for _ in B.counts]
        for s, toks in train:
            w = SENSES[s][0]
            at = attrs_of(toks, w)
            _, r = B.route(w, at)
            for v, x in at:
                new_counts[r][SLOT[(v, x)]] += 1
            new_mem[r][s] += 1
            new_occ[r].append((at, s))
        keep = [r for r in range(len(B.counts)) if new_counts[r].sum() > 0]
        if B.mode == "light":
            B.D.phase.data[:len(keep)] = B.D.phase.data[keep].clone()
            B.opt = torch.optim.Adam(B.D.parameters(), lr=0.02)
        B.word = [B.word[r] for r in keep]
        B.counts = [new_counts[r] for r in keep]
        B.members = [new_mem[r] for r in keep]
        B.occ = [new_occ[r] for r in keep]
        B._cache = None
        B.fit(3 * B.new_steps)
    global LAST_B
    LAST_B = B                                                      # I14 などが語板を使い回すため
    rec = dict(exp="I1", n_train_per_sense=a.n_train, mode=a.mode, theta=a.theta, consolidate=a.consolidate, reread=a.reread, split=a.split, merged=merged, splits=splits, seed=a.seed, n_train=len(train), n_boards=len(B.counts), table_tv=B.tv())
    # (1) 分かれ方
    gold_n = Counter(w for w, _ in SENSES.values())
    rec["boards"] = [dict(word=B.word[r], n=int(sum(B.members[r].values())), members=dict(B.members[r])) for r in range(len(B.counts))]
    rec["boards_per_word"] = {w: sum(1 for x in B.word if x == w) for w in WORDS}
    rec["gold_per_word"] = dict(gold_n)
    pure = sum(m.most_common(1)[0][1] for m in B.members)
    rec["purity"] = pure / sum(sum(m.values()) for m in B.members)
    major = [m.most_common(1)[0][0] for m in B.members]
    rec["senses_found"] = len(set(major))
    # (2) 振り分け（学びを止めて読む）
    ok = n = 0
    for s, toks in test:
        w = SENSES[s][0]
        at = attrs_of(toks, w)
        if determined(w, at) != s:
            continue
        d, r = B.route(w, at)
        n += 1
        ok += r is not None and major[r] == s
    rec["route_acc"], rec["route_n"] = ok / max(n, 1), n
    # (3) 違和感
    rng = random.Random(a.seed + 99)
    pos, neg = [], []
    for s, toks in test:
        w = SENSES[s][0]
        bad = corrupt(s, toks, rng)
        if bad is None:
            continue
        neg.append(B.route(w, attrs_of(toks, w))[0])
        pos.append(B.route(w, attrs_of(bad, w))[0])
    rec["anomaly_auroc"], rec["anomaly_n"] = auroc(pos, neg), len(pos)
    rec["anomaly_mean"] = dict(normal=float(np.mean(neg)), odd=float(np.mean(pos)))
    # (4)(5) 問い
    Q = questions()
    rec["n_q"] = Counter(q["kind"] for q in Q)
    res = {}
    for how in ("deny", "lsq", "corr"):
        for loo in (True, False):
            per = defaultdict(lambda: [0, 0])
            effs = {}
            for q in Q:
                key = q["word"] if loo else None
                if key not in effs:
                    effs[key] = effects(pairs(B, exclude_word=key), how)
                ans = answer(B, q, effs[key])
                per[q["kind"]][0] += ans == q["ans"]
                per[q["kind"]][1] += 1
                if ans != q["ans"] and how == "deny" and loo:
                    res.setdefault("wrong_deny_loo", []).append(" ".join(q["toks"]))
            res[f"{how}{'_loo' if loo else ''}"] = {k: v[0] / v[1] for k, v in per.items()}
            if loo:
                res[f"{how}_effects_loo"] = {k if k else "-": v for k, v in effs.items()}
    rec["qa"] = res
    rec["effects_all"] = {h: effects(pairs(B), h) for h in ("deny", "lsq")}
    rec["pairs"] = [(w, {k: round(float(x), 2) for k, x in d.items()}, round(float(dt), 2), src) for w, d, dt, src in pairs(B)]
    rec["always_no"] = {k: sum(not q["ans"] for q in Q if q["kind"] == k) / sum(q["kind"] == k for q in Q) for k in ("塗る", "熟す", "見たまま")}
    rec["light_s"] = time.perf_counter() - t0
    # 比べる相手
    if a.baselines:
        sent_lines = [["<s>"] + toks + ["</s>"] for _, toks in train]
        obs = [qa_line(q) for q in Q if q["kind"] == "見たまま"]
        vocab = {"<pad>": 0}
        for l in sent_lines + [qa_line(q) for q in Q]:
            for t in l:
                vocab.setdefault(t, len(vocab))
        base = {}
        for kind in ("gru", "tf"):
            for var in ("obs", "int"):
                per = defaultdict(lambda: [0, 0])
                for w in FRUIT_WORDS:
                    extra = [qa_line(q) for q in Q if q["kind"] != "見たまま" and q["word"] != w] if var == "int" else []
                    m = train_lm(kind, sent_lines + obs * 3 + extra * 3, vocab, a.seed, a.lm_steps)
                    for q in Q:
                        if (q["kind"] == "見たまま" and w != FRUIT_WORDS[0]) or (q["kind"] != "見たまま" and q["word"] != w):
                            continue
                        ans = lm_answer(m, vocab, q)
                        per[q["kind"]][0] += ans == q["ans"]
                        per[q["kind"]][1] += 1
                base[f"{kind}_{var}"] = {k: v[0] / v[1] for k, v in per.items()}
                print(kind, var, base[f"{kind}_{var}"], flush=True)
        rec["baselines"] = base
    rec["total_s"] = time.perf_counter() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="light", choices=["light", "digital"])
    ap.add_argument("--theta", type=float, default=0.5, help="分けるしきい値（違いがこれ以上なら新しい語板）")
    ap.add_argument("--consolidate", type=int, default=1, help="50 文ごとと最後に、違う観点が 1 つ以下の語板をまとめる")
    ap.add_argument("--split", type=int, default=1, help="2 つの関係を持つ語板を分け直す")
    ap.add_argument("--reread", type=int, default=1, help="最後の語板で教えの文を読み直して数え直す")
    ap.add_argument("--n_train", type=int, default=50, help="意味 1 つあたりの教えの文")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", type=int, default=1)
    ap.add_argument("--lm_steps", type=int, default=3000)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i01.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(4)
    rec = run(a)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("boards", "pairs")}, ensure_ascii=False, default=str))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
