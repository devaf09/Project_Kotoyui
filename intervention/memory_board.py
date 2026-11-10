"""介入の研究 I40a: 記憶板（行の板 → 話題・会話・日・週の段）を書く（融合の忘れはまだ無し）

持ち主（2026-11-10〜11）:「会話を行にする → 行板作る → 記憶板へ反映」「融合で 秒 → 分 → 時 → 日 → 週」
  置き場は PC の非公開フォルダ（暗号化はクラウドに置く時だけ）。試しでは 1 会話 = 1 時、20 会話 = 1 日（将来は実時間）。
設計: docs/記憶板の設計.md、図 figs/mech_14_memory_board.png（非公開の置き場）。
ここでは進める順の 1: 行の板を作り、下の段が閉じたら和で上の段を 1 枚作るだけ（期待との差で忘れるのは I40b）。
  行の板: 所 ("組", 名詞, 助詞:述語)、話題の名詞 ("名", 名詞)、誰、時刻、会話、結果（相手の次の行で 良・指摘）
  段: 分 = 話題の名詞が替わるまで、時 = 1 会話、日 = DAY 会話、週 = 7 日
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter

LEVELS = ("分", "時", "日", "週")
NOUN = ("NOUN", "PROPN")


def line_board(toks, pieces, who, conv, t=None):
    keys = Counter()
    for pc in pieces:
        keys[f"組|{pc['n']}|{pc['c']}:{pc['p']}"] += 1
    nouns = [x["lemma"] for x in toks if x["pos"] in NOUN]
    for n in nouns:
        keys[f"名|{n}"] += 1
    return dict(who=who, conv=conv, t=t if t is not None else time.time(), keys=dict(keys), nouns=nouns, result=None)


class Memory:
    DAY = 20                                               # 試し: 20 会話 = 1 日
    WEEK = 7

    def __init__(self, path=None):
        self.path = path
        self.lines = []                                     # 行の板（秒）
        self.open = {lv: Counter() for lv in LEVELS}        # 閉じていない段の板
        self.boards = {lv: [] for lv in LEVELS}             # 閉じた段の板: dict(keys, from, to, n)
        self._span = {lv: [None, None, 0] for lv in LEVELS}
        self._topic = set()
        self.n_conv = 0
        if path and os.path.exists(path):
            self.load(path)

    # ---- 書く ----
    def add(self, toks, pieces, who, conv):
        lb = line_board(toks, pieces, who, conv)
        nouns = set(lb["nouns"])
        if who == "人" and nouns and self._topic and not (nouns & self._topic):   # 話題の名詞が替わった: 分の段を閉じる
            self.close("分")
        if who == "人" and nouns:
            self._topic = nouns if not (nouns & self._topic) else (self._topic | nouns)
        self.lines.append(lb)
        self._push("分", lb["keys"], lb["t"])
        return lb

    def result(self, lab):
        """前のボットの行の結果（相手の次の行で 良・指摘）。"""
        for lb in reversed(self.lines):
            if lb["who"] == "ボット":
                lb["result"] = lab
                return

    def _push(self, lv, keys, t, n=1):
        self.open[lv].update(keys)
        sp = self._span[lv]
        sp[0] = t if sp[0] is None else sp[0]
        sp[1] = t
        sp[2] += n

    def close(self, lv):
        """段を閉じて 1 枚にし、上の段に足す（親の語板 = 子の語板の和、と同じ）。"""
        if not self.open[lv]:
            return None
        sp = self._span[lv]
        b = dict(keys=dict(self.open[lv]), start=sp[0], end=sp[1], n=sp[2])
        self.boards[lv].append(b)
        i = LEVELS.index(lv)
        if i + 1 < len(LEVELS):
            self._push(LEVELS[i + 1], b["keys"], b["end"])
        self.open[lv] = Counter()
        self._span[lv] = [None, None, 0]
        if lv == "分":
            self._topic = set()
        return b

    def end_conv(self):
        self.close("分")
        self.close("時")
        self.n_conv += 1
        if self.n_conv % self.DAY == 0:
            self.close("日")
            if (self.n_conv // self.DAY) % self.WEEK == 0:
                self.close("週")
        if self.path:
            self.save(self.path)

    # ---- 読む（I40d で返事に使う。今は数えるだけ） ----
    def recall(self, words):
        """新しい段から順に、語のある鍵を探す: [(段, 何番目, 鍵, 数)]。"""
        out = []
        for lv in LEVELS:
            for i in range(len(self.boards[lv]) - 1, -1, -1):
                for k, v in self.boards[lv][i]["keys"].items():
                    if any(f"|{w}|" in k + "|" for w in words):
                        out.append((lv, i, k, v))
        return out

    def summary(self):
        return {lv: dict(枚=len(self.boards[lv]), 鍵=sum(len(b["keys"]) for b in self.boards[lv])) for lv in LEVELS} | \
            dict(行=len(self.lines))

    # ---- 置き場（非公開。PC のフォルダに JSON） ----
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = dict(lines=self.lines, boards=self.boards, open={k: dict(v) for k, v in self.open.items()},
                 span=self._span, n_conv=self.n_conv, day=self.DAY)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    def load(self, path):
        d = json.load(open(path, encoding="utf-8"))
        self.lines, self.boards, self.n_conv = d["lines"], d["boards"], d["n_conv"]
        self.open = {k: Counter(v) for k, v in d["open"].items()}
        self._span = d["span"]
