"""I49zzj: 状況波（時制）の小さな実験を、作り物の文で数える（持ち主の案、未確定の試し）。

作り物の世界: 動詞 6 つ（形は 5 つずつ）、だれが 6 つ、時の語 8 つ。時の語ごとに、一緒に出てよい時制を決めて文を作る（世界の決まり）。
  学習: 「時の語、だれが＋動詞の形」の文を黄金比の並びで聞き、時の語に時制の波を足す。学習には動詞 4 つだけを使う。
数える物:
  1. 学習の進み: 聞いた文の数ごとに、時の語の一番近い時制が世界の決まりに合う割合。
  2. 時の語と時制の近さの表（学習の後）。
  3. 形を選ぶ: 学習に出ていない動詞 2 つで、時の語から状況波を作り、近い形を選ぶ。世界の決まりに合う割合。
  4. 会話: 時の語の有る文の後に、時の語の無い文を 2 つ続ける。状況波が残り、後の文の形も合う割合。
  どれも「山の形のまま持つ」と「周波数 1 つにまとめて持つ」の 2 通りで数える。
手で置いた物: 世界の決まり（どの時の語がどの時制と出るか、その重さ）、学習と試しの動詞の分け方、部品の側の値。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))

import i49zzj_tense_wave as TW

FORMS = {
    "歩く": dict(未来="歩くだろう", 現在="歩く", 現在進行="歩いている", 過去="歩いた", 過去進行="歩いていた"),
    "走る": dict(未来="走るだろう", 現在="走る", 現在進行="走っている", 過去="走った", 過去進行="走っていた"),
    "食べる": dict(未来="食べるだろう", 現在="食べる", 現在進行="食べている", 過去="食べた", 過去進行="食べていた"),
    "寝る": dict(未来="寝るだろう", 現在="寝る", 現在進行="寝ている", 過去="寝た", 過去進行="寝ていた"),
    "笑う": dict(未来="笑うだろう", 現在="笑う", 現在進行="笑っている", 過去="笑った", 過去進行="笑っていた"),
    "泣く": dict(未来="泣くだろう", 現在="泣く", 現在進行="泣いている", 過去="泣いた", 過去進行="泣いていた"),
}
WHO = ("犬", "子供", "人", "彼", "猫", "母")
# 世界の決まり: 時の語 → 一緒に出る時制と重さ
RULE = {
    "今": dict(現在進行=3, 現在=1),
    "さっき": dict(過去=3, 過去進行=1),
    "今まで": dict(過去進行=2, 現在進行=2),
    "これから": dict(未来=4),
    "昨日": dict(過去=3, 過去進行=1),
    "明日": dict(未来=4),
    "いつも": dict(現在=3, 現在進行=1),
    "ずっと": dict(現在進行=2, 過去進行=2),
}
TRAIN = ("歩く", "走る", "食べる", "寝る")
TEST = ("笑う", "泣く")
PHI = 0.6180339887498949


def golden(items):
    return [x for _, x in sorted((((i + 1) * PHI) % 1.0, x) for i, x in enumerate(items))]


def sentences(verbs):
    out = []
    for word, rule in RULE.items():
        for tense, w in rule.items():
            for _ in range(w):
                for who in WHO:
                    for v in verbs:
                        out.append((word, who, FORMS[v][tense]))
    return golden(out)


def top_ok(board, tw, single):
    ok = 0
    for word, rule in RULE.items():
        v = tw.wave(word, single)
        ok += int(v is not None and board.nearest(v)[0] in rule)
    return ok / len(RULE)


def run():
    board = TW.TenseBoard(FORMS)
    tw = TW.TimeWords(board)
    seq = sentences(TRAIN)
    marks = [1, 2, 4, 8, 16, 32, 64, 128, 256, len(seq)]
    curve = []
    for i, (word, _, form) in enumerate(seq, 1):
        tw.learn(word, form)
        if i in marks:
            curve.append(dict(文=i, 山の形のまま=round(top_ok(board, tw, False), 4), 周波数1つ=round(top_ok(board, tw, True), 4)))
    table = {}
    for word, rule in RULE.items():
        near, c = board.nearest(tw.wave(word))
        near1, _ = board.nearest(tw.wave(word, True))
        table[word] = dict(世界の決まり=rule, 近さ={t: round(x, 3) for t, x in c.items()}, 一番近い=near,
                           周波数1つの位置=round(tw.fsum[word] / tw.n[word], 2), 周波数1つで一番近い=near1)
    pick = {}
    for name, single in (("山の形のまま", False), ("周波数1つ", True)):
        ok = n = 0
        for word, rule in RULE.items():
            for v in TEST:
                s = TW.Situation()
                s.hear(tw.wave(word, single))
                ok += int(board.tense_of[board.pick_form(v, s.state)] in rule)
                n += 1
        pick[name] = dict(問い=n, 合う=ok, 割合=round(ok / n, 4))
    pick["当てずっぽう"] = round(sum(len(r) for r in RULE.values()) / (len(RULE) * len(board.order)), 4)
    talk = {}
    words = golden(list(RULE))
    for name, single in (("山の形のまま", False), ("周波数1つ", True)):
        s = TW.Situation()
        ok = [0, 0, 0]
        for word in words:
            s.hear(tw.wave(word, single))
            for k, v in enumerate((TEST[0], TEST[1], TEST[0])):
                form = board.pick_form(v, s.state)
                ok[k] += int(board.tense_of[form] in RULE[word])
                s.hear(board.wave_of_form(form))      # 自分で選んだ形も聞く
        talk[name] = dict(時の語=len(words), 直後の文=ok[0], その次の文=ok[1], その次の次の文=ok[2])
    return dict(時制の山の位置=board.center, 隣の時制との近さ=round(TW.cos(board.wave("未来"), board.wave("現在")), 3),
                学習の文=len(seq), 学習の進み=curve, 時の語と時制=table, 形を選ぶ=pick, 会話=talk)


def sweep(widths=(2.0, 4.0, 6.0, 8.0, 12.0), halves=(0.25, 0.5, 1.0, 2.0)):
    """山の幅と、状況波の薄れる速さを変えて、同じ数えをする（触る場所でどう変わるかを見る）。"""
    keep = (TW.WIDTH, TW.HALF)
    out = []
    try:
        for w in widths:
            for h in halves:
                TW.WIDTH, TW.HALF = w, h
                r = run()
                out.append(dict(山の幅=w, 薄れる速さ=h, 隣の時制との近さ=r["隣の時制との近さ"],
                                時の語が合う=r["学習の進み"][-1], 形を選ぶ={k: v["割合"] for k, v in r["形を選ぶ"].items() if isinstance(v, dict)},
                                会話=r["会話"]))
    finally:
        TW.WIDTH, TW.HALF = keep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    res = dict(振る=sweep()) if a.sweep else run()
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
