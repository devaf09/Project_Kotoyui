"""I49zzo: 状況波を入れ替えて、出る文の終わりが変わるかを見る（持ち主の頼み、未確定の試し）。

持ち主の頼み:「ちょっと状況波を変えて文章が変わるかの試験をしてくれる？時の波以外にやり取りの波や言葉遣いの波を変えてくれる？」
作り方: I49zzm の波（前の文の終わり → 今の文の終わり）を材料の全部から学ぶ。状況は「前の文がこう終わっていた」という形で手で入れる（入れる終わりは PROBES）。
  同じ話し手が続ける時の波（同じ発話の中）と、次の話し手が受ける時の波（1 個前の発話 → 今の発話）の 2 つを持つ。
  候補は、材料に COUNT 回以上出た文の終わり（6 文字まで）の全部。状況との合い方（点の足し算）の高い順に並べる。
出す物:
  1. 状況ごとに、上から TOP 個の終わりのうち、印（丁寧・過去・問い・古風）の付いた物の割合。状況を入れ替えると割合が動くか。
  2. 読むための例: 同じ字（行・知・思・来 など）を含む終わりに絞って、状況ごとの上から 3 つ。
手で置いた物: 入れる終わり PROBES、印の決め方（古風の印はここで足した）、COUNT、TOP、例に使う字。
出力には材料の文の終わり（6 文字まで）が入るので、リポジトリには入れない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49zzl_tense_text_run as ZL
import i49zzm_prev_tail_run as ZM

COUNT = 20   # 候補にする終わりの、出た回数の下限（手で置いた値）
TOP = 50     # 割合を見る上位の数（手で置いた値）
OLD = ("ござ", "まする", "おりま", "じゃ", "ぬ", "申し")
# 状況の名前 → (波の名前, 向き, 前の文の終わりとして入れる物)
PROBES = {
    "間柄: 丁寧": ("間柄", "+", ("ですね", "ました", "ますよ")),
    "間柄: くだけ": ("間柄", "-", ("だよ", "んだ", "だぜ")),
    "時: 過去": ("時", "+", ("ったんだ", "いました", "だった")),
    "時: 今": ("時", "-", ("ている", "います", "るんだ")),
    "言葉づかい: 古風": ("言葉づかい", "+", ("ござります", "じゃ", "おりまする")),
    "言葉づかい: 今風": ("言葉づかい", "-", ("んですよ", "のです", "ですね")),
    "やり取り: 問われた": ("やり取り", "+", ("ですか", "のか", "かい")),
    "やり取り: 言い切られた": ("やり取り", "-", ("です", "だ", "よ")),
}
MARK_OF = {"間柄": "丁寧", "時": "過去", "言葉づかい": "古風", "やり取り": "問い"}
STEMS = ("行", "知", "思", "来", "分", "い")


def marks(tail):
    m = ZM.marks(tail)
    m["古風"] = any(x in tail for x in OLD)
    return m


def candidates(tails):
    c = {}
    for t in tails:
        c[t] = c.get(t, 0) + 1
    return sorted((t for t, n in c.items() if n >= COUNT), key=lambda t: (-c[t], t))


def rank(assoc, probe, cands):
    return sorted(cands, key=lambda c: (-sum(assoc.score(p, c) for p in probe), c))


def steer(assoc, cands):
    base = {k: round(sum(marks(c)[k] for c in cands) / max(len(cands), 1), 4) for k in ("丁寧", "過去", "問い", "古風")}
    share, examples = {}, {}
    for name, (wave, _, probe) in PROBES.items():
        order = rank(assoc, probe, cands)
        top = order[:TOP]
        share[name] = {k: round(sum(marks(c)[k] for c in top) / max(len(top), 1), 4) for k in base}
        share[name]["見る印"] = MARK_OF[wave]
        examples[name] = dict(上から=top[:8], 字ごと={s: [c for c in order if s in c][:3] for s in STEMS})
    return dict(候補=len(cands), 候補の全体の割合=base, 状況ごとの上位の割合=share, 例=examples)


def across_pairs(runs):
    """1 個前の発話の最後の文の終わり → 今の発話の初めの文の終わり（続きの切れ目はまたがない。同じ組は 1 回）。"""
    across = {}
    for r in runs:
        for i in range(1, len(r)):
            across.setdefault((r[i - 1][1], r[i][0]), None)
    return across


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)   # 空の行を続きの切れ目にした、1 行 1 発話のテキスト
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    lines = []
    for f in a.text.split(","):
        lines += open(f, encoding="utf-8", errors="replace").read().split("\n")
    inside, _ = ZM.pairs_of(lines)
    across = list(across_pairs(ZM.runs_of(lines)))
    res = dict(行=len(lines), 手で置いた値=dict(COUNT=COUNT, TOP=TOP, TAIL=ZL.TAIL, SEATS=ZM.SEATS, MIN=ZM.MIN),
               同じ話し手が続ける=steer(ZM.Assoc(inside), candidates([c for _, c in inside])),
               次の話し手が受ける=steer(ZM.Assoc(across), candidates([c for _, c in across])))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
