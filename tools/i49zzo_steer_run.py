"""I49zzo: 状況波を入れ替えて、出る文の終わりが変わるかを見る（持ち主の頼み、未確定の試し）。

持ち主の頼み:「ちょっと状況波を変えて文章が変わるかの試験をしてくれる？時の波以外にやり取りの波や言葉遣いの波を変えてくれる？」
作り方: I49zzm の波（前の文の終わり → 今の文の終わり）を材料の全部から学ぶ。状況は「前の文がこう終わっていた」という形で手で入れる（入れる終わりは PROBES）。
  同じ話し手が続ける時の波（同じ発話の中）と、次の話し手が受ける時の波（1 個前の発話 → 今の発話）の 2 つを持つ。
  候補は、材料に COUNT 回以上出た文の終わり（6 文字まで）の全部。状況との合い方（点の足し算）の高い順に並べる。
出す物:
  1. 状況ごとに、上から TOP 個の終わりのうち、印（丁寧・過去・問い・古風）の付いた物の割合。状況を入れ替えると割合が動くか。
  2. 読むための例: 同じ字（行・知・思・来 など）を含む終わりに絞って、状況ごとの上から 3 つ。
手で置いた物: 入れる終わり PROBES、印の決め方（古風の印はここで足した）、COUNT、TOP、例に使う字。
  3. `--rewrite`: 行を選び（材料から黄金比の並びで PICK 行と、自分で入れた行）、終わりだけを状況ごとに書き替えた行を出す。
     終わりの替え方は、つなぎ目の表（左の 2 文字の後に、材料でその終わりが続いた事が有るか）で決める。分ける道具は使わない。
出力には材料の文や文の終わりが入るので、リポジトリには入れない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49_situation as SIT
import i49zzl_tense_text_run as ZL
import i49zzm_prev_tail_run as ZM
from i49_situation import across_pairs, joins, rewrite  # noqa: F401  書き替えの部品は i49_situation に有る


def marks(tail):
    """読むための印（古風も入れる）。"""
    return SIT.marks(tail, old=True)

COUNT = 20   # 候補にする終わりの、出た回数の下限（手で置いた値）
TOP = 50     # 割合を見る上位の数（手で置いた値）
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


PICK = 6     # 材料から抜き出す行の数（手で置いた値）
MODE_OF = {"間柄": "同じ話し手が続ける", "言葉づかい": "同じ話し手が続ける", "時": "次の話し手が受ける", "やり取り": "次の話し手が受ける"}


def run_rewrite(lines, extra=()):
    sents = ZL.distinct(list(ZL.sentences(lines)))
    table = joins(sents)
    inside, _ = ZM.pairs_of(lines)
    assoc = {"同じ話し手が続ける": ZM.Assoc(inside), "次の話し手が受ける": ZM.Assoc(list(across_pairs(ZM.runs_of(lines))))}
    pool = [s for s in sents if 10 <= len(s) <= 22]
    picked = [pool[i] for i in ZL.golden_order(len(pool))[:PICK]] if pool else []
    out = []
    for sent in list(extra) + picked:
        row = dict(元の行=sent, 書き替え={})
        for name, (wave, _, probe) in PROBES.items():
            r = rewrite(sent, probe, assoc[MODE_OF[wave]], table, n=2)
            row["書き替え"][name] = [x for x, _ in r]
        out.append(row)
    return dict(文=len(sents), つなぎ目の種類=len(table), 行ごと=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite", action="store_true")   # 行の終わりを、状況ごとに書き替えて出す
    ap.add_argument("--lines", default="")              # 自分で入れる行（｜で区切る）
    ap.add_argument("--text", required=True)   # 空の行を続きの切れ目にした、1 行 1 発話のテキスト
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    lines = []
    for f in a.text.split(","):
        lines += open(f, encoding="utf-8", errors="replace").read().split("\n")
    if a.rewrite:
        res = dict(手で置いた値=dict(JOIN=SIT.JOIN, SEEN=SIT.SEEN, PICK=PICK, TAIL=ZL.TAIL), **run_rewrite(lines, [x for x in a.lines.split("｜") if x]))
        res["秒"] = round(time.time() - t0)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    inside, _ = ZM.pairs_of(lines)
    across = list(across_pairs(ZM.runs_of(lines)))
    res = dict(行=len(lines), 手で置いた値=dict(COUNT=COUNT, TOP=TOP, TAIL=ZL.TAIL, SEATS=SIT.SEATS, MIN=SIT.MIN),
               同じ話し手が続ける=steer(ZM.Assoc(inside), candidates([c for _, c in inside])),
               次の話し手が受ける=steer(ZM.Assoc(across), candidates([c for _, c in across])))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
