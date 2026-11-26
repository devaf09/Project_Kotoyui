"""I49zzzb: 会話（相手の行 → 返事）で、だれが・何を・どうした がどれだけ引き継がれるか（持ち主の案 3-2。数えるだけ。未確定の試し）。

持ち主の案: 「物語の地の文ではなく「相手の発話 → 自分の返事」のペアにおいて、だれが の引き継ぎ率と遷移率がどう変わるかを確認する。」
74〜78 節の数えは、1 人が書いた「A が B を C する」の文だった。ここでは解析済みの会話（雑談など）で同じ事を見る。
行からの取り方（仮の切り出し。解析の係り受けをそのまま使う）:
  どうした = 行の最後の文の根（ROOT）の語（動詞か形容詞の時だけ）。だれが = その根に nsubj で係る語。何を = その根に obj で係る語。語は原形（lemma）。
  会話の行は、主語や目的語を言わない事が多い。取れた行だけで数える事になるので、取れた行の割合も出す（偏りの大きさ）。
組: 相手の行 → 返事（1 個前の行。話し手が違う時だけ）／同じ話し手の 1 個前の行 → 今の行／離れた行（同じ会話の、半分ずれた行。たまたま同じになる割合の比べ役）。
  席ごとに、両方の行で取れた組だけを数え、同じ語だった割合を出す。
だれが は、77 節と同じ分け方（続く・1 つ前の別の語へ戻る・もっと前に出た語へ戻る・初めての語）も、会話の中の「だれが を言った行」の並びで数える。
  一人称の語（私・僕 など）の割合も出す（会話では「私」が話し手ごとに別の人を指す。語が同じでも人は違う）。
話し手は「同じかどうか」だけを見る。語や行の例は出さない（数だけ）。乱数は使わない。
手で置いた物: 取り方（最後の文の根、nsubj、obj）、一人称の語の並び、離れた行のずらし方。
引数: --cache 解析済み JSON --out 記録.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49zzz_talk_memory_run as Z  # noqa: E402

SLOTS = ("だれが", "何を", "どうした")
PAIRS = ("相手の行 → 返事", "同じ話し手の 1 個前の行 → 今の行", "離れた行（比べ役）")
FIRST = ("私", "わたし", "僕", "ぼく", "俺", "おれ", "自分", "あたし", "うち", "わたくし")


def slots(toks):
    """→ (だれが, 何を, どうした)。取れない席は None。"""
    roots = [t for t in toks if t["dep"] == "ROOT"]
    if not roots:
        return (None, None, None)
    r = roots[-1]
    kids = {t["dep"]: t["lemma"] for t in toks if t["head"] == r["i"] and t["i"] != r["i"] and t["dep"] in ("nsubj", "obj")}
    return (kids.get("nsubj"), kids.get("obj"), r["lemma"] if r["pos"] in ("VERB", "ADJ") else None)


def count(works):
    lines = dict(行=0, **{s: 0 for s in SLOTS})
    out = {p: {s: [0, 0] for s in SLOTS} for p in PAIRS}
    kinds = {k: 0 for k in Z.KINDS}
    first = [0, 0]
    for w in works:
        rows = [(l.get("speaker"), slots(l["toks"])) for l in w["lines"] if l["toks"]]
        own = {}
        hist = []
        for i, (who, sl) in enumerate(rows):
            lines["行"] += 1
            for si, s in enumerate(SLOTS):
                lines[s] += sl[si] is not None
            prev = rows[i - 1] if i > 0 and rows[i - 1][0] != who else None
            far = rows[(i + len(rows) // 2) % len(rows)] if len(rows) > 3 else None
            for name, other in zip(PAIRS, (prev, own.get(who), far)):
                if other is None:
                    continue
                for si, s in enumerate(SLOTS):
                    if sl[si] is not None and other[1][si] is not None:
                        out[name][s][0] += 1
                        out[name][s][1] += sl[si] == other[1][si]
            if sl[0] is not None:
                first[0] += 1
                first[1] += sl[0] in FIRST
                if hist:
                    kinds[Z.kind_of(hist, sl[0])[0]] += 1
                hist.append(sl[0])
            own[who] = (who, sl)
    return dict(行の数=lines, 組=out, だれがの分け方=kinds, だれがが一人称=first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) and "works" in works else works
    res = dict(会話=len(works), **count(works))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    n = res["行の数"]["行"]
    print("行", n, {s: round(res["行の数"][s] / n, 3) for s in SLOTS})
    for p in PAIRS:
        print(p, {s: (c[0], round(c[1] / max(c[0], 1), 3)) for s, c in res["組"][p].items()})
    k = res["だれがの分け方"]
    print({x: round(v / max(sum(k.values()), 1), 3) for x, v in k.items()}, "一人称", round(res["だれがが一人称"][1] / max(res["だれがが一人称"][0], 1), 3))


if __name__ == "__main__":
    main()
