"""I49e 二歩目の下ごしらえ: 先生 2 人の読む記録（tools/i49e_read_together.py の jsonl）を番の並びにして、
聞くだけで育つ拾われやすさ（intervention/i49e_listen.LivePickup、部品は変えない）に聞かせ、数える。

番の順: ページ（話し手「本」）→ A1（話し手 A）→ B1（B）→ A2（A）→ 次のページ。作品が替わる所では直前の番を切る（数えは持ち越す）。
内容語は I48 の CONTENT から STOP を除いた (原形, 品詞)（GiNZA）。
組み合わせ: prior なし ／ prior = 会話の資料の拾われやすさの表（tools/i49d_pickup.PickupTable.value）、
  それぞれ 全部の番 ／ 先生どうしだけ（ページを番に入れない。A2 → 次のページの A1 は同じ話し手なので数えに入らない）。
出す数: report()（全体・前半・後半）、番の種類ごと（本→A1、A1→B1、B1→A2、A2→本 など、直前の番 → 今の番）の拾われた割合と AUC
  （初めの見込みだけ ／ 聞いて育てた見込み）、作品ごとの同じ数。数だけで、語や本文は出さない。
手で置く物: 作品の順（--logs に並べた順）、番の順。
python tools/i49e_listen_count.py --src <青空の本文> --meta <目録の csv> --logs <記録1.jsonl>,<記録2.jsonl>,... --books 1,2,... \
    --pickup <拾われやすさの表.json> --out <stats.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

KEYS = (("A1", "A"), ("B1", "B"), ("A2", "A"))


def words_of(nlp, texts):
    import i48_wave_board as W48
    return [[(t.lemma_, t.pos_) for t in d if t.pos_ in W48.CONTENT and t.lemma_ not in W48.STOP] for d in nlp.pipe(texts, batch_size=64)]


def turns_of(nlp, log, pages_text):
    """1 作品の番の並び [(種類, 話し手, 語)]。種類は 本・A1・B1・A2。"""
    rows = [json.loads(l) for l in open(log, encoding="utf-8")]
    rows = [r for r in rows if "ページ" in r]
    texts, kinds = [], []
    for r in rows:
        texts.append(pages_text[r["ページ"] - 1])
        kinds.append(("本", "本"))
        for k, who in KEYS:
            texts.append(r["発言"][k])
            kinds.append((k, who))
    W = words_of(nlp, texts)
    return [(k, who, w) for (k, who), w in zip(kinds, W)]


def r4(v):
    return None if v is None else round(float(v), 4)


def summary(rec_rows, auc):
    if not rec_rows:
        return dict(語の数=0)
    p0 = np.array([x[1] for x in rec_rows])
    p1 = np.array([x[2] for x in rec_rows])
    y = np.array([x[3] for x in rec_rows], dtype=bool)
    return dict(語の数=len(y), 拾われた割合=r4(y.mean()), 初めの見込みだけ=r4(auc(p0, y)), 聞いて育てた見込み=r4(auc(p1, y)))


def listen(books, prior, with_pages):
    import i49e_listen as L
    lp = L.LivePickup(prior=prior)
    meta = {}                                             # 番の番号 → (直前の種類 → 今の種類, 作品の番号)
    for bi, turns in enumerate(books):
        lp.last = None                                    # 作品が替わる所では直前の番を切る
        prev = None
        for k, who, w in turns:
            if k == "本" and not with_pages:
                continue
            meta[lp.n] = (f"{prev}→{k}" if prev else None, bi + 1)
            lp.hear(who, w)
            prev = k
    rec = lp.rec
    kinds = sorted({meta[n][0] for n, *_ in rec})
    by_kind = {k: summary([x for x in rec if meta[x[0]][0] == k], L.auc) for k in kinds}
    by_book = {str(b): summary([x for x in rec if meta[x[0]][1] == b], L.auc) for b in range(1, len(books) + 1)}
    return dict(report=lp.report(), 番の種類ごと=by_kind, 作品ごと=by_book)


def main():
    import spacy
    import i49d_pickup as PU
    import i49e_read_together as RT
    ap = argparse.ArgumentParser()
    for k in ("src", "meta", "logs", "books", "pickup", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    logs, ranks = a.logs.split(","), [int(x) for x in a.books.split(",")]
    nlp = spacy.load("ja_ginza")
    books, info = [], []
    for log, rk in zip(logs, ranks):
        bi, S = RT.pick_book(a.src, a.meta, rank=rk)
        head = json.loads(open(log, encoding="utf-8").readline())
        assert head["本"]["作品ID"] == bi["作品ID"], "記録と本の並びの番号が合わない"
        books.append(turns_of(nlp, log, RT.pages_of(S)))
        info.append(dict(並びの番号=rk, 文の数=bi["文の数"], 番の数=len(books[-1])))
    tb = PU.PickupTable.load(a.pickup)
    res = dict(作品=info, 組み合わせ={})
    for pname, prior in (("prior なし", None), ("prior = 会話の表", tb.value)):
        for mname, wp in (("全部の番", True), ("先生どうしだけ", False)):
            res["組み合わせ"][f"{pname}・{mname}"] = listen(books, prior, wp)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
