"""I49f: 違和感 → 器 の部品（intervention/i49f_odd_to_vessel.OddToGain、中身は変えない）の あり／なし を、同じページで比べる。数えだけ。

材料は 3 人で読む試し（tools/i49e_three_run.py）と同じ: 今ある読む記録（--warm）を拾われやすさ（LivePickup、prior = 会話の表）に聞かせ、
--three_log の記録の作品・ページを 1 ページずつ。先生は呼ばない: 1 ページ = 本 → A1 → B1（記録の物を聞く）→ こ（speak、声に出す）→ A2（記録の物を聞く）。
組み立ては三歩目の ア と同じ（Scored ＋ Pick(今の拾われやすさ, k=3)）。部品ありは Thinker.attach(OddToGain())（seal はしない）。
数える物（なし／あり）: 話した番の数、話した行が なし と同じだった割合、行の語の数の平均、した事の内訳、器の張り付き（波ごとの合計と、いちばん多い波）、
  死んだ刻み、感度の終わりの値（速い波・まん中・遅い波）、違和感の種類ごとの平均の強さ。部品ありでは |倍率 − 1| の平均と最大（波ごと）。
記録はリポジトリの外。手で置く物: k=3、器（Vessel(2048, bands=20, longest=1024)）、まん中の波 = 20 本の 10 番目。
python tools/i49f_odd_run.py --src ... --meta ... --wave <波の板> --scorer <型.pkl> --pickup <拾われやすさの表.json> --warm <記録1.jsonl>,...
    --three_log <three_ア.jsonl> --out <stats.json>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

K = 3
MID = 10


def run_one(live0, rows, pages, wv, scorer, nlp, with_part):
    import i49d_two_loops as TL
    import i49e_listen_count as LC
    import i49e_three as T3
    from i49a_inner_vessel import Vessel
    from i49c_think_loop import Words
    from i49f_odd_to_vessel import OddToGain
    live = copy.deepcopy(live0)
    pos_of = {}
    th = TL.Thinker(Vessel(2048, bands=20, longest=1024), Words(), None, TL.Scored(scorer, pos_of, pick=TL.Pick(T3.live_weight(live, pos_of), K)))
    part = th.attach(OddToGain()) if with_part else None
    seat = T3.Seat(th, live, pos_of)
    words = lambda text: [(l, p, wv.of(l, p)) for l, p in LC.words_of(nlp, [text])[0]]
    lines, t0, last = [], time.time(), None
    for r in rows:
        if r["作品"] != last:
            live.last = None
            last = r["作品"]
        seat.listen("本", words(pages[r["作品"]][r["ページ"] - 1]))
        seat.listen("A", words(r["発言"]["A1"]))
        seat.listen("B", words(r["発言"]["B1"]))
        line = seat.speak(aloud=True)
        lines.append(line)
        seat.listen("A", words(r["発言"]["A2"]))
    odd = Counter()
    nraw = Counter()
    for d in th.odd.raw:
        for k, v in d.items():
            odd[k] += float(v)
            nraw[k] += 1
    w = th.v.watch
    out = dict(秒=round(time.time() - t0), 歩=int(th.n), 内訳=th.counts(),
               話した番=sum(1 for x in lines if x), 行の内容語の数の平均=round(float(np.mean([len([y for y in x if pos_of.get(y)]) for x in lines if x])), 3)
               if any(lines) else None,
               張り付き=dict(合計=int(w["張り付き"].sum()), 波ごと=w["張り付き"].tolist(), いちばん多い波=int(np.argmax(w["張り付き"]))),
               死んだ刻み=int(w["死んだ"]),
               感度の終わり=dict(速い波=round(float(th.v.gain[0]), 4), まん中=round(float(th.v.gain[MID]), 4), 遅い波=round(float(th.v.gain[-1]), 4)),
               違和感の平均の強さ={k: round(odd[k] / max(1, len(th.odd.raw)), 4) for k in sorted(odd)},
               違和感が出た歩の割合={k: round(nraw[k] / max(1, len(th.odd.raw)), 4) for k in sorted(nraw)})
    if part is not None:
        ar = part.arrays()
        if len(ar["factor"]):
            dev = np.abs(ar["factor"] - 1.0)
            out["倍率"] = dict(歩=int(len(dev)), 平均_波ごと=[round(float(x), 5) for x in dev.mean(0)],
                             最大_波ごと=[round(float(x), 5) for x in dev.max(0)], 全体の平均=round(float(dev.mean()), 5), 全体の最大=round(float(dev.max()), 5))
    return out, lines


def main():
    import spacy
    import i49c_read_words as Rw
    import i49d_odd as OD
    import i49d_pickup as PU
    import i49e_read_together as RT
    import i49e_three_run as TR
    ap = argparse.ArgumentParser()
    for k in ("src", "meta", "wave", "scorer", "pickup", "warm", "three_log", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    live0 = TR.warm_live(nlp, a.warm.split(","), a.src, a.meta, PU.PickupTable.load(a.pickup))
    wv = Rw.WordVecs.load(a.wave)
    scorer = OD.OddScorer.load(a.scorer)
    rows = [json.loads(l) for l in open(a.three_log, encoding="utf-8")]
    rows = [r for r in rows if "ページ" in r][: a.limit]
    pages = {rk: RT.pages_of(RT.pick_book(a.src, a.meta, rank=rk)[1]) for rk in sorted({r["作品"] for r in rows})}
    print("用意", round(time.time() - t0), flush=True)
    res = dict(ページ=len(rows))
    got = {}
    for name, flag in (("なし", False), ("あり", True)):
        res[name], got[name] = run_one(live0, rows, pages, wv, scorer, nlp, flag)
        print(name, round(time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    both = [(x, y) for x, y in zip(got["なし"], got["あり"]) if x or y]
    res["話した行が同じだった割合"] = round(float(np.mean([x == y for x, y in both])), 4) if both else None
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
