"""I49h: 下がる順を段ごとの当たり外れの数で決める形（intervention/i49h_tally.TallyRelation、中身は変えない）を、文の 3 つ組で比べる。数えだけ。

材料と数え方は tools/i49e_relation_run.py の「聞きながら育つ形」（stream）と同じ: 作った会話 600 本（覚える用 500 ＋ Gemma の試し用 100、この順）を
1 本ずつ読み、会話の中の 3 つ組ごとに 3 席それぞれを ask（候補 = その会話の内容語、見せた 2 語を除く、＋答え）→ 答え合わせ → learn。会話 1 本ごとに tick 1 回。
並べる物: Relation（今の順）／ TallyRelation 選ぶ ／ TallyRelation 合わせる。どれも薄れあり（half 1,024）。
始め方: 白紙から ／ 青空の 3 つ組を先に learn してから（この時、TallyRelation の当たり外れの数えは 0 から始まる。learn だけでは数えが付かない）。
数える物: 100 本ごとの 1 位の割合と白紙の割合、全体の 1 位の割合（席ごと）、答えた段の内訳。TallyRelation では終わりの時の段ごと・席ごとの当たり・外れと割合。
Relation と答えが変わった問いの数と、そのうち良くなった数・悪くなった数（と、その差が揺れの内かの目安 z = (良 − 悪) / √(良 + 悪)）。
記録はリポジトリの外。手で置く物: 100 本ごとの区切り、half 1,024、揺れの目安の式。
python tools/i49h_tally_run.py --src ... --meta ... --wave <波の板> --conllu <青空の conllu> --learn_talks <会話.jsonl>,... --stream_test <Gemma の試し用.jsonl> --out <stats.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

HALF = 1024.0


def stream_rec(rel, talks):
    """i49e_relation_run.stream と同じ流れで、問いごとの記録（会話の番, 席, 当たり, 段）を返す。"""
    recs = []
    for ti, (tris, words) in enumerate(talks):
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                q = tuple(None if j == si else tri[j] for j in range(3))
                o = rel.ask(si, q, cands)
                recs.append((ti, si, o["答え"] == tri[si], o["段"]))
            rel.learn(tri)
        rel.tick()
    return recs


def summary(recs, n_talks, every=100):
    from i49e_relation import SLOTS
    rows = []
    for end in list(range(every, n_talks + 1, every)) + ([n_talks] if n_talks % every else []):
        cur = [r for r in recs if end - every <= r[0] < end]
        n = max(1, len(cur))
        lv = Counter(r[3] for r in cur)
        rows.append({"会話": end, "問い": len(cur), "一位": round(sum(r[2] for r in cur) / n, 4),
                     "白紙": round(lv["答えられない"] / n, 4), "段": {k: round(v / n, 4) for k, v in lv.items()}})
    n = max(1, len(recs))
    seat = {}
    for si, s in enumerate(SLOTS):
        x = [r for r in recs if r[1] == si]
        seat[s] = dict(問い=len(x), 一位=round(sum(r[2] for r in x) / max(1, len(x)), 4))
    lv = Counter(r[3] for r in recs)
    return dict(流れ=rows, 全体=dict(問い=len(recs), 一位=round(sum(r[2] for r in recs) / n, 4), 白紙=round(lv["答えられない"] / n, 4)),
                席ごと=seat, 段={k: round(v / n, 4) for k, v in lv.items()})


def compare(base, other):
    """同じ問いの並びで、base（Relation）と答えの当たりが変わった数。"""
    assert len(base) == len(other)
    better = sum(1 for b, o in zip(base, other) if not b[2] and o[2])
    worse = sum(1 for b, o in zip(base, other) if b[2] and not o[2])
    moved = sum(1 for b, o in zip(base, other) if b[3] != o[3] or b[2] != o[2])
    return dict(当たりが変わった=better + worse, 良くなった=better, 悪くなった=worse, 段か当たりが変わった=moved,
                揺れの目安_z=round((better - worse) / math.sqrt(better + worse), 2) if better + worse else None)


def tally_table(rel):
    out = {}
    for seat, d in rel.tally_levels().items():
        out[seat] = {lv: dict(x, 割合=round(x["当たり"] / (x["当たり"] + x["外れ"]), 4) if x["当たり"] + x["外れ"] else None) for lv, x in d.items()}
    return out


def run_all(vec_of, talks, pre_tris, makers=None):
    """makers = ((名前, 作る関数), ...)。1 つ目は Relation（比べる元）。渡さなければ I49h の 3 つ。"""
    import i49e_relation as RL
    import i49h_tally as TY
    makers = makers or (("Relation", lambda: RL.Relation(vec_of, half=HALF)),
                        ("選ぶ", lambda: TY.TallyRelation(vec_of, half=HALF, mode="選ぶ")),
                        ("合わせる", lambda: TY.TallyRelation(vec_of, half=HALF, mode="合わせる")))
    res, recs = {}, {}
    for name, make in makers:
        rel = make()
        for tri in pre_tris:
            rel.learn(tri)
        t1 = time.time()
        recs[name] = stream_rec(rel, talks)
        res[name] = summary(recs[name], len(talks))
        res[name]["秒"] = round(time.time() - t1, 1)
        if hasattr(rel, "tally_levels"):
            res[name]["終わりの段ごとの当たり外れ"] = tally_table(rel)
        if hasattr(rel, "used"):
            res[name]["座り方の近さを使った回数"] = rel.used
        if name != makers[0][0]:
            res[name]["Relation との違い"] = compare(recs[makers[0][0]], recs[name])
    return res


def load(argv=None):
    """材料を読む。→ (引数, vec_of, 会話, 青空の 3 つ組, 始めの時刻)"""
    import spacy
    import i49c_read_words as Rw
    import i49e_relation_run as RR
    import i49e_triple_count as TC
    ap = argparse.ArgumentParser()
    for k in ("src", "meta", "wave", "conllu", "learn_talks", "stream_test", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--books", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--cap", type=int, default=2000)
    a = ap.parse_args(argv)
    t0 = time.time()
    nlp = spacy.load("ja_ginza")
    wv = Rw.WordVecs.load(a.wave)
    vec_of = lambda w: wv.of(*w)
    _, learn, _ = TC.prepare(a, nlp, wv, t0)
    rows = [json.loads(l) for f in a.learn_talks.split(",") for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r["語"]["選び方"] in ("い", "覚える側")]
    talks = RR.talk_words(nlp, rows) + RR.talk_words(nlp, [json.loads(l) for l in open(a.stream_test, encoding="utf-8")])
    aoz = [tri for w in sorted(learn) for tri in learn[w]]
    return a, vec_of, talks, aoz, t0


def main():
    a, vec_of, talks, aoz, t0 = load()
    res = dict(会話=len(talks), 青空の組=len(aoz), 注="青空を先に入れる時、TallyRelation の当たり外れの数えは 0 から始まる（learn だけでは数えが付かない）")
    print("用意", round(time.time() - t0), flush=True)
    for name, pre in (("白紙から", []), ("青空を先に", aoz)):
        res[name] = run_all(vec_of, talks, pre)
        print(name, round(time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
