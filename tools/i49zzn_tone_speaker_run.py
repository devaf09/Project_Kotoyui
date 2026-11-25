"""I49zzn: 口調（文の終わり）から、話し手と、相手との関係を読めるかを数える（持ち主の案、未確定の試し）。

持ち主の見方:「丁寧さは会話相手が丁寧語じゃないからじゃないかな？相手との関係でかわるから？」「逆説的に口調から関係を特定できる可能性があるってことだね。」
材料: 話し手の印の有る会話（1 行 1 発言の jsonl。列は dialogue_id, turn, speaker, text, addressee, meta.type, meta.relationship）。
  MRMP（名大、CC BY-SA 4.0）は 3 人の雑談で、半分は「家族 2 人 + 初対面 1 人」。だれ宛ての発言かの印が 2 割に有る。材料はリポジトリに入れない。
数える物（どれも文の終わりの切り出しだけを見る。分ける道具は使わない）:
  1. 続き方: 何個か前の発言と、話し手が同じ時・違う時で、3 つの印（丁寧・過去・問い）が続く倍率（I49zzm の「偶数個前は自分」を、本当の話し手で確かめる）。
  2. 相手で変わるか: 家族入りの会話で、同じ人が、家族宛ての時と初対面の人宛ての時で、丁寧の割合がどう違うか。
  3. 口調から話し手を当てる: 会話ごとに、前半の発言で話し手ごとの終わりの波を作り、後半の発言の話し手を 3 人から当てる。
  4. 口調から関係を当てる: 家族入りの会話の、宛先の有る発言で、宛先が家族か初対面かを終わりの波から当てる。会話を並べ替えた半分で学び、残りで試す。
手で置いた物: 3 つの印の決め方（I49zzm と同じ。読むためだけ）、前半と後半の分け方、平均を引く事、数える切り出しの下限 MIN。
材料の文は出さない。出すのは数だけ。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49zzl_tense_text_run as ZL
import i49zzm_prev_tail_run as ZM

MIN = 20   # 4 で数える切り出しの、出た回数の下限（手で置いた値）
KEYS = ("丁寧", "過去", "問い")


def tails(text):
    return [ZL.tail_of(s) for s in ZL.sentences(text.splitlines() or [text])]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    talks = {}
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ts = tails(r["text"])
            if ts:
                talks.setdefault(r["dialogue_id"], []).append(dict(who=r["speaker"], tails=ts, to=r.get("addressee") or [],
                                                                  kind=(r.get("meta") or {}).get("type"), fam=(r.get("meta") or {}).get("relationship") or []))
    return list(talks.values())


def any_mark(ts, k):
    return any(ZM.marks(t)[k] for t in ts)


def carry_by_speaker(talks, max_lag=4):
    out = {}
    for lag in range(1, max_lag + 1):
        row = {}
        for name, same in (("同じ話し手", True), ("違う話し手", False)):
            ps = [(u[i - lag]["tails"][-1], u[i]["tails"][0]) for u in talks for i in range(lag, len(u))
                  if (u[i - lag]["who"] == u[i]["who"]) is same]
            row[name] = dict(組=len(ps), **{k: v["倍"] for k, v in ZM.carry(ps).items()})
        out[str(lag)] = row
    return out


def by_addressee(talks):
    """家族入りの会話で、宛先が 1 人の発言。話し手が家族の側の時、宛先が家族か初対面かで丁寧の割合を比べる。"""
    per = {}
    for u in talks:
        for x in u:
            if x["kind"] != "Family" or len(x["to"]) != 1 or x["who"] not in x["fam"]:
                continue
            side = "家族宛て" if x["to"][0] in x["fam"] else "初対面の人宛て"
            d = per.setdefault(x["who"], {"家族宛て": [0, 0], "初対面の人宛て": [0, 0]})
            d[side][0] += 1
            d[side][1] += int(any_mark(x["tails"], "丁寧"))
    tot = {"家族宛て": [0, 0], "初対面の人宛て": [0, 0]}
    lower = both = 0
    for d in per.values():
        for k in tot:
            tot[k][0] += d[k][0]
            tot[k][1] += d[k][1]
        if d["家族宛て"][0] >= 10 and d["初対面の人宛て"][0] >= 10:
            both += 1
            lower += int(d["家族宛て"][1] / d["家族宛て"][0] < d["初対面の人宛て"][1] / d["初対面の人宛て"][0])
    res = {k: dict(発言=v[0], 丁寧の割合=round(v[1] / max(v[0], 1), 4)) for k, v in tot.items()}
    res["両方に 10 回以上話した人"] = both
    res["そのうち家族宛ての方が丁寧でない人"] = lower
    return res


def guess_speaker(talks):
    """会話ごとに、前半で話し手ごとの終わりの波（切り出しの割合 − 会話の平均）を作り、後半の発言の話し手を当てる。"""
    out = {}
    for kind in sorted({u[0]["kind"] or "?" for u in talks if u}):
        ok = base = n = 0.0
        for u in talks:
            if not u or (u[0]["kind"] or "?") != kind:
                continue
            half = len(u) // 2
            ix = {}
            prof = {}
            cnt = {}
            for x in u[:half]:
                cnt[x["who"]] = cnt.get(x["who"], 0) + 1
                p = prof.setdefault(x["who"], {})
                for t in x["tails"]:
                    for s in set(ZL.spans(t)):
                        j = ix.setdefault(s, len(ix))
                        p[j] = p.get(j, 0) + 1
            if len(prof) < 2:
                continue
            names = sorted(prof)
            m = np.zeros((len(names), len(ix)), dtype=np.float32)
            for a, w in enumerate(names):
                for j, c in prof[w].items():
                    m[a, j] = c
            m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
            m = m - m.mean(axis=0, keepdims=True)
            top = max(names, key=lambda w: (cnt[w], w))
            for x in u[half:]:
                if x["who"] not in prof:
                    continue
                v = np.zeros(len(ix), dtype=np.float32)
                for t in x["tails"]:
                    for s in set(ZL.spans(t)):
                        if s in ix:
                            v[ix[s]] += 1
                sc = m @ v
                best = [names[a] for a in range(len(names)) if sc[a] == sc.max()]
                n += 1
                ok += (1.0 / len(best)) if x["who"] in best else 0.0
                base += int(x["who"] == top)
        out[kind] = dict(問い=int(n), 当たり=round(ok / max(n, 1), 4), 一番多く話した人を言い続ける=round(base / max(n, 1), 4))
    return out


def guess_relation(talks):
    """宛先が家族か初対面かを、終わりの切り出しの波（家族宛ての割合 − 全体の割合）で当てる。"""
    fam = [u for u in talks if u and u[0]["kind"] == "Family"]
    order = ZL.golden_order(len(fam))
    half = len(order) // 2

    def items(us):
        out = []
        for u in us:
            for x in u:
                if len(x["to"]) == 1 and x["who"] in x["fam"]:
                    out.append((sorted({s for t in x["tails"] for s in ZL.spans(t)}), x["to"][0] in x["fam"]))
        return out

    train, held = items([fam[i] for i in order[:half]]), items([fam[i] for i in order[half:]])
    c = {}
    for sp, y in train:
        for s in sp:
            d = c.setdefault(s, [0, 0])
            d[0] += 1
            d[1] += int(y)
    p = sum(int(y) for _, y in train) / max(len(train), 1)
    w = {s: d[1] / d[0] - p for s, d in c.items() if d[0] >= MIN}
    ok = n = 0.0
    yes = sum(int(y) for _, y in held)
    for sp, y in held:
        sc = sum(w.get(s, 0.0) for s in sp)
        n += 1
        ok += 1.0 if (sc > 0) is y and sc != 0 else (.5 if sc == 0 else 0.0)
    return dict(学ぶ発言=len(train), 試す発言=len(held), 数えた切り出し=len(w), 当たり=round(ok / max(n, 1), 4),
                多い方を言い続ける=round(max(yes, len(held) - yes) / max(len(held), 1), 4), 家族宛ての割合=round(yes / max(len(held), 1), 4))


def run(talks):
    return dict(会話=len(talks), 発言=sum(len(u) for u in talks), 続き方=carry_by_speaker(talks), 相手で変わるか=by_addressee(talks),
                口調から話し手を当てる=guess_speaker(talks), 口調から関係を当てる=guess_relation(talks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = run(load(a.jsonl))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
