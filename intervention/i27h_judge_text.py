"""介入の研究 実験I27h: 算術の介入（I29・I29b・I29c の判断の板）を、文章の係り先の判断に持ち込む

持ち主（2026-11-07）の順番 D（算術の介入の成果を入れる）→ B → C → A の D。
算術では、答えが違ったら「記憶（演算の帯の 1 つの組）」か「手順（つなぎの板の 1 画素）」を 1 つ替えて解き直し、
答えが合う替え方に票を入れ（候補が k 個なら 1 / k）、同じ替え方に V 票たまったら、最近合った式を壊さないか確かめてから替えた。
手順の誤りは多くの式に出るので票が早くたまり、記憶の誤りは遅くたまる。これで「どちらが悪いか」を決めた（I29c）。
文章では:
  - 問い: 曖昧な修飾 M の係り先（I27c・I27d の手の印 73 か所）。1 つずつ読み、答え（印）が返ってくる。
  - 記憶 = 話の板の 1 つの組（その話の、候補の名詞 + 「修飾:M」の数え）。話の中の別の版に同じ句があれば効く（個別の誤り）。
  - 手順 = 場面の形（M の種類・A が「A の X」で使われやすい語か・根拠の状態）ごとの読み方の決まり:
      板 → 近い方（I27c の既定）、板 → 遠い方、近い方、遠い方。
  - 誤ったら、(a) 記憶の替え（正しい候補に「修飾:M」を足す / 誤って選んだ候補の「修飾:M」を消す）と、
    (b) 手順の替え（その形の決まりを別のものに）を 1 つずつ試し、答えが合うものに 1 / k 票。V 票（既定 2）たまったら、
    最近合った問い（既定 20 個）を壊さないか確かめてから替える。1 つの問いで替えるのは 1 つだけ。
  - 比べ: 直さない / 記憶だけ / 手順だけ / 両方。読む順は 6 通り（seed）。
物差し: 読みながらの正解率（その時点の状態で答えた割合）、読み終えたあとの全部の正解率、
        話を分けた試し（8 話で学び、残り 4 話に学んだ手順を使う。3 通りの分け方 × 6 seed）。
結果: ../Project_Kotoyui_private/intervention/results/i27h.jsonl
実行: python intervention/i27h_judge_text.py --tales <tales_v1_parsed.json> --labels <i27c_labels.json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i27a_story_check import story_boards  # noqa: E402
from i27c_compress_units import LEVELS, candidates, modifiers, story_has, units  # noqa: E402
from i27d_attention_boards import clean_board, copy_board, kindness, mtype  # noqa: E402

RULES = ("板→近", "板→遠", "近", "遠")


def build_items(tales, labels, kind_th=0.5):
    kind = kindness(tales)
    items = []
    for ti, t in enumerate(tales):
        for li, lv in enumerate(LEVELS):
            longer = LEVELS[:li]
            full = story_boards(t, longer) if longer else None
            clean = clean_board(full, copy_board(t, longer)) if longer else None
            for si, s in enumerate(t["parsed"].get(lv, [])):
                toks = s["toks"]
                tok2u, us = units(toks)
                for ui, gu in modifiers(toks, tok2u, us):
                    cand, _ = candidates(toks, tok2u, us, ui)
                    if len(cand) < 2:
                        continue
                    pid = f"{ti}.{lv}.{si}.{us[ui]['head']}"
                    g = labels.get(pid)
                    if g is None or g == "対象外":
                        continue
                    mlem = toks[us[ui]["head"]]["lemma"]
                    ev = [story_has(full, toks, us, u, mlem) if full else 0 for u in cand]
                    evc = [story_has(clean, toks, us, u, mlem) if clean else 0 for u in cand]
                    one = [k for k, v in enumerate(ev) if v > 0]
                    onec = [k for k, v in enumerate(evc) if v > 0]
                    state = "きれい" if len(onec) == 1 else ("写し" if len(one) == 1 else "なし")
                    A = us[cand[0]]["text"]
                    kr, kn = kind.get(A, (0.0, 0))
                    items.append(dict(id=pid, tale=ti, level=lv, gold=g, cand=[us[u]["text"] for u in cand], m=mlem, ev=ev,
                                      shape=(mtype(toks[us[ui]["head"]]), kr >= kind_th and kn >= 2, state)))
    return items


class Judge:
    def __init__(self, mem=True, proc=True, V=2.0, check=20):
        self.mem, self.proc, self.V, self.check = mem, proc, V, check
        self.delta = Counter()                     # (話, 候補, M) → 足した数え（記憶の替え）
        self.rule = {}                             # 形 → 決まり（無ければ 板→近）
        self.votes = Counter()
        self.recent = []
        self.changes = Counter()

    def board_pick(self, x, delta=None, rule=None):
        d = self.delta if delta is None else delta
        ev = [v + d[(x["tale"], c, x["m"])] for v, c in zip(x["ev"], x["cand"])]
        pos = [k for k, v in enumerate(ev) if v > 0]
        return pos[0] if len(pos) == 1 else None

    def read(self, x, delta=None, rule=None):
        r = (self.rule if rule is None else rule).get(x["shape"], "板→近")
        near, far = 0, len(x["cand"]) - 1
        if r == "近":
            return x["cand"][near]
        if r == "遠":
            return x["cand"][far]
        b = self.board_pick(x, delta)
        if b is not None:
            return x["cand"][b]
        return x["cand"][near if r == "板→近" else far]

    @staticmethod
    def right(x, r):
        return x["gold"] == "両方" or r == x["gold"]

    def step(self, x):
        r = self.read(x)
        ok = self.right(x, r)
        if ok:
            self.recent = (self.recent + [x])[-self.check:]
            return ok
        cands = []
        if self.mem:
            g = x["gold"]
            cands.append(("記憶", (x["tale"], g, x["m"]), +2))         # 正しい候補に「修飾:M」を足す
            cands.append(("記憶", (x["tale"], r, x["m"]), -99))        # 誤って選んだ候補の「修飾:M」を消す
        if self.proc:
            for rr in RULES:
                if rr != self.rule.get(x["shape"], "板→近"):
                    cands.append(("手順", x["shape"], rr))
        good = [c for c in cands if self.right(x, self.try_read(x, c))]
        for c in good:
            self.votes[c] += 1.0 / len(good)
        best = max(good, key=lambda c: self.votes[c], default=None)
        if best is not None and self.votes[best] >= self.V and self.safe(best):
            self.apply(best)
            self.changes[best[0]] += 1
            for c in [c for c in self.votes if c[0] == best[0] and c[1] == best[1]]:
                del self.votes[c]
        return ok

    def try_read(self, x, c):
        if c[0] == "記憶":
            d = Counter(self.delta)
            d[c[1]] += c[2]
            return self.read(x, delta=d)
        rule = dict(self.rule)
        rule[c[1]] = c[2]
        return self.read(x, rule=rule)

    def safe(self, c):
        return all(self.right(y, self.try_read(y, c)) for y in self.recent)

    def apply(self, c):
        if c[0] == "記憶":
            self.delta[c[1]] += c[2]
        else:
            self.rule[c[1]] = c[2]


def run(items, mem, proc, seed, V=2.0, check=20):
    rng = random.Random(seed)
    order = items[:]
    rng.shuffle(order)
    J = Judge(mem, proc, V, check)
    online = sum(J.step(x) for x in order) / len(order)
    final = sum(J.right(x, J.read(x)) for x in items) / len(items)
    return J, online, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--V", type=float, default=2.0)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i27h.jsonl")
    a = ap.parse_args()
    tales = json.load(open(a.tales, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8"))
    items = build_items(tales, labels)
    short = [x for x in items if x["level"] != "20"]
    seeds = range(6)
    conds = {"直さない": (False, False), "記憶だけ": (True, False), "手順だけ": (False, True), "両方": (True, True)}
    rec = dict(exp="I27h", n=len(items), n_short=len(short), V=a.V, base={}, cv={})
    J0 = Judge(False, False)
    rec["base"]["all"] = sum(J0.right(x, J0.read(x)) for x in items) / len(items)
    rec["base"]["short"] = sum(J0.right(x, J0.read(x)) for x in short) / len(short)
    for name, (m, p) in conds.items():
        on, fin, fin_s, ch, rules = [], [], [], Counter(), Counter()
        for sd in seeds:
            J, o, f = run(items, m, p, sd, a.V)
            on.append(o)
            fin.append(f)
            fin_s.append(sum(J.right(x, J.read(x)) for x in short) / len(short))
            ch.update(J.changes)
            rules.update(f"{'/'.join(map(str, k))}→{v}" for k, v in J.rule.items())
        rec[name] = dict(online=sum(on) / 6, final=sum(fin) / 6, final_short=sum(fin_s) / 6,
                         final_min=min(fin), final_max=max(fin), changes={k: v / 6 for k, v in ch.items()},
                         rules=dict(rules.most_common(8)))
    # 話を分けた試し: 学んだ手順だけを、学んでいない話に使う（記憶は話ごとなので持ち越さない）
    tales_ids = sorted({x["tale"] for x in items})
    folds = [tales_ids[k::3] for k in range(3)]
    for name, (m, p) in conds.items():
        acc = []
        for f in folds:
            test = [x for x in items if x["tale"] in f]
            train = [x for x in items if x["tale"] not in f]
            for sd in seeds:
                J, _, _ = run(train, m, p, sd, a.V)
                J.delta = Counter()
                acc.append(sum(J.right(x, J.read(x)) for x in test) / len(test))
        rec["cv"][name] = sum(acc) / len(acc)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
