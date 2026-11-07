"""介入の研究 実験I29c: 手順と記憶の両方が誤っているとき、どちらが悪いかを判断の板が決める
I29 では計算の手順（つなぎの板）は正しく、演算の帯の記憶だけが誤っていた。ここでは両方を誤らせる。
  - 手順の誤り: 3 演算とも、答えの十の位を「桁上りを足す前の照射の出口」から読む（十 = 2.一。桁上りを十の位に足すのを忘れた手順）。
  - 記憶の誤り: 0〜9 の板の演算の帯の文 300 個のうち割合 q を誤らせる（I29 と同じ）。
  - 判断の板: 答えが違ったら、2 種類の介入を試す。
      (a) 記憶: 照らした帯の組を 1 つずつ、結果だけを別の値に替えて解き直す（I29）。
      (b) 手順: その演算の層のつなぎの板の行を 1 つずつ、画素を別の列に替えて解き直す（I28 の一手替え）。
    答えが合う替え方を候補にし、候補が k 個ならそれぞれに 1 / k 票。票が V たまり、その行（または組）でいちばん多ければ替える。
    替える前に、答えが合った最近の式（演算ごとに 40 個）を解き直し、1 つでも壊れたら替えない（I29b の確かめ）。
  - 比べ: 記憶だけを直す（手順は直せない）/ 手順だけを直す / 両方。
  - 手順の誤りは多くの式に出るので票が早くたまり、記憶の誤りはその組を照らした式にしか出ない。この違いで「どちらが悪いか」が決まるかを見る。
物差し: 0〜99 の組の正解率、手順が直ったか、誤った文の残り、正しい文を誤って替えた数。
結果: ../Project_Kotoyui_private/intervention/results/i29c.jsonl
実行: python intervention/i29c_which_is_wrong.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i19_grain_arith as G  # noqa: E402
from i21_board_alu import OpBands  # noqa: E402
from i22_carry import digit_fact  # noqa: E402
from i28_route_board import N_STEP, OPS, OUT_ROW, OUTS, PAIRS, PORTS, ROWS, hand_choice, order_of  # noqa: E402
from i29_fix_memory import FACTS, alternatives, lookup, wrong_result  # noqa: E402


def solve(ch, A, B, tab):
    order = order_of(ch)
    if order is None:
        return -1, []
    p = [A % 10, A // 10, B % 10, B // 10] + [0] * (2 * N_STEP)
    used = []
    for k in order:
        l, r, o = ch[3 * k], ch[3 * k + 1], ch[3 * k + 2]
        key = (p[l], OPS[o], p[r])
        used.append(key)
        p[4 + 2 * k], p[5 + 2 * k] = tab[key]
    h, t, d = (p[ch[OUT_ROW[x]]] for x in OUTS)
    return h * 100 + t * 10 + d, used


def accuracy(chs, tab):
    return {op: sum(solve(chs[op], x, y, tab)[0] == G.FN[op](x, y) for x, y in PAIRS[op]) / len(PAIRS[op]) for op in OPS}


def run(seed, q, method, n_ex, V):
    rng = random.Random(seed * 101 + int(q * 1000))
    bad = set(rng.sample(FACTS, int(round(q * len(FACTS)))))
    wrong = {k: wrong_result(k, rng) for k in sorted(bad, key=str)}
    bands = OpBands()
    for key in FACTS:
        a, op, b = key
        bands.teach(a, op, b, wrong[key] if key in bad else digit_fact(a, op, b))
    tab = {key: lookup(bands, key) for key in FACTS}
    chs = {}
    for op in OPS:
        c = hand_choice(op)
        c[OUT_ROW["十"]] = PORTS.index("2.一")                      # 手順の誤り: 桁上りを十の位に足すのを忘れた
        chs[op] = c
    rec = dict(q=q, method=method, votes=V, n_bad=len(bad), acc_before=accuracy(chs, tab))
    ex_rng = random.Random(seed * 7 + 3)
    votes = defaultdict(Counter)
    memory = defaultdict(list)
    st = Counter()
    for n in range(1, n_ex + 1):
        op = ex_rng.choice(OPS)
        x, y = ex_rng.choice(PAIRS[op])
        target = G.FN[op](x, y)
        ch = chs[op]
        got, used = solve(ch, x, y, tab)
        st["solves"] += 1
        if got == target:
            memory[op].append((x, y, target))
            del memory[op][:-40]
            continue
        cands = []
        if method in ("記憶だけ", "両方"):
            for key in dict.fromkeys(used):
                keep = tab[key]
                for r in alternatives(key):
                    if r == keep:
                        continue
                    tab[key] = r
                    st["solves"] += 1
                    if solve(ch, x, y, tab)[0] == target:
                        cands.append(("記憶", key, r))
                tab[key] = keep
        if method in ("手順だけ", "両方"):
            for i, (_, ncol) in enumerate(ROWS):
                for j in range(ncol):
                    if j == ch[i]:
                        continue
                    c2 = ch[:]
                    c2[i] = j
                    st["solves"] += 1
                    if solve(c2, x, y, tab)[0] == target:
                        cands.append(("手順", (op, i), j))
        if not cands:
            st["見送り"] += 1
            continue
        for kind, key, r in cands:
            w = 1.0 / len(cands)
            if kind == "記憶" and key[1] in ("+", "*") and tab[(key[2], key[1], key[0])] == r:
                w += 1.0                                           # 入れ替えた組が同じ結果を言っている
            votes[(kind, key)][r] += w
        for kind, key, r in cands:
            vk = votes[(kind, key)]
            if not vk:
                continue
            (best, v), = vk.most_common(1)
            if best != r or v < V:
                continue
            # 確かめ: 替えたあとで、答えが合った最近の式が壊れないか
            if kind == "記憶":
                keep = tab[key]
                tab[key] = r
                ok = all(solve(chs[o2], x2, y2, tab)[0] == t2 for o2 in OPS for x2, y2, t2 in memory[o2])
                tab[key] = keep
            else:
                c2 = chs[key[0]][:]
                c2[key[1]] = r
                ok = all(solve(c2, x2, y2, tab)[0] == t2 for x2, y2, t2 in memory[key[0]])
            st["solves"] += sum(len(m) for m in memory.values())
            if not ok:
                st["取り消し"] += 1
                vk[r] = 0.0
                continue
            if kind == "記憶":
                a, o, b = key
                cur = sum(bands.band[a][("左", o, b)].values()) + sum(bands.band[b][("右", o, a)].values())
                bands.teach(a, o, b, r, n=cur + 1)
                tab[key] = lookup(bands, key)
                st["記憶を替えた"] += 1
                st["記憶: 正しく直した" if r == digit_fact(*key) else "記憶: 誤って替えた"] += 1
            else:
                chs[key[0]][key[1]] = r
                st["手順を替えた"] += 1
                st["手順の替え: " + f"{key[0]} {ROWS[key[1]][0]}→{(PORTS + OPS)[r] if ROWS[key[1]][1] == len(PORTS) else OPS[r]}"] += 1
            vk.clear()
            break                                                  # 1 つの式で替えるのは 1 つだけ
    wrong_now = {k for k in FACTS if tab[k] != digit_fact(*k)}
    rec.update(acc_after=accuracy(chs, tab), still_wrong_bad=len(wrong_now & bad), newly_wrong=len(wrong_now - bad),
               procedure_fixed={op: chs[op][OUT_ROW["十"]] == hand_choice(op)[OUT_ROW["十"]] for op in OPS}, **dict(st))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_ex", type=int, default=3000)
    ap.add_argument("--votes", type=float, default=2.0)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i29c.jsonl")
    a = ap.parse_args()
    t0 = time.perf_counter()
    out = dict(exp="I29c", seed=a.seed, n_ex=a.n_ex, runs=[])
    for q in (0.0, 0.1, 0.2):
        for method in ("記憶だけ", "手順だけ", "両方"):
            out["runs"].append(run(a.seed, q, method, a.n_ex, a.votes))
    out["total_s"] = time.perf_counter() - t0
    print(json.dumps(out, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
