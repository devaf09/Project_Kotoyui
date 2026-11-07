"""介入の研究 実験I27j: 文章の判断を、経験の器に 1 回通す光の判断に置き換える（算術スレッドの I8・I8b・I9・I12 の移植）

持ち主（2026-11-07）: 「算術のスレッドから、可能ならデジタル重み（判断類）を減らせる動作も参考にして移植してほしい」。
算術・小さな世界で判断の重みを減らした動き:
  I8  判断（規則の数え上げ・比べの連なり）を、場面の記号を経験の器（光の器、口 = 結論）に 1 回通すだけにする。
  I8b 口ごとに 1 種類の結論だけを学ばせてから重ねる（混ざらない）。
  I9  経験の結論を、規則ではなく「答えが合ったか」で付ける。
  I12 結論を 1 つに選ばず、結論の明るさの割合をそのまま次の器へ渡す（選ぶ処理を 1 つ減らす）。
文章への移植:
  T. 童話の曖昧な係り先（I27h と同じ 73 か所の手の印）。I27h の判断の板（形ごとの決まりの表 + 票 + V + 最近の問いの確かめ）の代わりに、
     場面の記号（M の種類・A が「A の X」で使われやすいか・話の板の根拠の状態・話の板が指す方・候補の数）と、
     合った読み（結論: 近 / 遠 / 中）を経験として器に学ばせ、問いごとに場面を 1 回通して、いちばん明るい読みで答える。
     話を分けた試し（I27h と同じ 3 通り × 6 seed）。
  W. Web 小説の補い（I27g の 7 通りの補い方、I27h_wn と同じ問い）。経験 = 場面の形 + 1 位で当てた補い方（I9 の答えが合ったか）。
     問いごとに場面を 1 回通し、補い方の明るさの割合で、補い方ごとの点（1 位で割ったもの）を重ねて答える（I12。補い方を選ばない）。
     作品を 1 つ抜いて学び、抜いた作品で試す。
  どちらも、判断 1 回 = 器を 1 回通す数（光の判断）と、I27h の判断の板が解き直した数（デジタル）を数えて並べる。
結果: --out（Web 小説は _wn を付け、非公開の置き場だけに置く）。
実行: python intervention/i27j_light_judge_text.py --tales <tales_v1_parsed.json> --labels <i27c_labels.json> --out <jsonl>
      python intervention/i27j_light_judge_text.py --wn_policies <I27g の --dump_policies> --out <..._wn.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i08_light_judge import Memory  # noqa: E402
from i27h_judge_text import Judge, build_items, run  # noqa: E402


def scene_t(x):
    mt, kind, state = x["shape"]
    pos = [k for k, v in enumerate(x["ev"]) if v > 0]
    n = len(x["cand"])
    points = "なし" if len(pos) != 1 else ("近" if pos[0] == 0 else ("遠" if pos[0] == n - 1 else "中"))
    return [f"種類:{mt}", f"の型:{'よく使う' if kind else 'あまり'}", f"根拠:{state}", f"板が指す:{points}", f"候補:{'2' if n == 2 else '3+'}"]


def concl_t(x):
    if x["gold"] == "両方":
        return None
    k = x["cand"].index(x["gold"]) if x["gold"] in x["cand"] else None
    if k is None:
        return None
    return "近" if k == 0 else ("遠" if k == len(x["cand"]) - 1 else "中")


def read_t(mem, x):
    b = mem.recall(scene_t(x), ("近", "遠", "中"))
    n = len(x["cand"])
    if n == 2:
        b["中"] = 0.0
    r = max(b, key=b.get)
    return x["cand"][0 if r == "近" else (n - 1 if r == "遠" else 1)]


def count_digital(J_items, seed):
    """I27h の判断の板（手順だけ）が、学びの間に解き直した数（候補の試し + 確かめ）。"""
    J = Judge(False, True)
    n = 0
    orig_try = J.try_read

    def counted(x, c):
        nonlocal n
        n += 1
        return orig_try(x, c)
    J.try_read = counted
    rng = random.Random(seed)
    order = J_items[:]
    rng.shuffle(order)
    for x in order:
        J.step(x)
    return n


def run_tales(a):
    tales = json.load(open(a.tales, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8"))
    items = build_items(tales, labels)
    tids = sorted({x["tale"] for x in items})
    folds = [tids[k::3] for k in range(3)]
    rec = dict(exp="I27j", part="T", n=len(items), cv={})
    acc_light, acc_judge, acc_base, n_dig = [], [], [], []
    ports = None
    for f in folds:
        test = [x for x in items if x["tale"] in f]
        train = [x for x in items if x["tale"] not in f]
        eps = [scene_t(x) + [f"結論:{concl_t(x)}"] for x in train if concl_t(x)]
        for sd in range(6):
            mem = Memory(eps, sd, how="labeled")
            acc_light.append(sum(Judge.right(x, read_t(mem, x)) for x in test) / len(test))
            J, _, _ = run(train, False, True, sd, a.V)
            J.delta = Counter()
            acc_judge.append(sum(J.right(x, J.read(x)) for x in test) / len(test))
            acc_base.append(sum(Judge.right(x, Judge(False, False).read(x)) for x in test) / len(test))
            n_dig.append(count_digital(train, sd))
            if ports is None:
                ports = mem.ports_summary()
    # 全部の経験で学んだ器（話を分けない、見たことのある問いでの正解率）
    eps = [scene_t(x) + [f"結論:{concl_t(x)}"] for x in items if concl_t(x)]
    mem = Memory(eps, 0, how="labeled")
    rec["in_sample"] = sum(Judge.right(x, read_t(mem, x)) for x in items) / len(items)
    rec["cv"] = dict(直さない=sum(acc_base) / len(acc_base), 判断の板=sum(acc_judge) / len(acc_judge), 光の判断=sum(acc_light) / len(acc_light),
                     光の判断_min=min(acc_light), 光の判断_max=max(acc_light))
    rec["cost"] = dict(光の判断_試しの問いあたり器を通す数=1, 判断の板_学びの間の解き直し=sum(n_dig) / len(n_dig),
                       判断の板_学びの問い数=sum(len([x for x in items if x["tale"] not in f]) for f in folds) / 3)
    rec["ports"] = mem.ports_summary()
    rec["n_episodes"] = len(eps)
    return rec


SCORE_K = 10


def run_wn(a):
    P = json.load(open(a.wn_policies, encoding="utf-8"))
    lists = P["lists"]
    pols = list(lists)
    T = []
    for i, (wi, li, nm, kind, ans, dep, is_char) in enumerate(P["targets"]):
        has = bool(lists["属性だけ"][i])
        sc = [f"種類:{kind}", f"係り:{dep}", f"束ね:{'あり' if has else 'なし'}", f"人物:{'はい' if is_char else 'いいえ'}"]
        right = [p for p in pols if lists[p][i] and lists[p][i][0][0] == ans]
        T.append(dict(i=i, wi=wi, ans=ans, scene=sc, right=right))
    works = sorted({t["wi"] for t in T})

    def pid(p):
        return "P" + str(pols.index(p))                         # 器の記号は短い番号にする

    def answer(i, w):
        sc = Counter()
        for p in pols:
            lst = lists[p][i]
            if not lst or w.get(p, 0) <= 0:
                continue
            top = lst[0][1] or 1.0
            for c, v in lst[:SCORE_K]:
                sc[c] += w[p] * v / top
        return max(sc, key=lambda c: (sc[c], c)) if sc else None

    rec = dict(exp="I27j_wn", part="W", n=len(T), policies=pols, cv={})
    fixed = {p: sum(1 for t in T if p in t["right"]) / len(T) for p in pols}
    rec["fixed"] = fixed
    hits = Counter()
    passes = 0
    for w in works:
        train = [t for t in T if t["wi"] != w]
        test = [t for t in T if t["wi"] == w]
        eps = [t["scene"] + [f"結論:{pid(p)}"] for t in train for p in t["right"]]
        for sd in range(3):
            mem = Memory(eps, sd, how="labeled")
            for t in test:
                b = mem.recall(t["scene"], [pid(p) for p in pols])
                passes += 1
                wsoft = {p: b[pid(p)] for p in pols}
                hard = max(pols, key=lambda p: wsoft[p])
                hits["光の判断 割合で重ねる（I12）"] += answer(t["i"], wsoft) == t["ans"]
                hits["光の判断 いちばん明るい補い方を選ぶ（I8b）"] += bool(lists[hard][t["i"]]) and lists[hard][t["i"]][0][0] == t["ans"]
            hits["重ね 等しい重み"] += 0
        for t in test:
            hits["重ね 等しい重み"] += 3 * (answer(t["i"], {p: 1.0 for p in pols}) == t["ans"])
    rec["cv"] = {k: v / (3 * len(T)) for k, v in hits.items()}
    rec["cost"] = dict(器を通す数=passes / 3, 問い=len(T))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--wn_policies", default=None)
    ap.add_argument("--V", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = run_wn(a) if a.wn_policies else run_tales(a)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
