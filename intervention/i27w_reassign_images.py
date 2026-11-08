"""語板の帯 S（像の偏り）で台詞の像を当て直し、I27t の話し手の推定から決めた像と合う割合を見る（足切りなし / あり）。

像の重みは i27w_extra_tables.py の --quote_images。
漏れを避けるため、作品の半分 A の台詞は半分 B の表で、半分 B の台詞は半分 A の表で当てる（比べに全部の表も出す）。
台詞の像は語尾からも作られているので、語尾の語が効く分は循環がある。語尾を除いた内容語（名詞・動詞・形容詞・副詞）だけの場合も出す。
結果は非公開の置き場だけ。本文は書かない。
実行: python intervention/i27w_reassign_images.py --quotes <quotes_words_wn.jsonl> --images <npz> --prior <json> --table <tsv> --half_a <tsv> --half_b <tsv> --works <tsv> --out <json>
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def read(path):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for x in r:
            if len(x) >= 12:
                try:
                    out[(x[0], x[1])] = (float(x[2]), float(x[3]) if x[3] else float("nan"), np.array([float(v) for v in x[4:12]]))
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    for n in ("quotes", "images", "prior", "table", "half_a", "half_b", "works", "out"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--breath", type=int, default=6, help="息づかいの像の番号（1 始まり）。帯 S から除く。0 なら 8 つの像をすべて使う")
    ap.add_argument("--min_count", type=float, default=20)
    ap.add_argument("--min_stable", type=float, default=0.5)
    ap.add_argument("--min_works", type=int, default=20)
    a = ap.parse_args()
    STYLE = [k for k in range(8) if k != a.breath - 1]
    q = np.array(json.load(open(a.prior, encoding="utf-8"))["prior"])
    logq = np.log(q[STYLE] / q[STYLE].sum())
    Z = np.load(a.images)
    work, quote, D, half = Z["work"], Z["quote"], Z["d"], Z["half"]
    gold = np.array(STYLE)[np.argmax(D[:, STYLE], axis=1)]
    full = read(a.table)
    ha, hb = read(a.half_a), read(a.half_b)
    works = defaultdict(int)
    with open(a.works, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            works[(r["pos"], r["lemma"])] = int(r["n_works"])
    keep = {k for k, (c, st, _) in full.items() if c >= a.min_count and st >= a.min_stable and
            (k[0] not in ("NOUN", "PROPN") or works[k] >= a.min_works)}

    def logs(tab, ks=None):
        """語 → 7 つの像での log 割合（prior を掛けて割合に戻す）。"""
        out = {}
        for k, (_, _, b) in tab.items():
            if ks is not None and k not in ks:
                continue
            s = np.exp(b[STYLE] + logq)
            out[k] = np.log(s / s.sum())
        return out
    tabs = {"足切りなし": (logs(hb), logs(ha), logs(full)), "足切りあり": (logs(hb, keep), logs(ha, keep), logs(full, keep))}

    # 台詞の語を読む（作品は i27w と同じく work_id の順）
    rows = []
    with open(a.quotes, encoding="utf-8") as f:
        for x in f:
            r = json.loads(x)
            rows.append((r["work_id"], [q_.get("w") for q_ in r["quotes"]]))
    rows.sort(key=lambda t: t[0])
    res = {}
    for scope in ("すべての語", "内容語だけ"):
        for name, (ta, tb, tf) in tabs.items():
            # ta = 半分 B の表（半分 A の作品に使う）、tb = 半分 A の表（半分 B の作品に使う）
            c = dict(半分の表=[0, 0, 0], 全部の表=[0, 0, 0], 文字どおり=[0, 0, 0])   # 合う、数、（空き）
            total = 0
            conf = np.zeros((len(STYLE), len(STYLE)), int)
            pos = {v: i for i, v in enumerate(STYLE)}
            for wi, qi, g in zip(work, quote, gold):
                w = rows[wi][1][qi]
                total += 1
                if not w:
                    continue
                h = int(half[wi])
                t = tb if h == 1 else ta
                # 下の 3 通りを同じ台詞で数える
                sc_h, sc_f = np.zeros(len(STYLE)), np.zeros(len(STYLE))
                n_h = n_f = 0
                for p, lemma, v in w:
                    if scope == "内容語だけ" and p not in CONTENT:
                        continue
                    k = (p, lemma)
                    if k in t:
                        sc_h += v * t[k]
                        n_h += v
                    if k in tf:
                        sc_f += v * tf[k]
                        n_f += v
                if n_h:
                    lit = int(np.argmax(sc_h))
                    bayes = int(np.argmax(sc_h - (n_h - 1) * logq))    # 語ごとの比の和 + 像の割合 1 回
                    c["文字どおり"][0] += STYLE[lit] == g
                    c["文字どおり"][1] += 1
                    c["半分の表"][0] += STYLE[bayes] == g
                    c["半分の表"][1] += 1
                    conf[pos[g], bayes] += 1
                if n_f:
                    bayes = int(np.argmax(sc_f - (n_f - 1) * logq))
                    c["全部の表"][0] += STYLE[bayes] == g
                    c["全部の表"][1] += 1
            res[f"{scope}_{name}"] = {
                "台詞": total,
                "半分の表で当てた（比の和）": dict(当たる語のある台詞=c["半分の表"][1], 合う=round(c["半分の表"][0] / max(1, c["半分の表"][1]), 4)),
                "半分の表で当てた（割合の対数の和、文字どおり）": dict(当たる語のある台詞=c["文字どおり"][1], 合う=round(c["文字どおり"][0] / max(1, c["文字どおり"][1]), 4)),
                "全部の表で当てた（漏れあり、比べ）": dict(当たる語のある台詞=c["全部の表"][1], 合う=round(c["全部の表"][0] / max(1, c["全部の表"][1]), 4)),
                "像ごとの再現率": {f"像{STYLE[i]+1}": round(float(conf[i, i] / max(1, conf[i].sum())), 3) for i in range(len(STYLE))},
            }
            print(scope, name, json.dumps(res[f"{scope}_{name}"], ensure_ascii=False), flush=True)
    base = {f"像{v+1}": round(float((gold == v).mean()), 4) for v in STYLE}
    res["いちばん多い像に全部当てたとき"] = max(base.values())
    res["像の分け方（台詞ごとの最大）"] = base
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
