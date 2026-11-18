"""I48at (四): 「正解が候補に無い」の扱い（持ち主の問い「無い、か、無いが有るとするとか？」）。

1. 数: 試しの間と学びの間のうち、正解が「その他」（候補の組の外）の割合。今は「その他」を選べないので、その間は必ず外れ。
   その分を除いた時の間の当たり。
2. 「無いが有る」の型（G.OTHER = True）: 「その他」を普通の候補として選べるようにする（本体の数えは前から持っている。伏せるのをやめるだけ）。
   介入の単位（型 C）でも「その他」を教えられる。文として出す時は「その他」の所を □ で空けたまま、文の当たりは今まで通り数える。
   測る物: 間・文末・文、「その他」の当たり（元が「その他」の間で「その他」を選んだ割合）と害（元が候補の中なのに「その他」にした数）、
   ほかの助詞の当たりと害、筋の無い共振。違和感の上位 N を（「その他」も含めて）介入の単位で教えた時の同じ数え。
3. 違和感との関係: 違和感の上位 1 割・2 割に、元が「その他」の間が何割入るか（全体の割合と比べる）。
   照らし直しの行（i48at_recheck.py --rows の記録）を使う。
手で置く物: N の並び。β は伏せた形で選んだ物をそのまま使う（伏せをやめた後に選び直さない）。
python intervention/i48at_other.py --conllu ... --content ... --func ... --rows rows.json --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48at_surprise as AW  # noqa: E402

NS = (100, 300, 1000)


def other_share(m, sents):
    """間（文末を除く）のうち、正解が候補の組の外の数。"""
    known = set(m.pg.cls[:-1])
    n = o = 0
    for s in sents:
        gaps = G.split(s)[1]
        for y in gaps[:-1]:
            n += 1
            o += y not in known
    return n, o


def hit_without_other(m, te):
    known = set(m.pg.cls[:-1])
    n = h = 0
    for s in te:
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        gaps = G.split(s)[1]
        rs, _ = m.walk(AK.keys_of(s), xs)
        for i, (r, y) in enumerate(zip(rs, gaps)):
            if i + 1 == len(gaps) or y not in known:
                continue
            n += 1
            h += m.pg.cls[r["選び"]] == y
    return round(h / max(1, n), 4), n


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out", "rows"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    ap.add_argument("--ns", nargs="*", type=int, default=list(NS))
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    G.OTHER = False
    m = AP.UModel(S, fill, a.r1).fit(tr, hold).init_units("C")
    m.refresh()
    res = dict(学びの文=len(tr), 試しの文=len(te))

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 1. 数
    nt, ot = other_share(m, te)
    nl, ol = other_share(m, tr)
    h, nk = hit_without_other(m, te)
    res["数"] = dict(候補の数=m.nc - 1, 試しの間=nt, 試しのその他=ot, 試しのその他の割合=round(ot / max(1, nt), 4),
                    学びの間=nl, 学びのその他=ol, 学びのその他の割合=round(ol / max(1, nl), 4),
                    その他を除いた間の当たり=h, その他を除いた間=nk)
    res["伏せる（今まで）"] = AP.measure(m, te, surf)
    res["伏せる（今まで）"]["筋の無い共振"] = AW.monitor(m, te)
    print(f"学び {time.time() - t0:.0f} 秒", json.dumps(res["数"], ensure_ascii=False), flush=True)
    print("伏せる", json.dumps(res["伏せる（今まで）"], ensure_ascii=False), flush=True)
    save()
    # 3. 違和感との関係（照らし直しの行は伏せた形で作った物）
    rows = json.load(open(a.rows, encoding="utf-8"))
    order = RC.oddness(rows)
    oth = np.array([r["gold"] == m.nc - 1 for r in rows])
    top = lambda q: order[: int(len(order) * q)]
    res["違和感とその他"] = dict(間=len(rows), その他の割合_全体=round(float(oth.mean()), 4),
                            上位1割のその他の割合=round(float(oth[top(.1)].mean()), 4), 上位2割のその他の割合=round(float(oth[top(.2)].mean()), 4),
                            その他のうち上位1割に入る割合=round(float(oth[top(.1)].sum() / max(1, oth.sum())), 4),
                            その他のうち上位2割に入る割合=round(float(oth[top(.2)].sum() / max(1, oth.sum())), 4),
                            上位Nのその他の割合={str(n): round(float(oth[order[:n]].mean()), 3) for n in (10, 100, 1000, 10000)})
    print("違和感とその他", json.dumps(res["違和感とその他"], ensure_ascii=False), flush=True)
    save()
    # 2. 無いが有る
    G.OTHER = True
    m.refresh()
    t1 = time.time()
    res["無いが有る"] = AP.measure(m, te, surf)
    res["無いが有る"]["筋の無い共振"] = AW.monitor(m, te)
    print("無いが有る", round(time.time() - t1), "秒", json.dumps(res["無いが有る"], ensure_ascii=False), flush=True)
    save()
    res["無いが有る・違和感の上位を教える"] = {}
    for n in a.ns:
        t1 = time.time()
        mm = m.clone()
        mm.kind = "C"
        for o in order[:n]:
            r = rows[o]
            s = tr[r["j"]]
            x = AE.items(S, s, "節")[r["i"]]
            mm.teach_unit(("違和感", f"上位 {n}", AK.keys_of(s), G.split(s)[1]), xs=[x])
        mm.refresh()
        res["無いが有る・違和感の上位を教える"][str(n)] = rr = AP.measure(mm, te, surf)
        rr["単位"] = len(mm.units)
        print("無いが有る・上位", n, round(time.time() - t1), "秒", json.dumps(rr, ensure_ascii=False), flush=True)
        save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
