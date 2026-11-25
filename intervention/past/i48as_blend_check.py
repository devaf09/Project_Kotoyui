"""I48as の確かめ: 機械の単位 1 万の元の文（学びに使っていない Tatoeba 2,046 文）を、単位として別に持つ代わりに本体の学びに普通に足した時（5 万 → 5 万 ＋ 2,046）の数字。
python intervention/past/i48as_blend_check.py out.json <conllu> <content_boards.json> <func_boards.json>
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
import json, os, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402


def main():
    out, conllu, content, func = sys.argv[1:5]
    t0 = time.time()
    sents = [s for s in G.read_conllu(conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][:1000]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][:50000]
    used, n = [], 0
    for s in tr_all[:-2000][50000:]:
        if n >= 10002:
            break
        cont, gaps, _ = G.split(s)
        if cont:
            used.append(s)
            n += len(gaps)
    tr2 = tr + used
    S = G.AttrSpace(json.load(open(content, encoding="utf-8")), G.Func(json.load(open(func, encoding="utf-8"))), 8192, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr2:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    m = AK.Model(S, fill, "架空・前").fit(tr2, hold)
    r = AO.measure(m, te, surf)
    r.update(学びの文=len(tr2), 足した文=len(used), 秒=round(time.time() - t0))
    json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
