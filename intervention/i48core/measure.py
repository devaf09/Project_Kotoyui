"""測り: 間・文末・文・助詞ごとの当たりと害（measure）、と の例の順位（ranks_of・WANT）。
ほかの測り（と の人と人以外・筋の無い共振・想定外の反応・違和感）は、使う時に各ファイルから読む。

実装は i48ao_many_particles（measure・PARTICLES）と i48am_attr_weight（ranks_of・WANT）から移した物。古いファイルは同じ名前で見せる。
手で置く物: PARTICLES（数える助詞）、WANT（と の例: 広げたい 5・広げたくない 4、自作の要点）。
"""
from __future__ import annotations

import numpy as np

import i48ab_long_grain as G
import i48ae_marks as AE
import i48aj_resonance as AJ
import i48ak_teach as AK

PARTICLES = ("と", "で", "に", "を", "が", "へ", "から", "も")
N, V, P_ = "NOUN", "VERB", "PRON"
WANT = [("広げたい", "昨日・友達・散歩・行く", [("昨日", N), ("友達", N), ("散歩", N), ("行く", V)], 1),
        ("広げたい", "先生・学校・行く", [("先生", N), ("学校", N), ("行く", V)], 0),
        ("広げたい", "母・買い物・行く", [("母", N), ("買い物", N), ("行く", V)], 0),
        ("広げたい", "彼・映画・行く", [("彼", P_), ("映画", N), ("行く", V)], 0),
        ("広げたい", "友達・公園・遊ぶ", [("友達", N), ("公園", N), ("遊ぶ", V)], 0),
        ("広げたくない", "傘・帽子・買う の 2 つ目", [("傘", N), ("帽子", N), ("買う", V)], 1),
        ("広げたくない", "昨日・公園・犬・散歩・する の「で」の間", [("昨日", N), ("公園", N), ("犬", N), ("散歩", N), ("する", V)], 1),
        ("広げたくない", "本・机・置く の 1 つ目", [("本", N), ("机", N), ("置く", V)], 0),
        ("広げたくない", "本・机・置く の 2 つ目", [("本", N), ("机", N), ("置く", V)], 1)]


def ranks_of(m):
    out = {}
    wc = m.cid("と")
    for kind, name, keys, gi in WANT:
        res, _ = m.walk(keys, AK.made_items(m.S, keys, None))
        sc = res[gi]["今×三つ"]
        out[name] = dict(種類=kind, 順位=int(AJ.ranks(sc)[wc]) + 1, 点の差=round(float(sc[wc] - np.max(np.delete(sc, wc))), 3), 選び=m.pg.cls[res[gi]["選び"]])
    return out


def measure(m, te, surf):
    c = {}

    def inc(k, v=1):
        c[k] = c.get(k, 0) + v
    for s in te:
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        cont, gaps, head = G.split(s)
        rs, _ = m.walk(AK.keys_of(s), xs)
        seq = [m.pg.cls[r["選び"]] for r in rs]
        known = set(m.pg.cls[:-1])
        inc("文")
        for i, (g, y) in enumerate(zip(seq, gaps)):
            if i + 1 == len(gaps):
                inc("文末当たり", g == y)
                continue
            inc("間")
            inc("間当たり", g == y or (g is None and y not in known))
            if G.other_ok(m):
                if y not in known:
                    inc("その他n")
                    inc("その他当たり", g is None)
                elif g is None:
                    inc("その他害")
            for p in PARTICLES:
                if y == p:
                    inc(p + "n")
                    inc(p + "当たり", g == p)
                elif g == p:
                    inc(p + "害")
        sent = "".join(surf.form(l, u, g) + ("□" if g is None else g) for (l, f_, u), g in zip(cont, seq))
        inc("文当たり", G.norm(head + sent) == G.norm(s["text"]))
    d = lambda a, b: round(c.get(a, 0) / max(1, c.get(b, 0)), 4)
    return dict(間=d("間当たり", "間"), 文末=d("文末当たり", "文"), 文=d("文当たり", "文"),
                助詞={p: dict(数=c.get(p + "n", 0), 当たり=d(p + "当たり", p + "n"), 害=c.get(p + "害", 0)) for p in PARTICLES},
                例={k: v["順位"] for k, v in ranks_of(m).items()},
                **(dict(その他=dict(数=c.get("その他n", 0), 当たり=d("その他当たり", "その他n"), 害=c.get("その他害", 0))) if G.other_ok(m) else {}))


_LAZY = {"measure_to": ("i48ap_units", "measure"), "monitor": ("i48core.learn", "monitor"), "unexpected": ("i48as_two_stage", "unexpected"),
         "oddness": ("i48at_recheck", "oddness"), "recheck": ("i48at_recheck", "recheck")}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(name)
