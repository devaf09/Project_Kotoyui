"""I49zzg: 1 位と 2 位の点の差ごとの数え（小さな世界）。"""
import i49zzg_margin as G
from test_i49zw import attr as small


def test_margin_and_bins():
    a, b, d = G.margin({"x": 2.0, "y": 1.0, "z": 0.0}, ["x", "y", "z"])
    assert (a, b, d) == ("x", "y", 0.5) and G.bin_of(d) == "〜0.5"
    a, b, d = G.margin({"x": 1.0, "y": 1.0}, ["y", "x"])            # 同じ点なら、候補の並びで先の物が 1 位（本体と同じ）
    assert (a, b, d) == ("y", "x", 0.0) and G.bin_of(d) == "0（全く同じ）"
    assert G.bin_of(1e-7) == "〜0.000001" and G.bin_of(0.9) == "それより上"


def _talks():
    people, foods = ["太郎", "花子", "次郎"], ["りんご", "みかん", "パン"]
    out = []
    for i in range(12):
        p, f = (people[i % 3], "NOUN"), (foods[(i // 3) % 3], "NOUN")
        tris = [(p, f, ("食べる", "VERB")), (p, ("本", "NOUN"), ("読む", "VERB"))]
        out.append((tris, sorted({w for t in tris for w in t} | {("歌", "NOUN"), ("歌う", "VERB")}, key=str)))
    return out


def test_run_counts_every_answered_question_once():
    get = lambda w: small(w[0])
    d = G.run(get, get, _talks(), ks=(0,))["元の順"]
    n = sum(c["問い"] for c in d["全部の段"].values())
    assert n > 0 and n + d["数えない問い"] == 6 * 2 * 3                # 後半 6 話 × 3 つ組 2 つ × 席 3 つ
    assert n == sum(c["問い"] for lv in d["段ごと"].values() for c in lv.values())
    assert all(c["一位が当たり"] + c["二位が当たり"] <= c["問い"] for c in d["全部の段"].values())
