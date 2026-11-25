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


def test_vessel_pick_moves_only_inside_the_tie_band():
    import numpy as np
    dirs = dict(x=np.array([1, 0], dtype=np.complex64), y=np.array([0, 1], dtype=np.complex64), z=None)
    state = np.array([0, 1], dtype=np.complex64)                       # 器は y の向き
    pts = {"x": 1.0, "y": 0.95, "z": 0.99}
    assert G.vessel_pick(pts, ["x", "y", "z"], 0.01, state, dirs.get) == "x"      # 幅の中に、向きを持つ候補が 1 位だけ
    assert G.vessel_pick(pts, ["x", "y", "z"], 0.1, state, dirs.get) == "y"       # 幅を広げると、器に近い y に変わる
    assert G.vessel_pick(pts, ["x", "y", "z"], 0.1, state, dirs.get, far=True) == "x"
    assert G.vessel_pick(pts, ["x", "y", "z"], 0.1, None, dirs.get) == "x"        # 器がまだ空なら 1 位のまま
    assert G.vessel_pick({"z": 1.0, "y": 1.0}, ["z", "y"], 0.1, state, dirs.get) == "z"   # 1 位が向きを持たなければ変えない


def test_run_vessel_small_world():
    get = lambda w: small(w[0])
    n = len(get(("太郎", "NOUN")))
    d = G.run_vessel(get, get, _talks(), n, ks=(0,))["元の順"]["幅ごと"]
    names = [name for _, name in G.BINS]
    assert list(d) == names and len({c["問い"] for c in d.values()}) == 1
    ch = [d[x]["答えが変わった"] for x in names]
    assert ch == sorted(ch)                                             # 幅を広げるほど、変わる答えは減らない
    assert all(c["当たりに変わった"] + c["外れに変わった"] <= c["答えが変わった"] for c in d.values())
