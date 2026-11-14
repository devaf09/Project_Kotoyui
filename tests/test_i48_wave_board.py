"""I48 の波の板の小さな確かめ（numpy だけで動く）。"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48_wave_board as W  # noqa: E402


def toy_works(n=12, lines=40, seed=0):
    rng = random.Random(seed)
    topics = [[f"語{t}_{i}" for i in range(12)] for t in range(6)]
    works = []
    for k in range(n):
        L = []
        for j in range(lines):
            t = topics[(k + j // 8) % len(topics)]
            ws = rng.sample(t, 3)
            L.append(dict(text="「" + " ".join(ws) + "」", toks=[dict(i=i, text=w, lemma=w, pos="NOUN", dep="dep", head=0) for i, w in enumerate(ws)]))
        works.append(dict(name=f"w{k}", lines=L))
    return works


def test_store_roundtrip():
    rng = np.random.default_rng(0)
    U = np.exp(1j * rng.uniform(0, 2 * np.pi, (5, 64)))
    for name, tol in (("c128", 1e-12), ("c64", 1e-6), ("p8", 2 * np.pi / 256)):
        st = W.Store(name, unit=True)
        assert np.abs(st.decode_np(st.encode(U)) - U).max() <= tol
        assert st.bytes_per(64) == {"c128": 1024, "c64": 512, "p8": 64}[name]
    X = rng.normal(size=(5, 64)) + 1j * rng.normal(size=(5, 64))
    st = W.Store("p8")
    assert np.abs(st.decode_np(st.encode(X)) - X).max() < 1e-2


def test_bank_top_matches_brute_force():
    rng = np.random.default_rng(1)
    K = rng.normal(size=(300, 32)) + 1j * rng.normal(size=(300, 32))
    Q = rng.normal(size=(7, 32)) + 1j * rng.normal(size=(7, 32))
    bank = W.Bank(K, W.Store("c128"), None, chunk=64)       # 区切りをまたいで上位を合わせる
    idx, val = bank.top(Q, 5)
    R = np.abs(K @ np.conj(Q).T).T
    for q in range(7):
        assert list(idx[q]) == list(np.argsort(-R[q])[:5])
        assert np.allclose(val[q], np.sort(R[q])[::-1][:5])


def test_wave_board_reads_like_table_and_layers_can_be_added():
    works = toy_works()
    bd = W.WaveBoard(D=2048, store="c64", wins=(1, 3), devices=[None], top=10).fit(works[2:])
    assert len(bd.banks) == 2
    bd.add_layer(10)                                         # 層を後から足す
    assert bd.wins == [1, 3, 10] and len(bd.banks) == 3 and bd.banks[2].N == len(bd.scenes)
    test = W.test_items(bd, works[:2], bd.wins)
    Q = [np.stack([bd.query(c[l]) for c, _ in test]) for l in range(3)]
    res = bd.scoop(Q)
    same = 0
    for qi in range(len(test)):
        S = res[0][0][qi] / res[0][1][qi]
        v = bd.read_dense(S[None])[0]                        # 波を語に戻す
        c = {}
        for j, w2 in zip(res[0][2][qi], res[0][3][qi]):      # 同じ場面から作る表の板
            for w in bd.nxt[j]:
                c[w] = c.get(w, 0.0) + w2 / res[0][1][qi]
        top_t = sorted(c, key=lambda w: -c[w])[:3]
        top_w = [bd.answer_vocab[j] for j in np.argsort(-v)[:3]]
        same += len(set(top_t) & set(top_w))
        for w in top_w[:1]:
            assert abs(v[bd.aix[w]] - c[w]) < 0.05           # 明るさも表とほぼ同じ
    assert same / (3 * len(test)) > 0.8                      # 明るさの並ぶ語があるので、全部は同じにならない


def test_wave_source_returns_lines():
    works = toy_works()
    bd = W.WaveBoard(D=1024, store="p8", wins=(1, 3), devices=[None], top=10, keep_dir=False).fit(works)
    src = W.WaveSource(bd, n=5)
    ctx = [[t["lemma"] for t in ln["toks"]] for ln in works[0]["lines"][:4]]
    out = src.lines(ctx)
    assert 1 <= len(out) <= 5 and out[0][0] == 1.0 and out[0][1].startswith("「")
    assert src.lines([["知らない語"]]) == []


def test_day_add_line_is_scooped_at_once_and_saved(tmp_path):
    works = toy_works()
    bd = W.WaveBoard(D=2048, store="c64", wins=(1, 3), devices=[None], top=10, keep_dir=False, spare=16).fit(works)
    n0, v0 = len(bd.scenes), len(bd.answer_vocab)
    tok = lambda ws: [dict(i=i, text=w, lemma=w, pos="NOUN", dep="dep", head=0) for i, w in enumerate(ws)]
    assert not bd.add_line("今日", "新語A 新語B", tok(["新語A", "新語B"]))      # 1 行目は場面にならない
    assert bd.add_line("今日", "新語C 新語D", tok(["新語C", "新語D"]))          # 知らない語だけの場面を足す
    assert len(bd.scenes) == n0 + 1 and len(bd.answer_vocab) == v0 and "新語C" in bd.aix
    res = bd.scoop([bd.query(["新語A", "新語B"])[None]] * 2)
    assert res[0][2][0][0] == n0                             # 足した場面がいちばん強く共鳴する
    v = bd.read_dense((res[0][0] / res[0][1][:, None]))[0]
    assert {bd.answer_vocab[j] for j in np.argsort(-v)[:2]} == {"新語C", "新語D"}   # 足した返事の語が読める
    out = W.WaveSource(bd, n=3).lines([["新語A", "新語B"]])
    assert out[0][1] == "「新語C 新語D」"
    bd.add_layer(10)                                         # 層を足しても、昼の場面は付いてくる
    assert len(bd.day_K[2]) == 1
    p = tmp_path / "day.jsonl"
    bd.save_day(str(p))
    import json
    day = [json.loads(x) for x in open(p, encoding="utf-8")]
    assert len(day) == 1 and len(day[0]["lines"]) == 2 and day[0]["lines"][0]["text"].startswith("「")


def test_night_verdict():
    k1, k2 = W.NIGHT_KEYS
    mk = lambda a, d: {"試しの組": 10000, k1[0]: {k1[1]: a}, k2[0]: {k2[1]: d}}
    assert W.night_verdict(mk(0.50, 0.50), mk(0.495, 0.51))["判定"] == "合流"      # 揺れの中
    assert W.night_verdict(mk(0.50, 0.50), mk(0.51, 0.40))["判定"] == "戻す"       # 向きが崩れた


def test_save_and_load_give_same_scoop(tmp_path):
    works = toy_works()
    for store in ("c64", "p8"):
        bd = W.WaveBoard(D=512, store=store, wins=(1, 3), devices=[None], top=10, spare=8).fit(works)
        p = str(tmp_path / ("board_" + store))
        bd.save(p)
        b2 = W.WaveBoard.load(p, works, devices=[None])
        Q = [np.stack([bd.query(["語0_1", "語0_2"]), bd.query(["語3_4"])])] * 2
        r1, r2 = bd.scoop(Q), b2.scoop(Q)
        assert (r1[1][2] == r2[1][2]).all() and np.allclose(r1[1][0], r2[1][0], atol=1e-4)
        assert np.allclose(bd.read_dense(r1[0][0]), b2.read_dense(r2[0][0]), atol=1e-5)
        tok = lambda ws: [dict(i=i, text=w, lemma=w, pos="NOUN", dep="dep", head=0) for i, w in enumerate(ws)]
        b2.add_line("今日", "新語A", tok(["新語A"]))
        assert b2.add_line("今日", "新語B", tok(["新語B"]))       # 読み込んだ板にも昼の足しができる
