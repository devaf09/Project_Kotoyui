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
