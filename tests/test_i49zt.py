"""I49zt の関門: 選び直しを、切り出しを通さない材料で数え直す道具。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import V  # noqa: E402

LINES = ["友達が魚を釣る", "先生が魚を釣る", "友達が海を泳ぐ", "友達が魚を見る", "先生がゲームを見る", "",
         "先生が海を泳ぐ", "友達がゲームを釣る", "先生が魚を釣る", "友達が海を見る"]


def test_seen_flags_line_up_with_questions():
    import i49za_fold_run as ZR
    import i49zp_text_talks_run as ZP
    import i49zt_repick_clean_run as T
    talks, _, _ = ZP.talks_of(LINES)
    seq = [(t[0], t[1]) for t in talks]
    flags = T.seen_flags(seq)

    class Zero:
        def ask(self, si, q, cands):
            return dict(答え=None, 点={})

        def learn(self, tri):
            pass

        def tick(self):
            pass
    assert len(flags) == len(ZR.stream(Zero(), seq, {}))
    assert flags[0] == (False, False) and any(f[0] for f in flags) and all(f[1] or not f[0] for f in flags)


def test_run_shapes():
    import i49zp_text_talks_run as ZP
    import i49zt_repick_clean_run as T
    talks, _, _ = ZP.talks_of(LINES)
    get = lambda w: V.get(w[0])
    res = T.run(get, get, talks, ks=(0, 1))
    for d in res.values():
        assert 0 <= d["正解が同じ席でもう出た"] <= d["正解がどの席でも出た"] <= 1
        for name, _, _ in T.VERS:
            assert 0 <= d[name]["正しい物"] <= 1
            assert d[name]["正解がもう出た問い"]["問い"] + d[name]["正解がまだ出ていない問い"]["問い"] == d["問い"]
        assert len(d["直の比べ"]) == len(T.DUELS)
